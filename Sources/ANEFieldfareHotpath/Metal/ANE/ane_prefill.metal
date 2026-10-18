#include <metal_stdlib>
using namespace metal;
struct ANEPair { uint token, expert, rank, weight; };
struct ANEJobs { uint start, count; };
struct ANEBlobs { const device uchar *blob[4]; };
static inline half ane_affine(const device uchar *blob, uint index, uint w, uint s, uint b) {
    uint code = (blob[w + index / 2] >> ((index & 1) * 4)) & 15;
    float scale = as_type<float>(uint(*reinterpret_cast<const device ushort *>(blob + s + (index / 64) * 2)) << 16);
    float bias = as_type<float>(uint(*reinterpret_cast<const device ushort *>(blob + b + (index / 64) * 2)) << 16);
    return half(float(code) * scale + bias);
}
// Coalesced compact reads followed by a tile transpose into ANE's [D,2176]
// runtime-input carrier. Every weight is rewritten before reuse of a bank.
kernel void ane_expert_weights(constant ANEBlobs &blobs [[buffer(0)]], device half *carrier [[buffer(1)]],
    constant uint *off [[buffer(2)]], const device ANEJobs *jobs [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    carrier += group.z * 2816 * 2176;
    if (jobs[group.z].count == 0) {
        for (uint row = lane / 32; row < 32; row += 8) {
            uint d = group.x * 32 + row, f = group.y * 32 + lane % 32;
            carrier[d * 2176 + 64 + f] = 0;
            carrier[d * 2176 + 768 + f] = 0;
            carrier[d * 2176 + 1472 + f] = 0;
        }
        return;
    }
    const device uchar *blob = blobs.blob[group.z];
    threadgroup half gate[32][32], up[32][32];
    for (uint row = lane / 32; row < 32; row += 8) {
        uint f = group.y * 32 + row, d = group.x * 32 + lane % 32;
        gate[row][lane % 32] = ane_affine(blob, f * 2816 + d, off[0], off[1], off[2]);
        up[row][lane % 32] = ane_affine(blob, f * 2816 + d, off[3], off[4], off[5]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = lane / 32; row < 32; row += 8) {
        uint d = group.x * 32 + row, f = group.y * 32 + lane % 32;
        carrier[d * 2176 + 64 + f] = gate[lane % 32][row];
        carrier[d * 2176 + 768 + f] = up[lane % 32][row];
        carrier[d * 2176 + 1472 + f] = ane_affine(blob, d * 704 + f, off[6], off[7], off[8]);
    }
}
kernel void ane_expert_inputs(const device half *hidden [[buffer(0)]], device half *carrier [[buffer(1)]],
    const device ANEPair *pairs [[buffer(2)]], const device ANEJobs *jobs [[buffer(3)]],
    device atomic_uint *invalid [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint slot = i / (2816 * 64), d = (i / 64) % 2816, p = i % 64;
    ANEJobs job = jobs[slot];
    half value = p < job.count ? hidden[pairs[job.start + p].token * 2816 + d] * half(64) : half(0);
    if (!isfinite(value)) atomic_store_explicit(invalid, 1, memory_order_relaxed);
    carrier[(slot * 2816 + d) * 2176 + p] = value;
}
kernel void ane_expert_outputs(const device float *output [[buffer(0)]], device half *partials [[buffer(1)]],
    const device ANEPair *pairs [[buffer(2)]], const device ANEJobs *jobs [[buffer(3)]],
    device atomic_uint *invalid [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint slot = i / (2816 * 64), d = i % 2816, p = (i / 2816) % 64;
    ANEJobs job = jobs[slot];
    if (p >= job.count) return;
    ANEPair pair = pairs[job.start + p];
    half value = half(output[(slot * 2816 + d) * 64 + p]);
    if (!isfinite(value)) atomic_store_explicit(invalid, 1, memory_order_relaxed);
    partials[(pair.token * 8 + pair.rank) * 2816 + d] = value;
}
