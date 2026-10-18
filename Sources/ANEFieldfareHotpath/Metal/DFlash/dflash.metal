#include <metal_stdlib>
using namespace metal;

// DFlash block-diffusion drafter (z-lab, Qwen3-style layers). Activations are
// BF16 to match the drafter's training dtype; the residual stream is FP32.
// GEMMs run on the BF16 register-tiled GEMM; these kernels are the glue.

constant constexpr uint kDFlashHeadDim = 128;

/// Tied INT4 embedding rows, scaled: out[r][i] = E[ids[r]][i] * scale (FP32).
kernel void dflash_embed_int4(
    device const uint8_t* table    [[buffer(0)]],
    device const bfloat*  scales   [[buffer(1)]],
    device const bfloat*  biases   [[buffer(2)]],
    device const uint*    ids      [[buffer(3)]],
    device float*         out      [[buffer(4)]],
    constant uint&        D        [[buffer(5)]],
    constant float&       scale    [[buffer(6)]],
    uint2                 gid      [[thread_position_in_grid]]
) {
    const uint i = gid.x, r = gid.y;
    if (i >= D) return;
    const uint token = ids[r];
    const uint groups = D / 64u;
    const uint8_t byte = table[token * (D / 2u) + (i >> 1)];
    const uint q = (i & 1u) ? uint(byte >> 4) : uint(byte & 0xFu);
    const float s = float(scales[token * groups + i / 64u]);
    const float b = float(biases[token * groups + i / 64u]);
    out[r * D + i] = (float(q) * s + b) * scale;
}

/// Row RMSNorm with a BF16 weight (x * inv_rms * w), one threadgroup per row.
template <typename T>
inline void dflash_rmsnorm_rows_impl(device const T* x, device const bfloat* w, device bfloat* out,
                                     uint D, float eps, uint row, uint lid, uint lsize,
                                     uint simd_lane, uint simd_group, threadgroup float* partial) {
    device const T* xr = x + row * D;
    float acc = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        const float v = float(xr[i]);
        acc += v * v;
    }
    acc = simd_sum(acc);
    if (simd_lane == 0) partial[simd_group] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        float v = simd_lane < (lsize + 31) / 32 ? partial[simd_lane] : 0.0f;
        v = simd_sum(v);
        if (simd_lane == 0) partial[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = rsqrt(partial[0] / float(D) + eps);
    for (uint i = lid; i < D; i += lsize) {
        out[row * D + i] = bfloat(float(xr[i]) * inv * float(w[i]));
    }
}

kernel void dflash_rmsnorm_rows_f32(
    device const float*  x     [[buffer(0)]],
    device const bfloat* w     [[buffer(1)]],
    device bfloat*       out   [[buffer(2)]],
    constant uint&       D     [[buffer(3)]],
    constant float&      eps   [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float partial[32];
    dflash_rmsnorm_rows_impl(x, w, out, D, eps, row, lid, lsize, simd_lane, simd_group, partial);
}

kernel void dflash_rmsnorm_rows_bf16(
    device const bfloat* x     [[buffer(0)]],
    device const bfloat* w     [[buffer(1)]],
    device bfloat*       out   [[buffer(2)]],
    constant uint&       D     [[buffer(3)]],
    constant float&      eps   [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float partial[32];
    dflash_rmsnorm_rows_impl(x, w, out, D, eps, row, lid, lsize, simd_lane, simd_group, partial);
}

/// Per-head RMSNorm (BF16 weight over 128 lanes) followed by NeoX RoPE.
/// Row `row` is head `row % heads` of token `row / heads`, at absolute
/// position `pos0 + row / heads`. One SIMD group per row; lane l owns
/// elements {2l, 2l+1} and {64+2l, 64+2l+1}.
kernel void dflash_head_norm_rope(
    device const bfloat* x      [[buffer(0)]],
    device const bfloat* w      [[buffer(1)]],
    device bfloat*       out    [[buffer(2)]],
    constant uint&       heads  [[buffer(3)]],
    constant uint&       pos0   [[buffer(4)]],
    constant float&      theta  [[buffer(5)]],
    constant float&      eps    [[buffer(6)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    device const bfloat* xr = x + row * kDFlashHeadDim;
    const uint a = 2u * lane, b = 64u + 2u * lane;
    float v[4] = {float(xr[a]), float(xr[a + 1]), float(xr[b]), float(xr[b + 1])};
    float ss = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
    ss = simd_sum(ss);
    const float inv = rsqrt(ss / float(kDFlashHeadDim) + eps);
    // The reference normalizes in float and rounds to BF16 before RoPE.
    v[0] = float(bfloat(v[0] * inv * float(w[a])));
    v[1] = float(bfloat(v[1] * inv * float(w[a + 1])));
    v[2] = float(bfloat(v[2] * inv * float(w[b])));
    v[3] = float(bfloat(v[3] * inv * float(w[b + 1])));
    const float pos = float(pos0 + row / heads);
    device bfloat* o = out + row * kDFlashHeadDim;
    for (uint j = 0; j < 2; ++j) {
        const uint i = a + j;  // frequency index, 0..63
        const float freq = pos * pow(theta, -float(2u * i) / float(kDFlashHeadDim));
        const float c = cos(freq), s = sin(freq);
        const float x1 = v[j], x2 = v[2 + j];
        o[i] = bfloat(x1 * c - x2 * s);
        o[i + 64u] = bfloat(x2 * c + x1 * s);
    }
}

/// Block attention: R queries at positions [pos0, pos0+R) against context keys
/// [0, ctxLen) from the cache and the block's own R keys. Sliding layers are
/// causal inside the block and windowed; the full layer sees the whole block.
/// One SIMD group per (query, head); lane l owns dims [4l, 4l+4).
kernel void dflash_block_attention(
    device const bfloat* q         [[buffer(0)]],   // [R][H][128]
    device const bfloat* kCache    [[buffer(1)]],   // [ctx][KV][128]
    device const bfloat* vCache    [[buffer(2)]],
    device const bfloat* kBlock    [[buffer(3)]],   // [R][KV][128]
    device const bfloat* vBlock    [[buffer(4)]],
    device bfloat*       out       [[buffer(5)]],   // [R][H][128]
    constant uint&       H         [[buffer(6)]],
    constant uint&       KV        [[buffer(7)]],
    constant uint&       ctxLen    [[buffer(8)]],
    constant uint&       R         [[buffer(9)]],
    constant uint&       sliding   [[buffer(10)]],
    constant uint&       window    [[buffer(11)]],
    uint2 tg   [[threadgroup_position_in_grid]],    // x = head, y = query
    uint lane  [[thread_index_in_simdgroup]]
) {
    const uint h = tg.x, r = tg.y;
    const uint kvh = h / (H / KV);
    const float scale = rsqrt(float(kDFlashHeadDim));
    device const bfloat* qr = q + (r * H + h) * kDFlashHeadDim + 4u * lane;
    const float4 qv = float4(float(qr[0]), float(qr[1]), float(qr[2]), float(qr[3])) * scale;
    const uint qpos = ctxLen + r;  // context covers positions [0, ctxLen)
    float m = -INFINITY, l = 0.0f;
    float4 acc = 0.0f;
    const uint first = (sliding != 0u && qpos + 1u > window) ? qpos + 1u - window : 0u;
    for (uint j = first; j < ctxLen; ++j) {
        device const bfloat* kr = kCache + (j * KV + kvh) * kDFlashHeadDim + 4u * lane;
        const float s = simd_sum(dot(qv, float4(float(kr[0]), float(kr[1]), float(kr[2]), float(kr[3]))));
        const float mn = max(m, s);
        const float c = exp(m - mn), p = exp(s - mn);
        device const bfloat* vr = vCache + (j * KV + kvh) * kDFlashHeadDim + 4u * lane;
        acc = acc * c + p * float4(float(vr[0]), float(vr[1]), float(vr[2]), float(vr[3]));
        l = l * c + p;
        m = mn;
    }
    const uint last = sliding != 0u ? r + 1u : R;
    for (uint j = 0; j < last; ++j) {
        device const bfloat* kr = kBlock + (j * KV + kvh) * kDFlashHeadDim + 4u * lane;
        const float s = simd_sum(dot(qv, float4(float(kr[0]), float(kr[1]), float(kr[2]), float(kr[3]))));
        const float mn = max(m, s);
        const float c = exp(m - mn), p = exp(s - mn);
        device const bfloat* vr = vBlock + (j * KV + kvh) * kDFlashHeadDim + 4u * lane;
        acc = acc * c + p * float4(float(vr[0]), float(vr[1]), float(vr[2]), float(vr[3]));
        l = l * c + p;
        m = mn;
    }
    acc /= l;
    device bfloat* o = out + (r * H + h) * kDFlashHeadDim + 4u * lane;
    o[0] = bfloat(acc.x); o[1] = bfloat(acc.y); o[2] = bfloat(acc.z); o[3] = bfloat(acc.w);
}

/// out = silu(gate) * up, BF16.
kernel void dflash_silu_mul(
    device const bfloat* gate [[buffer(0)]],
    device const bfloat* up   [[buffer(1)]],
    device bfloat*       out  [[buffer(2)]],
    constant uint&       n    [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= n) return;
    const float g = float(gate[i]);
    out[i] = bfloat(float(bfloat(g / (1.0f + exp(-g)))) * float(up[i]));
}

/// residual += delta (FP32 += BF16).
kernel void dflash_residual_add(
    device float*        residual [[buffer(0)]],
    device const bfloat* delta    [[buffer(1)]],
    constant uint&       n        [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= n) return;
    residual[i] += float(delta[i]);
}

/// FP32 rows -> FP16 rows, for the INT4 greedy head.
kernel void dflash_f32_to_f16(
    device const float* x   [[buffer(0)]],
    device half*        out [[buffer(1)]],
    constant uint&      n   [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= n) return;
    out[i] = half(x[i]);
}

/// Captured target layers `[layers][rows][D]` FP16 -> fused feature rows
/// `[rows][layers * D]` BF16, the drafter's `fc` input.
kernel void dflash_gather_context(
    device const half* captured [[buffer(0)]],
    device bfloat*     out      [[buffer(1)]],
    constant uint&     rows     [[buffer(2)]],
    constant uint&     layers   [[buffer(3)]],
    constant uint&     D        [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]   // x = layer * D + i, y = row
) {
    const uint col = gid.x, row = gid.y;
    if (col >= layers * D || row >= rows) return;
    const uint layer = col / D, i = col % D;
    out[row * layers * D + col] = bfloat(captured[(layer * rows + row) * D + i]);
}
