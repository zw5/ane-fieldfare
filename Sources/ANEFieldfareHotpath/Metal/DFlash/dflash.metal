#include <metal_stdlib>
using namespace metal;

// DFlash block-diffusion drafter (z-lab, Qwen3-style layers). Activations are
// BF16 to match the drafter's training dtype; the residual stream is FP32.
// Projections run on the INT4 matrix kernels; these kernels are the glue.
//
// A draft's context length lives on the GPU in `DFlashState`, so a draft can
// be queued behind the verify whose acceptance sets it: `dflash_accept`
// reads the verified tokens, advances the context and writes the next
// block's anchor without a host round trip.

constant constexpr uint kDFlashHeadDim = 128;
constant constexpr uint kDFlashMaxBlock = 16;

struct DFlashState {
    uint context;   // positions in the drafter's K/V cache
    uint accepted;  // drafts the last `dflash_accept` kept
    uint pad0;
    uint pad1;
};

/// Tied Q4_0 embedding rows of the block's tokens, scaled:
/// out[r][i] = E[block[r]][i] * scale (FP32).
kernel void dflash_embed_block(
    device const uint8_t* table    [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const uint*    block    [[buffer(2)]],
    device float*         out      [[buffer(3)]],
    constant uint&        D        [[buffer(4)]],
    constant float&       scale    [[buffer(5)]],
    uint2                 gid      [[thread_position_in_grid]]
) {
    const uint i = gid.x, r = gid.y;
    if (i >= D) return;
    const uint token = block[r];
    const uint groups = D / 32u;
    const uint8_t byte = table[token * (D / 2u) + (i >> 1)];
    const uint q = (i & 1u) ? uint(byte >> 4) : uint(byte & 0xFu);
    const float s = float(scales[token * groups + i / 32u]);
    out[r * D + i] = (float(q) - 8.0f) * s * scale;
}

/// 1/rms of row `xr` (D elements), with `partial` as threadgroup scratch.
template <typename T>
static inline float dflash_row_inv_rms(device const T* xr, uint D, float eps, uint lid, uint lsize,
                                       uint simd_lane, uint simd_group, threadgroup float* partial) {
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
    return rsqrt(partial[0] / float(D) + eps);
}

/// Row RMSNorm with a BF16 weight (x * inv_rms * w), one threadgroup per row.
template <typename T>
static inline void dflash_rmsnorm_row(device const T* x, device const bfloat* w, device bfloat* out,
                                      uint D, float eps, uint row, uint lid, uint lsize,
                                      uint simd_lane, uint simd_group, threadgroup float* partial) {
    device const T* xr = x + row * D;
    const float inv = dflash_row_inv_rms(xr, D, eps, lid, lsize, simd_lane, simd_group, partial);
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
    dflash_rmsnorm_row(x, w, out, D, eps, row, lid, lsize, simd_lane, simd_group, partial);
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
    dflash_rmsnorm_row(x, w, out, D, eps, row, lid, lsize, simd_lane, simd_group, partial);
}

/// residual += delta, then normed = rmsnorm(residual) * w: the residual add
/// and the next norm in one pass per row. Each thread re-reads only the
/// elements it wrote, so this is bitwise the add followed by
/// `dflash_rmsnorm_rows_f32`.
kernel void dflash_residual_add_norm(
    device float*        residual [[buffer(0)]],
    device const bfloat* delta    [[buffer(1)]],
    device const bfloat* w        [[buffer(2)]],
    device bfloat*       out      [[buffer(3)]],
    constant uint&       D        [[buffer(4)]],
    constant float&      eps      [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float partial[32];
    device float* xr = residual + row * D;
    for (uint i = lid; i < D; i += lsize) xr[i] += float(delta[row * D + i]);
    dflash_rmsnorm_row(residual, w, out, D, eps, row, lid, lsize, simd_lane, simd_group, partial);
}

/// The last layer's residual add, written as FP16 for the INT4 greedy head:
/// out[r - 1] = residual[r] + delta[r] for rows 1..R-1, the rows that predict
/// drafts.
kernel void dflash_residual_add_head_rows(
    device const float*  residual [[buffer(0)]],
    device const bfloat* delta    [[buffer(1)]],
    device half*         out      [[buffer(2)]],
    constant uint&       n        [[buffer(3)]],
    constant uint&       D        [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= n) return;
    out[i] = half(residual[D + i] + float(delta[D + i]));
}

/// Per-head RMSNorm (BF16 weight over 128 lanes) and NeoX RoPE of one head,
/// in float, rounded to BF16 before the rotation as the reference does. One
/// SIMD group; lane l owns elements {2l, 2l+1} and {64+2l, 64+2l+1}.
static inline void dflash_norm_rope_head(device const bfloat* x, device const bfloat* w, device bfloat* o,
                                         float pos, float theta, float eps, uint lane) {
    const uint a = 2u * lane, b = 64u + 2u * lane;
    float v[4] = {float(x[a]), float(x[a + 1]), float(x[b]), float(x[b + 1])};
    float ss = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
    ss = simd_sum(ss);
    const float inv = rsqrt(ss / float(kDFlashHeadDim) + eps);
    v[0] = float(bfloat(v[0] * inv * float(w[a])));
    v[1] = float(bfloat(v[1] * inv * float(w[a + 1])));
    v[2] = float(bfloat(v[2] * inv * float(w[b])));
    v[3] = float(bfloat(v[3] * inv * float(w[b + 1])));
    for (uint j = 0; j < 2; ++j) {
        const uint i = a + j;  // frequency index, 0..63
        const float freq = pos * pow(theta, -float(2u * i) / float(kDFlashHeadDim));
        const float c = cos(freq), s = sin(freq);
        const float x1 = v[j], x2 = v[2 + j];
        o[i] = bfloat(x1 * c - x2 * s);
        o[i + 64u] = bfloat(x2 * c + x1 * s);
    }
}

/// Q and K normalization and RoPE of a draft block, in place in the fused
/// Q/K/V rows `[R][H + 2 KV][128]`: threadgroup (j, r) is head j of row r,
/// a Q head for j < H, else K head j - H. Row r sits at position
/// state.context + r.
kernel void dflash_block_qk_norm_rope(
    device bfloat*            qkv     [[buffer(0)]],
    device const bfloat*      qNorm   [[buffer(1)]],
    device const bfloat*      kNorm   [[buffer(2)]],
    device const DFlashState& state   [[buffer(3)]],
    constant uint&            H       [[buffer(4)]],
    constant uint&            KV      [[buffer(5)]],
    constant float&           theta   [[buffer(6)]],
    constant float&           eps     [[buffer(7)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint lane  [[thread_index_in_simdgroup]]
) {
    const uint j = tg.x, r = tg.y;
    device bfloat* head = qkv + (r * (H + 2u * KV) + j) * kDFlashHeadDim;
    dflash_norm_rope_head(head, j < H ? qNorm : kNorm, head, float(state.context + r), theta, eps, lane);
}

/// The context update's K/V for `n` rows at positions p0.., from fused K|V
/// projection rows `[n][2 KV][128]`: K is normalized and rotated, V copied,
/// both into cache slot `position % capacity`. Threadgroup (h, r) is KV
/// head h of row r.
kernel void dflash_store_context_kv(
    device const bfloat* kv       [[buffer(0)]],
    device const bfloat* kNorm    [[buffer(1)]],
    device bfloat*       kCache   [[buffer(2)]],
    device bfloat*       vCache   [[buffer(3)]],
    constant uint&       KV       [[buffer(4)]],
    constant uint&       p0       [[buffer(5)]],
    constant uint&       capacity [[buffer(6)]],
    constant float&      theta    [[buffer(7)]],
    constant float&      eps      [[buffer(8)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint lane  [[thread_index_in_simdgroup]]
) {
    const uint h = tg.x, r = tg.y;
    const uint position = p0 + r;
    const uint slot = position % capacity;
    device const bfloat* row = kv + r * 2u * KV * kDFlashHeadDim;
    dflash_norm_rope_head(row + h * kDFlashHeadDim, kNorm, kCache + (slot * KV + h) * kDFlashHeadDim,
                          float(position), theta, eps, lane);
    device const bfloat* v = row + (KV + h) * kDFlashHeadDim;
    device bfloat* vo = vCache + (slot * KV + h) * kDFlashHeadDim;
    for (uint i = lane; i < kDFlashHeadDim; i += 32u) vo[i] = v[i];
}

/// Block attention: R queries at positions [ctx, ctx + R) against context
/// keys [0, ctx) from the cache ring (slot = position % capacity) and the
/// block's own R keys, ctx = state.context. Sliding layers are causal inside
/// the block and windowed; the full layer sees the whole block. One SIMD
/// group per (query, head); lane l owns dims [4l, 4l + 4).
kernel void dflash_block_attention(
    device const bfloat*      qkv      [[buffer(0)]],   // [R][H + 2 KV][128]
    device const bfloat*      kCache   [[buffer(1)]],   // [capacity][KV][128]
    device const bfloat*      vCache   [[buffer(2)]],
    device bfloat*            out      [[buffer(3)]],   // [R][H][128]
    device const DFlashState& state    [[buffer(4)]],
    constant uint&            H        [[buffer(5)]],
    constant uint&            KV       [[buffer(6)]],
    constant uint&            R        [[buffer(7)]],
    constant uint&            sliding  [[buffer(8)]],
    constant uint&            window   [[buffer(9)]],
    constant uint&            capacity [[buffer(10)]],
    uint2 tg   [[threadgroup_position_in_grid]],    // x = head, y = query
    uint lane  [[thread_index_in_simdgroup]]
) {
    const uint h = tg.x, r = tg.y;
    const uint kvh = h / (H / KV);
    const uint rowWidth = (H + 2u * KV) * kDFlashHeadDim;
    const uint ctxLen = state.context;
    const float scale = rsqrt(float(kDFlashHeadDim));
    device const bfloat* qr = qkv + r * rowWidth + h * kDFlashHeadDim + 4u * lane;
    const float4 qv = float4(float(qr[0]), float(qr[1]), float(qr[2]), float(qr[3])) * scale;
    const uint qpos = ctxLen + r;
    float m = -INFINITY, l = 0.0f;
    float4 acc = 0.0f;
    const uint first = (sliding != 0u && qpos + 1u > window) ? qpos + 1u - window : 0u;
    for (uint j = first; j < ctxLen; ++j) {
        const uint slot = j % capacity;
        device const bfloat* kr = kCache + (slot * KV + kvh) * kDFlashHeadDim + 4u * lane;
        const float s = simd_sum(dot(qv, float4(float(kr[0]), float(kr[1]), float(kr[2]), float(kr[3]))));
        const float mn = max(m, s);
        const float c = exp(m - mn), p = exp(s - mn);
        device const bfloat* vr = vCache + (slot * KV + kvh) * kDFlashHeadDim + 4u * lane;
        acc = acc * c + p * float4(float(vr[0]), float(vr[1]), float(vr[2]), float(vr[3]));
        l = l * c + p;
        m = mn;
    }
    const uint last = sliding != 0u ? r + 1u : R;
    for (uint j = 0; j < last; ++j) {
        device const bfloat* kr = qkv + j * rowWidth + (H + kvh) * kDFlashHeadDim + 4u * lane;
        const float s = simd_sum(dot(qv, float4(float(kr[0]), float(kr[1]), float(kr[2]), float(kr[3]))));
        const float mn = max(m, s);
        const float c = exp(m - mn), p = exp(s - mn);
        device const bfloat* vr = qkv + j * rowWidth + (H + KV + kvh) * kDFlashHeadDim + 4u * lane;
        acc = acc * c + p * float4(float(vr[0]), float(vr[1]), float(vr[2]), float(vr[3]));
        l = l * c + p;
        m = mn;
    }
    acc /= l;
    device bfloat* o = out + (r * H + h) * kDFlashHeadDim + 4u * lane;
    o[0] = bfloat(acc.x); o[1] = bfloat(acc.y); o[2] = bfloat(acc.z); o[3] = bfloat(acc.w);
}

/// act[r][j] = silu(gate[r][j]) * up[r][j], from fused gate|up rows
/// `[R][2F]`, BF16.
kernel void dflash_silu_mul(
    device const bfloat* gateUp [[buffer(0)]],
    device bfloat*       out    [[buffer(1)]],
    constant uint&       F      [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint j = gid.x, r = gid.y;
    if (j >= F) return;
    const float g = float(gateUp[r * 2u * F + j]);
    out[r * F + j] = bfloat(float(bfloat(g / (1.0f + exp(-g)))) * float(gateUp[r * 2u * F + F + j]));
}

/// The target's captured layers for positions [p0, p0 + n), from the feature
/// ring `[layers][capacity][D]` FP16 (slot = position % capacity), as fused
/// feature rows `[n][layers * D]` BF16: the drafter's `fc` input.
kernel void dflash_gather_features(
    device const half* ring     [[buffer(0)]],
    device bfloat*     out      [[buffer(1)]],
    constant uint&     n        [[buffer(2)]],
    constant uint&     layers   [[buffer(3)]],
    constant uint&     D        [[buffer(4)]],
    constant uint&     p0       [[buffer(5)]],
    constant uint&     capacity [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]   // x = layer * D + i, y = row
) {
    const uint col = gid.x, row = gid.y;
    if (col >= layers * D || row >= n) return;
    const uint layer = col / D, i = col % D;
    const uint slot = (p0 + row) % capacity;
    out[row * layers * D + col] = bfloat(ring[(layer * capacity + slot) * D + i]);
}

/// Writes the draft head's picks into the block after its anchor: draft i is
/// `ids[picks[i]]` when the head scored a listed vocabulary, else `picks[i]`.
kernel void dflash_finish_block(
    device const uint* picks  [[buffer(0)]],
    device const uint* ids    [[buffer(1)]],
    device uint*       block  [[buffer(2)]],
    constant uint&     drafts [[buffer(3)]],
    constant uint&     listed [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= drafts) return;
    block[1u + i] = listed != 0u ? ids[picks[i]] : picks[i];
}

/// After a verify of `rows` rows of `block` (anchor and drafts) whose context
/// rows the update before this wrote: keeps the drafts the target agrees
/// with, advances the context past the anchor and them, and makes the
/// target's next token the anchor of the next block. One thread.
kernel void dflash_accept(
    device uint*        block    [[buffer(0)]],
    device const uint*  verified [[buffer(1)]],
    device DFlashState& state    [[buffer(2)]],
    constant uint&      rows     [[buffer(3)]],
    constant uint&      mask     [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;
    uint n = 0;
    while (n + 1u < rows && block[1u + n] == verified[n]) ++n;
    state.accepted = n;
    state.context += n + 1u;
    block[0] = verified[n];
    for (uint i = 1; i < kDFlashMaxBlock; ++i) block[i] = mask;
}
