#include <metal_stdlib>
using namespace metal;

#if defined(__HAVE_TENSOR__)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
#endif

constant constexpr uint kPrefillGroupSize = 64;
constant constexpr uint kPrefillRmsMaxSimdGroups = 8;
constant constexpr uint kPrefillPostMaxD = 4096;
constant constexpr uint kPrefillRouterMaxExperts = 256;
constant constexpr uint kPrefillRouterMaxTopK = 64;
constant constexpr uint kPrefillAttentionMaxSimdGroups = 16;
constant constexpr uint kPrefillMaxTileExperts = 16;
constant constexpr float kPrefillGeluSqrt2OverPi = 0.7978845608028654f;
constant constexpr float kPrefillGeluCubicCoeff = 0.044715f;
constant uint FC_PREFILL_KV_RING_CAP [[function_constant(76)]];

static inline float prefill_gelu_pytorch_tanh(float x) {
    const float x3 = x * x * x;
    float inner = kPrefillGeluSqrt2OverPi * (x + kPrefillGeluCubicCoeff * x3);
    inner = clamp(inner, -20.0f, 20.0f);
    return 0.5f * x * (1.0f + tanh(inner));
}
kernel void prefill_embed_lookup_int4_block(
    device const uint8_t* table     [[buffer(0)]],
    device const bfloat*  scales    [[buffer(1)]],
    device const bfloat*  biases    [[buffer(2)]],
    device const uint*    tokens    [[buffer(3)]],
    device half*          out       [[buffer(4)]],
    constant uint&        T         [[buffer(5)]],
    constant uint&        D         [[buffer(6)]],
    constant float&       out_scale [[buffer(7)]],
    uint2                 gid       [[thread_position_in_grid]]
) {
    const uint d = gid.x;
    const uint t = gid.y;
    if (t >= T || d >= D) return;

    const uint token = tokens[t];
    const uint groups_per_row = D / kPrefillGroupSize;
    device const uint8_t* row_q = table  + token * (D / 2u);
    device const bfloat*  row_s = scales + token * groups_per_row;
    device const bfloat*  row_b = biases + token * groups_per_row;

    const uint8_t byte = row_q[d >> 1];
    const uint q = (d & 1u) == 0u ? uint(byte & 0x0Fu) : uint(byte >> 4);
    const float s = float(row_s[d / kPrefillGroupSize]);
    const float b = float(row_b[d / kPrefillGroupSize]);
    out[t * D + d] = half((float(q) * s + b) * out_scale);
}

static inline float prefill_rms_block_inv(
    device const half* x,
    uint D,
    float eps,
    uint lid,
    uint lsize,
    uint simd_lane_id,
    uint simd_group_id,
    uint simdgroups,
    threadgroup float* partial
) {
    float acc = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        float v = float(x[i]);
        acc = fma(v, v, acc);
    }
    acc = simd_sum(acc);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        float v = (simd_lane_id < simdgroups) ? partial[simd_lane_id] : 0.0f;
        v = simd_sum(v);
        if (simd_lane_id == 0) {
            partial[0] = rsqrt(v / float(D) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return partial[0];
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void prefill_rmsnorm_bf16w_block(
    device const half*   x       [[buffer(0)]],
    device const bfloat* weight  [[buffer(1)]],
    device half*         out     [[buffer(2)]],
    constant uint&       T       [[buffer(3)]],
    constant uint&       D       [[buffer(4)]],
    constant float&      eps     [[buffer(5)]],
    uint                 row     [[threadgroup_position_in_grid]],
    uint                 lid     [[thread_position_in_threadgroup]],
    uint                 lsize   [[threads_per_threadgroup]],
    uint                 lane    [[thread_index_in_simdgroup]],
    uint                 sg      [[simdgroup_index_in_threadgroup]],
    uint                 sgs     [[simdgroups_per_threadgroup]]
) {
    if (row >= T) return;
    threadgroup float partial[kPrefillRmsMaxSimdGroups];
    device const half* xr = x + row * D;
    device half* yr = out + row * D;
    const float inv = prefill_rms_block_inv(xr, D, eps, lid, lsize, lane, sg, sgs, partial);

    for (uint i = lid; i < D; i += lsize) {
        yr[i] = half(float(xr[i]) * inv * float(weight[i]));
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void prefill_rmsnorm_bf16w_perhead_block(
    device const half*   x                   [[buffer(0)]],
    device const bfloat* weight              [[buffer(1)]],
    device half*         out                 [[buffer(2)]],
    constant uint&       T                   [[buffer(3)]],
    constant uint&       head_dim            [[buffer(4)]],
    constant uint&       num_heads           [[buffer(5)]],
    constant uint&       token_stride_elems  [[buffer(6)]],
    constant float&      eps                 [[buffer(7)]],
    uint3                tg                  [[threadgroup_position_in_grid]],
    uint3                lid3                [[thread_position_in_threadgroup]],
    uint3                lsize3              [[threads_per_threadgroup]],
    uint                 lane                [[thread_index_in_simdgroup]],
    uint                 sg                  [[simdgroup_index_in_threadgroup]],
    uint                 sgs                 [[simdgroups_per_threadgroup]]
) {
    const uint h = tg.x;
    const uint t = tg.y;
    const uint lid = lid3.x;
    const uint lsize = lsize3.x;
    if (t >= T || h >= num_heads) return;

    threadgroup float partial[kPrefillRmsMaxSimdGroups];
    device const half* xh = x + t * token_stride_elems + h * head_dim;
    device half* yh = out + t * token_stride_elems + h * head_dim;
    const float inv = prefill_rms_block_inv(xh, head_dim, eps, lid, lsize, lane, sg, sgs, partial);

    for (uint i = lid; i < head_dim; i += lsize) {
        yh[i] = half(float(xh[i]) * inv * float(weight[i]));
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void prefill_rmsnorm_no_scale_perhead_block(
    device const half*   x                   [[buffer(0)]],
    device half*         out                 [[buffer(1)]],
    constant uint&       T                   [[buffer(2)]],
    constant uint&       head_dim            [[buffer(3)]],
    constant uint&       num_heads           [[buffer(4)]],
    constant uint&       token_stride_elems  [[buffer(5)]],
    constant float&      eps                 [[buffer(6)]],
    uint3                tg                  [[threadgroup_position_in_grid]],
    uint3                lid3                [[thread_position_in_threadgroup]],
    uint3                lsize3              [[threads_per_threadgroup]],
    uint                 lane                [[thread_index_in_simdgroup]],
    uint                 sg                  [[simdgroup_index_in_threadgroup]],
    uint                 sgs                 [[simdgroups_per_threadgroup]]
) {
    const uint h = tg.x;
    const uint t = tg.y;
    const uint lid = lid3.x;
    const uint lsize = lsize3.x;
    if (t >= T || h >= num_heads) return;

    threadgroup float partial[kPrefillRmsMaxSimdGroups];
    device const half* xh = x + t * token_stride_elems + h * head_dim;
    device half* yh = out + t * token_stride_elems + h * head_dim;
    const float inv = prefill_rms_block_inv(xh, head_dim, eps, lid, lsize, lane, sg, sgs, partial);

    for (uint i = lid; i < head_dim; i += lsize) {
        yh[i] = half(float(xh[i]) * inv);
    }
}

// The same norms with one SIMD group per head row, eight rows a threadgroup,
// for head sizes that are multiples of 32 up to 512: each lane holds
// head_dim / 32 values, so a row needs one simd_sum and no barrier.
constant constexpr uint kPerHeadSimdRows = 8;

template <bool kScaled>
static inline void prefill_rmsnorm_perhead_simd(
    device const half* x, device const bfloat* weight, device half* out,
    uint T, uint head_dim, uint num_heads, uint token_stride_elems, float eps,
    uint tg, uint lane, uint sg
) {
    const uint row = tg * kPerHeadSimdRows + sg;
    if (row >= T * num_heads) return;
    const uint t = row / num_heads, h = row % num_heads;
    device const half* xh = x + t * token_stride_elems + h * head_dim;
    device half* yh = out + t * token_stride_elems + h * head_dim;
    const uint per_lane = head_dim / 32u;
    float v[16];
    float acc = 0.0f;
    for (uint i = 0; i < per_lane; ++i) {
        v[i] = float(xh[i * 32u + lane]);
        acc = fma(v[i], v[i], acc);
    }
    const float inv = rsqrt(simd_sum(acc) / float(head_dim) + eps);
    for (uint i = 0; i < per_lane; ++i) {
        const uint d = i * 32u + lane;
        yh[d] = half(kScaled ? v[i] * inv * float(weight[d]) : v[i] * inv);
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void prefill_rmsnorm_bf16w_perhead_simd(
    device const half*   x                   [[buffer(0)]],
    device const bfloat* weight              [[buffer(1)]],
    device half*         out                 [[buffer(2)]],
    constant uint&       T                   [[buffer(3)]],
    constant uint&       head_dim            [[buffer(4)]],
    constant uint&       num_heads           [[buffer(5)]],
    constant uint&       token_stride_elems  [[buffer(6)]],
    constant float&      eps                 [[buffer(7)]],
    uint                 tg                  [[threadgroup_position_in_grid]],
    uint                 lane                [[thread_index_in_simdgroup]],
    uint                 sg                  [[simdgroup_index_in_threadgroup]]
) {
    prefill_rmsnorm_perhead_simd<true>(x, weight, out, T, head_dim, num_heads, token_stride_elems, eps,
                                       tg, lane, sg);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void prefill_rmsnorm_no_scale_perhead_simd(
    device const half*   x                   [[buffer(0)]],
    device half*         out                 [[buffer(1)]],
    constant uint&       T                   [[buffer(2)]],
    constant uint&       head_dim            [[buffer(3)]],
    constant uint&       num_heads           [[buffer(4)]],
    constant uint&       token_stride_elems  [[buffer(5)]],
    constant float&      eps                 [[buffer(6)]],
    uint                 tg                  [[threadgroup_position_in_grid]],
    uint                 lane                [[thread_index_in_simdgroup]],
    uint                 sg                  [[simdgroup_index_in_threadgroup]]
) {
    prefill_rmsnorm_perhead_simd<false>(x, nullptr, out, T, head_dim, num_heads, token_stride_elems, eps,
                                        tg, lane, sg);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void prefill_post_attn_setup_block(
    device       half*   hidden                [[buffer(0)]],
    device const half*   attn                  [[buffer(1)]],
    device       half*   dense_x               [[buffer(2)]],
    device       half*   routed_x              [[buffer(3)]],
    device       half*   router_x              [[buffer(4)]],
    device const bfloat* w_post_attn           [[buffer(5)]],
    device const bfloat* w_pre_ffn             [[buffer(6)]],
    device const bfloat* w_pre_ffn2            [[buffer(7)]],
    constant uint&       T                     [[buffer(8)]],
    constant uint&       D                     [[buffer(9)]],
    constant uint&       hidden_stride_elems   [[buffer(10)]],
    constant uint&       attn_stride_elems     [[buffer(11)]],
    constant uint&       dense_stride_elems    [[buffer(12)]],
    constant uint&       routed_stride_elems   [[buffer(13)]],
    constant uint&       router_stride_elems   [[buffer(14)]],
    constant float&      rms_eps               [[buffer(15)]],
    uint                 row                   [[threadgroup_position_in_grid]],
    uint                 lid                   [[thread_position_in_threadgroup]],
    uint                 lsize                 [[threads_per_threadgroup]],
    uint                 lane                  [[thread_index_in_simdgroup]],
    uint                 sg                    [[simdgroup_index_in_threadgroup]],
    uint                 sgs                   [[simdgroups_per_threadgroup]]
) {
    if (row >= T || D > kPrefillPostMaxD) return;

    threadgroup half attn_norm_tg[kPrefillPostMaxD];
    threadgroup half hidden_tg[kPrefillPostMaxD];
    threadgroup float partial[kPrefillRmsMaxSimdGroups];

    device half* hidden_row = hidden + row * hidden_stride_elems;
    device const half* attn_row = attn + row * attn_stride_elems;
    device half* dense_row = dense_x + row * dense_stride_elems;
    device half* routed_row = routed_x + row * routed_stride_elems;
    device half* router_row = router_x + row * router_stride_elems;

    const float attn_inv = prefill_rms_block_inv(attn_row, D, rms_eps,
                                                 lid, lsize, lane, sg, sgs,
                                                 partial);
    for (uint i = lid; i < D; i += lsize) {
        attn_norm_tg[i] = half(float(attn_row[i]) * attn_inv * float(w_post_attn[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        half h = half(float(hidden_row[i]) + float(attn_norm_tg[i]));
        hidden_tg[i] = h;
        hidden_row[i] = h;
        float hf = float(h);
        acc = fma(hf, hf, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        partial[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        float sum = (lane < sgs) ? partial[lane] : 0.0f;
        sum = simd_sum(sum);
        if (lane == 0) {
            partial[0] = rsqrt(sum / float(D) + rms_eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float hidden_inv = partial[0];
    for (uint i = lid; i < D; i += lsize) {
        const float h = float(hidden_tg[i]) * hidden_inv;
        dense_row[i] = half(h * float(w_pre_ffn[i]));
        routed_row[i] = half(h * float(w_pre_ffn2[i]));
        router_row[i] = half(h);
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void prefill_layer_tail_block(
    device const half*   h2                    [[buffer(0)]],
    device const half*   h1                    [[buffer(1)]],
    device       half*   hidden                [[buffer(2)]],
    device const bfloat* w_postffn2            [[buffer(3)]],
    device const bfloat* w_postffn             [[buffer(4)]],
    constant uint&       T                     [[buffer(5)]],
    constant uint&       D                     [[buffer(6)]],
    constant uint&       h2_stride_elems       [[buffer(7)]],
    constant uint&       h1_stride_elems       [[buffer(8)]],
    constant uint&       hidden_stride_elems   [[buffer(9)]],
    constant float&      rms_eps               [[buffer(10)]],
    constant float&      layer_scalar          [[buffer(11)]],
    uint                 row                   [[threadgroup_position_in_grid]],
    uint                 lid                   [[thread_position_in_threadgroup]],
    uint                 lsize                 [[threads_per_threadgroup]],
    uint                 lane                  [[thread_index_in_simdgroup]],
    uint                 sg                    [[simdgroup_index_in_threadgroup]],
    uint                 sgs                   [[simdgroups_per_threadgroup]]
) {
    if (row >= T || D > kPrefillPostMaxD) return;

    threadgroup half tmp_tg[kPrefillPostMaxD];
    threadgroup half h12_tg[kPrefillPostMaxD];
    threadgroup float partial[kPrefillRmsMaxSimdGroups];

    device const half* h2_row = h2 + row * h2_stride_elems;
    device const half* h1_row = h1 + row * h1_stride_elems;
    device half* hidden_row = hidden + row * hidden_stride_elems;

    const float inv_h2 = prefill_rms_block_inv(h2_row, D, rms_eps,
                                               lid, lsize, lane, sg, sgs,
                                               partial);
    for (uint i = lid; i < D; i += lsize) {
        tmp_tg[i] = half(float(h2_row[i]) * inv_h2 * float(w_postffn2[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < D; i += lsize) {
        h12_tg[i] = h1_row[i] + tmp_tg[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        float v = float(h12_tg[i]);
        acc = fma(v, v, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        partial[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        float sum = (lane < sgs) ? partial[lane] : 0.0f;
        sum = simd_sum(sum);
        if (lane == 0) {
            partial[0] = rsqrt(sum / float(D) + rms_eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_h12 = partial[0];
    for (uint i = lid; i < D; i += lsize) {
        tmp_tg[i] = half(float(h12_tg[i]) * inv_h12 * float(w_postffn[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < D; i += lsize) {
        hidden_row[i] = hidden_row[i] + tmp_tg[i];
    }
    threadgroup_barrier(mem_flags::mem_device);

    const half h_scale = half(layer_scalar);
    for (uint i = lid; i < D; i += lsize) {
        hidden_row[i] = hidden_row[i] * h_scale;
    }
}

struct PrefillTokenExpertPairMSL {
    uint token;
    uint expert;
    uint rank;
    uint weight_bits_and_reserved;
};

struct PrefillStreamedRoutedBlobsMSL {
    device const uint8_t* blob[kPrefillMaxTileExperts];
};

struct PrefillGroupedRoutedMoEStreamedParamsMSL {
    uint pair_start;
    uint pair_count;
    uint D;
    uint F;
    uint top_k;
    uint hidden_stride_elements;
    uint live_expert_count;
    uint local_expert_0;
    uint local_expert_1;
    uint local_expert_2;
    uint local_expert_3;
    uint local_expert_4;
    uint local_expert_5;
    uint local_expert_6;
    uint local_expert_7;
    uint local_expert_8;
    uint local_expert_9;
    uint local_expert_10;
    uint local_expert_11;
    uint local_expert_12;
    uint local_expert_13;
    uint local_expert_14;
    uint local_expert_15;
    uint gate_W_off;
    uint gate_s_off;
    uint gate_b_off;
    uint up_W_off;
    uint up_s_off;
    uint up_b_off;
    uint down_W_off;
    uint down_s_off;
    uint down_b_off;
};

kernel void prefill_router_gemma4_block(
    device const uint8_t* W                [[buffer(0)]],
    device const bfloat*  scales           [[buffer(1)]],
    device const bfloat*  biases           [[buffer(2)]],
    device const half*    hidden           [[buffer(3)]],
    device const bfloat*  effective_scale  [[buffer(4)]],
    device const bfloat*  per_expert_scale [[buffer(5)]],
    device uint*          out_indices      [[buffer(6)]],
    device half*          out_weights      [[buffer(7)]],
    constant uint&        T                [[buffer(8)]],
    constant uint&        num_experts      [[buffer(9)]],
    constant uint&        D                [[buffer(10)]],
    constant uint&        top_k            [[buffer(11)]],
    constant uint&        hidden_stride    [[buffer(12)]],
    uint                  row              [[threadgroup_position_in_grid]],
    uint                  tid              [[thread_position_in_threadgroup]],
    uint                  tg_size          [[threads_per_threadgroup]]
) {
    if (row >= T) return;
    threadgroup float scores[kPrefillRouterMaxExperts];
    const uint NE = min(num_experts, kPrefillRouterMaxExperts);
    const uint KK = min(top_k, kPrefillRouterMaxTopK);
    device const half* row_hidden = hidden + row * hidden_stride;

    for (uint e = tid; e < NE; e += tg_size) {
        const uint n_groups = D / kPrefillGroupSize;
        device const uint8_t* W_row = W + e * D;
        device const bfloat* s_row = scales + e * n_groups;
        device const bfloat* b_row = biases + e * n_groups;

        float acc = 0.0f;
        for (uint g = 0; g < n_groups; ++g) {
            float s = float(s_row[g]);
            float b = float(b_row[g]);
            device const uint8_t* Wg = W_row + g * kPrefillGroupSize;
            device const half* xg = row_hidden + g * kPrefillGroupSize;
            device const bfloat* eg = effective_scale + g * kPrefillGroupSize;
            float dot_qx = 0.0f;
            float sum_x = 0.0f;
            for (uint k = 0; k < kPrefillGroupSize; ++k) {
                float q = float(uint(Wg[k]));
                float xv = float(xg[k]) * float(eg[k]);
                dot_qx = fma(q, xv, dot_qx);
                sum_x += xv;
            }
            acc = fma(s, dot_qx, acc);
            acc = fma(b, sum_x, acc);
        }
        scores[e] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        uint top_idx[kPrefillRouterMaxTopK];
        float top_score[kPrefillRouterMaxTopK];
        for (uint i = 0; i < kPrefillRouterMaxTopK; ++i) {
            top_idx[i] = 0u;
            top_score[i] = -INFINITY;
        }

        for (uint e = 0; e < NE; ++e) {
            float s = scores[e];
            if (KK > 0 && s <= top_score[KK - 1]) continue;
            uint pos = KK;
            for (uint i = 0; i < KK; ++i) {
                if (s > top_score[i] || (s == top_score[i] && e < top_idx[i])) {
                    pos = i;
                    break;
                }
            }
            if (pos >= KK) continue;
            for (uint i = KK - 1; i > pos; --i) {
                top_idx[i] = top_idx[i - 1];
                top_score[i] = top_score[i - 1];
            }
            top_idx[pos] = e;
            top_score[pos] = s;
        }

        float max_s = top_score[0];
        float sum_exp = 0.0f;
        float exps[kPrefillRouterMaxTopK];
        for (uint i = 0; i < KK; ++i) {
            float e = fast::exp(top_score[i] - max_s);
            exps[i] = e;
            sum_exp += e;
        }
        for (uint i = 0; i < KK; ++i) {
            const uint expert_idx = top_idx[i];
            const float w = exps[i] / sum_exp;
            const float gain = float(per_expert_scale[expert_idx]);
            out_indices[row * top_k + i] = expert_idx;
            out_weights[row * top_k + i] = half(w * gain);
        }
    }
}

kernel void prefill_moe_reduce_token_major(
    device const half* route_partials [[buffer(0)]],
    device const half* route_weights  [[buffer(1)]],
    device half*       h2             [[buffer(2)]],
    constant uint&     T              [[buffer(3)]],
    constant uint&     top_k          [[buffer(4)]],
    constant uint&     D              [[buffer(5)]],
    uint2              gid            [[thread_position_in_grid]]
) {
    const uint d = gid.x;
    const uint t = gid.y;
    if (t >= T || d >= D) return;

    float acc = 0.0f;
    for (uint r = 0; r < top_k; ++r) {
        const uint partial_index = (t * top_k + r) * D + d;
        acc = fma(float(route_weights[t * top_k + r]),
                  float(route_partials[partial_index]),
                  acc);
    }
    h2[t * D + d] = half(acc);
}

// ---- Routed experts as small matrix multiplies ------------------
// The host cuts each microbatch of expert-sorted pairs into tiles of up to
// 32 pairs that share one expert. A threadgroup of four SIMD groups computes
// 64 output rows for one tile on the tensor units: per 32-wide step along
// the input it stages the pairs' inputs and the dequantized weights in FP16
// and runs one 32x64x32 matmul2d per projection, accumulating in FP32.
struct PrefillMMATileMSL {
    uint pair;      // first pair, within the microbatch
    uint count;     // pairs, 1...32
    uint slot;      // the expert's slot in the routed tile
    uint reserved;
};

// Thirty-two weights of `row` from 32-weight chunk `c`, dequantized.
static inline void prefill_mma_dequant32(device const uint8_t* W, device const bfloat* S, device const bfloat* B,
                                         uint row, uint N, uint c, threadgroup half* out) {
    const uint groups = N / kPrefillGroupSize;
    const float sc = float(S[row * groups + c / 2u]), bi = float(B[row * groups + c / 2u]);
    device const uint* W_row = (device const uint*)(W + row * (N / 2u));
    for (uint k = 0; k < 4u; ++k) {
        const uint w = W_row[4u * c + k];
        for (uint j = 0; j < 8u; ++j) out[8u * k + j] = half(fma(float((w >> (4u * j)) & 0xFu), sc, bi));
    }
}

#if defined(__HAVE_TENSOR__)
// The activations between the two kernels are FP16 with a power-of-two
// scale per pair and 64-row block, so values past FP16's range (~78k seen)
// keep FP16's relative precision. The down kernel brings each block to its
// pair's largest scale as it stages it (exact unless a value falls below
// FP16's normal range relative to the pair's largest) and applies that
// scale to the pair's sums.
constant constexpr int kRoutedTensorPairs = 32;
constant constexpr int kRoutedTensorRows = 64;
constant constexpr int kRoutedTensorK = 64;
constant constexpr uint kRoutedTensorMaxBlocks = 16;   // F up to 1024

// Block scales follow this microbatch's FP16 activations in act_scratch.
static inline uint prefill_routed_tensor_scale_offset(constant PrefillGroupedRoutedMoEStreamedParamsMSL& p) {
    return p.pair_count * p.F / 2u;   // in floats
}

kernel void prefill_routed_gate_up_tensorops(
    device const half*                                   hidden               [[buffer(0)]],
    device const PrefillTokenExpertPairMSL*              sorted_pairs         [[buffer(1)]],
    device half*                                         act_scratch          [[buffer(7)]],
    device const PrefillStreamedRoutedBlobsMSL&          routed               [[buffer(9)]],
    constant PrefillGroupedRoutedMoEStreamedParamsMSL&   p                    [[buffer(10)]],
    constant PrefillMMATileMSL*                          tiles                [[buffer(11)]],
    uint2                                                tg                   [[threadgroup_position_in_grid]],
    uint                                                 tid                  [[thread_index_in_threadgroup]]
) {
    // 32-wide steps keep threadgroup memory near 10 KB, so several
    // threadgroups share a core.
    constexpr int kStep = 32;
    constexpr auto desc = matmul2d_descriptor(kRoutedTensorPairs, kRoutedTensorRows, kStep,
                                              false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;
    using tg_half_tensor = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;

    threadgroup half xs[kRoutedTensorPairs * kStep];
    // Gate and up weights as FP16 [row][k] during the loop, then the
    // activations as FP32 [pair][row].
    threadgroup float space[kRoutedTensorPairs * kRoutedTensorRows];
    threadgroup half* wg = reinterpret_cast<threadgroup half*>(space);
    threadgroup half* wu = wg + kRoutedTensorRows * kStep;
    threadgroup uint tokens[kRoutedTensorPairs];
    const PrefillMMATileMSL tile = tiles[tg.y];
    const uint row0 = tg.x * uint(kRoutedTensorRows);
    if (tid < uint(kRoutedTensorPairs)) {
        tokens[tid] = tid < tile.count ? sorted_pairs[p.pair_start + tile.pair + tid].token : 0u;
    }
    device const uint8_t* expert = routed.blob[tile.slot];
    const bool gate_thread = tid < uint(kRoutedTensorRows);
    device const uint8_t* W = expert + (gate_thread ? p.gate_W_off : p.up_W_off);
    device const bfloat* S = reinterpret_cast<device const bfloat*>(expert + (gate_thread ? p.gate_s_off : p.up_s_off));
    device const bfloat* B = reinterpret_cast<device const bfloat*>(expert + (gate_thread ? p.gate_b_off : p.up_b_off));
    const uint w_row = tid % uint(kRoutedTensorRows);
    threadgroup half* w_dst = (gate_thread ? wg : wu) + w_row * kStep;

    tg_half_tensor x_t(xs, dextents<int32_t, 2>(kStep, kRoutedTensorPairs), array<int32_t, 2>({1, kStep}));
    tg_half_tensor g_t(wg, dextents<int32_t, 2>(kStep, kRoutedTensorRows), array<int32_t, 2>({1, kStep}));
    tg_half_tensor u_t(wu, dextents<int32_t, 2>(kStep, kRoutedTensorRows), array<int32_t, 2>({1, kStep}));
    auto cg = op.get_destination_cooperative_tensor<decltype(x_t), decltype(g_t), float>();
    auto cu = op.get_destination_cooperative_tensor<decltype(x_t), decltype(u_t), float>();
    for (int e = 0; e < cg.get_capacity(); ++e) { cg[e] = 0.0f; cu[e] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Staging: four threads a pair, eight inputs each.
    const uint stage_pair = tid / 4u, stage_k = (tid % 4u) * 8u;
    threadgroup half4* x_dst = reinterpret_cast<threadgroup half4*>(xs + stage_pair * kStep + stage_k);
    const bool stage_live = stage_pair < tile.count;
    device const half* x_row = hidden + tokens[stage_pair] * p.hidden_stride_elements + stage_k;
    for (uint c = 0; c < p.D / uint(kStep); ++c) {
        if (stage_live) {
            device const half4* x_src = reinterpret_cast<device const half4*>(x_row + c * uint(kStep));
            x_dst[0] = x_src[0];
            x_dst[1] = x_src[1];
        } else {
            x_dst[0] = half4(0.0h);
            x_dst[1] = half4(0.0h);
        }
        prefill_mma_dequant32(W, S, B, row0 + w_row, p.D, c, w_dst);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(x_t, g_t, cg);
        op.run(x_t, u_t, cu);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // The two accumulators share one layout, so each element's activation is
    // formed in registers.
    for (int e = 0; e < cg.get_capacity(); ++e) {
        if (!cg.is_valid_element(e)) continue;
        const auto pos = cg.get_multidimensional_index(e);
        space[uint(pos[1]) * kRoutedTensorRows + uint(pos[0])] = prefill_gelu_pytorch_tanh(cg[e]) * cu[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Four threads a pair, sixteen rows each; the four are adjacent lanes.
    const uint pr = tid / 4u, r0 = (tid % 4u) * 16u;
    float peak = 0.0f;
    for (uint i = 0; i < 16u; ++i) peak = max(peak, abs(space[pr * kRoutedTensorRows + r0 + i]));
    peak = max(peak, simd_shuffle_xor(peak, ushort(1)));
    peak = max(peak, simd_shuffle_xor(peak, ushort(2)));
    int exponent = 0;
    frexp(peak, exponent);   // peak < 2^exponent
    const float to_half = ldexp(1.0f, 14 - exponent);
    if (pr < tile.count) {
        const uint pair = tile.pair + pr;
        device half* dst = act_scratch + pair * p.F + row0 + r0;
        for (uint i = 0; i < 16u; ++i) dst[i] = half(space[pr * kRoutedTensorRows + r0 + i] * to_half);
        if (tid % 4u == 0u) {
            device float* scales = reinterpret_cast<device float*>(act_scratch) + prefill_routed_tensor_scale_offset(p);
            scales[pair * (p.F / uint(kRoutedTensorK)) + tg.x] = ldexp(1.0f, exponent - 14);
        }
    }
}

kernel void prefill_routed_down_tensorops(
    device const PrefillTokenExpertPairMSL*              sorted_pairs         [[buffer(1)]],
    device half*                                         route_partials       [[buffer(5)]],
    device half*                                         act_scratch          [[buffer(7)]],
    device const PrefillStreamedRoutedBlobsMSL&          routed               [[buffer(9)]],
    constant PrefillGroupedRoutedMoEStreamedParamsMSL&   p                    [[buffer(10)]],
    constant PrefillMMATileMSL*                          tiles                [[buffer(11)]],
    uint2                                                tg                   [[threadgroup_position_in_grid]],
    uint                                                 tid                  [[thread_index_in_threadgroup]]
) {
    constexpr int kStep = 32;
    constexpr auto desc = matmul2d_descriptor(kRoutedTensorPairs, kRoutedTensorRows, kStep,
                                              false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;
    using tg_half_tensor = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;

    threadgroup half xs[kRoutedTensorPairs * kStep];
    threadgroup half wd[kRoutedTensorRows * kStep];
    // Each block's scale relative to its pair's largest (a power of two at
    // most 1), and that largest.
    threadgroup float relative[kRoutedTensorPairs * kRoutedTensorMaxBlocks];
    threadgroup float pair_scale[kRoutedTensorPairs];
    threadgroup uint dest[kRoutedTensorPairs];
    const PrefillMMATileMSL tile = tiles[tg.y];
    const uint row0 = tg.x * uint(kRoutedTensorRows);
    const uint blocks = p.F / uint(kRoutedTensorK);
    if (tid < uint(kRoutedTensorPairs)) {
        const uint pr = min(tid, tile.count - 1u);
        const PrefillTokenExpertPairMSL pair = sorted_pairs[p.pair_start + tile.pair + pr];
        dest[tid] = (pair.token * p.top_k + pair.rank) * p.D;
        device const float* scales = reinterpret_cast<device const float*>(act_scratch)
            + prefill_routed_tensor_scale_offset(p) + (tile.pair + pr) * blocks;
        float largest = 0.0f;
        for (uint b = 0; b < blocks; ++b) largest = max(largest, scales[b]);
        for (uint b = 0; b < blocks; ++b) relative[tid * blocks + b] = scales[b] / largest;
        pair_scale[tid] = largest;
    }
    device const uint8_t* expert = routed.blob[tile.slot];
    device const bfloat* dS = reinterpret_cast<device const bfloat*>(expert + p.down_s_off);
    device const bfloat* dB = reinterpret_cast<device const bfloat*>(expert + p.down_b_off);

    tg_half_tensor a_t(xs, dextents<int32_t, 2>(kStep, kRoutedTensorPairs), array<int32_t, 2>({1, kStep}));
    tg_half_tensor w_t(wd, dextents<int32_t, 2>(kStep, kRoutedTensorRows), array<int32_t, 2>({1, kStep}));
    auto acc = op.get_destination_cooperative_tensor<decltype(a_t), decltype(w_t), float>();
    for (int e = 0; e < acc.get_capacity(); ++e) acc[e] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Staging: four threads a pair, eight activations each, brought to the
    // pair's largest scale; the first 64 threads dequantize a row each.
    const uint stage_pair = tid / 4u, stage_k = (tid % 4u) * 8u;
    const bool stage_live = stage_pair < tile.count;
    device const half4* a_row = reinterpret_cast<device const half4*>(
        act_scratch + (tile.pair + stage_pair) * p.F + stage_k);
    threadgroup half4* x_dst = reinterpret_cast<threadgroup half4*>(xs + stage_pair * kStep + stage_k);
    for (uint c = 0; c < p.F / uint(kStep); ++c) {
        if (stage_live) {
            const half r = half(relative[stage_pair * blocks + c / 2u]);
            for (uint i = 0; i < 2u; ++i) x_dst[i] = a_row[c * uint(kStep) / 4u + i] * r;
        } else {
            for (uint i = 0; i < 2u; ++i) x_dst[i] = half4(0.0h);
        }
        if (tid < uint(kRoutedTensorRows)) {
            prefill_mma_dequant32(expert + p.down_W_off, dS, dB, row0 + tid, p.F, c, wd + tid * kStep);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(a_t, w_t, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int e = 0; e < acc.get_capacity(); ++e) {
        if (!acc.is_valid_element(e)) continue;
        const auto pos = acc.get_multidimensional_index(e);
        const uint pr = uint(pos[1]);
        if (pr < tile.count) route_partials[dest[pr] + row0 + uint(pos[0])] = half(acc[e] * pair_scale[pr]);
    }
}
#endif

static inline void prefill_rope_apply_neox_pair(
    device half* head_ptr,
    uint i,
    uint half_dim,
    uint freq_divisor,
    float position,
    float theta_base
) {
    const float exponent = -float(2u * i) / float(freq_divisor);
    const float freq = pow(theta_base, exponent);
    const float angle = position * freq;
    const float c = cos(angle);
    const float s = sin(angle);

    const uint i0 = i;
    const uint i1 = half_dim + i;
    const float x0 = float(head_ptr[i0]);
    const float x1 = float(head_ptr[i1]);
    head_ptr[i0] = half(x0 * c - x1 * s);
    head_ptr[i1] = half(x0 * s + x1 * c);
}

kernel void prefill_rope_default_neox_block(
    device half*   data                [[buffer(0)]],
    constant uint& start_position      [[buffer(1)]],
    constant uint& head_dim            [[buffer(2)]],
    constant uint& num_heads           [[buffer(3)]],
    constant uint& token_stride_elems  [[buffer(4)]],
    constant float& theta_base         [[buffer(5)]],
    uint3          gid                 [[thread_position_in_grid]]
) {
    const uint i = gid.x;
    const uint h = gid.y;
    const uint t = gid.z;
    const uint half_dim = head_dim / 2u;
    if (i >= half_dim) return;
    if (h >= num_heads) return;

    device half* head_ptr = data + t * token_stride_elems + h * head_dim;
    prefill_rope_apply_neox_pair(head_ptr, i, half_dim, head_dim,
                                 float(start_position + t), theta_base);
}

kernel void prefill_rope_proportional_neox_block(
    device half*   data                [[buffer(0)]],
    constant uint& start_position      [[buffer(1)]],
    constant uint& head_dim            [[buffer(2)]],
    constant uint& num_heads           [[buffer(3)]],
    constant uint& token_stride_elems  [[buffer(4)]],
    constant float& theta_base         [[buffer(5)]],
    constant uint& rotated_pairs       [[buffer(6)]],
    uint3          gid                 [[thread_position_in_grid]]
) {
    const uint i = gid.x;
    const uint h = gid.y;
    const uint t = gid.z;
    if (i >= rotated_pairs) return;
    if (h >= num_heads) return;

    const uint half_dim = head_dim / 2u;
    device half* head_ptr = data + t * token_stride_elems + h * head_dim;
    prefill_rope_apply_neox_pair(head_ptr, i, half_dim, head_dim,
                                 float(start_position + t), theta_base);
}

struct PrefillAttentionParams {
    uint startPosition;
    uint queryCount;
    uint headDim;
    uint numQHeads;
    uint numKVHeads;
    uint kvValidCount;
    uint slidingWindow;
    uint kvTokenStrideElements;
    uint qTokenStrideElements;
    uint oTokenStrideElements;
    float scale;
    uint bidirectionalBlockStart;
    uint bidirectionalBlockEnd;
};

static inline uint prefill_attention_last_exclusive(
    constant PrefillAttentionParams& p,
    uint abs_q
) {
    const bool in_bidirectional_block =
        abs_q >= p.bidirectionalBlockStart &&
        abs_q < p.bidirectionalBlockEnd;
    return min(
        p.kvValidCount,
        in_bidirectional_block ? p.bidirectionalBlockEnd : abs_q + 1u);
}

static inline uint prefill_kv_slot(uint logical) {
    return (is_function_constant_defined(FC_PREFILL_KV_RING_CAP) &&
            FC_PREFILL_KV_RING_CAP != 0u)
        ? (logical % FC_PREFILL_KV_RING_CAP)
        : logical;
}

static inline float prefill_attention_tg_sum(
    float value,
    uint lane,
    uint simd_group,
    uint simdgroups,
    threadgroup float* partial
) {
    float s = simd_sum(value);
    if (lane == 0u) {
        partial[simd_group] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0u) {
        float v = lane < simdgroups ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) {
            partial[0] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return partial[0];
}

constant uint FC_PREFILL_SIMD_D [[function_constant(77)]];
// One SIMD group owns a query/head. Keep Q and the output accumulator in
// registers; a score reduction needs no cross-SIMD threadgroup barriers.
[[kernel, max_total_threads_per_threadgroup(32)]]
kernel void attention_prefill_causal_simd(
    device const half* Q [[buffer(0)]], device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]], device half* O [[buffer(3)]],
    constant PrefillAttentionParams& p [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint t = tg.x, qh = tg.y, columns = FC_PREFILL_SIMD_D / 32;
    const uint kvh = qh / (p.numQHeads / p.numKVHeads);
    const uint absolute = p.startPosition + t;
    const uint first = p.slidingWindow && absolute + 1 > p.slidingWindow
        ? absolute + 1 - p.slidingWindow : 0;
    const uint end = prefill_attention_last_exclusive(p, absolute);
    float q[16], acc[16];
    #pragma unroll
    for (uint c = 0; c < columns; ++c) {
        q[c] = float(Q[t * p.qTokenStrideElements + qh * p.headDim + c * 32 + lane]);
        acc[c] = 0;
    }
    float maximum = -INFINITY, denominator = 0;
    for (uint key = first; key < end; ++key) {
        uint base = prefill_kv_slot(key) * p.kvTokenStrideElements + kvh * p.headDim;
        // Match the tiled reference's two-stage reduction exactly: each
        // contiguous 32-channel segment first, then the segment totals.
        // Reassociating this as per-lane dot products changes prefill KV.
        float laneTotal = 0;
        #pragma unroll
        for (uint c = 0; c < columns; ++c) {
            float subtotal = simd_sum(q[c] * float(K[base + c * 32 + lane]));
            if (lane == c) laneTotal = subtotal;
        }
        float score = simd_sum(laneTotal) * p.scale;
        float nextMaximum = max(maximum, score);
        float oldWeight = denominator > 0 ? fast::exp(maximum - nextMaximum) : 0;
        float newWeight = fast::exp(score - nextMaximum);
        #pragma unroll
        for (uint c = 0; c < columns; ++c)
            acc[c] = fma(newWeight, float(V[base + c * 32 + lane]), acc[c] * oldWeight);
        denominator = denominator * oldWeight + newWeight;
        maximum = nextMaximum;
    }
    #pragma unroll
    for (uint c = 0; c < columns; ++c)
        O[t * p.oTokenStrideElements + qh * p.headDim + c * 32 + lane] = half(denominator > 0 ? acc[c] / denominator : 0);
}

[[kernel, max_total_threads_per_threadgroup(512)]]
kernel void attention_prefill_causal_tiled(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    constant PrefillAttentionParams& p [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]]
) {
    const uint t = tg.x;
    const uint qh = tg.y;
    if (t >= p.queryCount || qh >= p.numQHeads) return;

    threadgroup float partial[2u * kPrefillAttentionMaxSimdGroups];

    const uint d = tid.x;
    const bool owns = d < p.headDim;
    const uint q_per_kv = p.numQHeads / p.numKVHeads;
    const uint kvh = qh / q_per_kv;
    const uint abs_q = p.startPosition + t;
    uint first = 0u;
    if (p.slidingWindow != 0u && abs_q + 1u > p.slidingWindow) {
        first = abs_q + 1u - p.slidingWindow;
    }
    const uint last_exclusive = prefill_attention_last_exclusive(p, abs_q);

    device const half* q_row = Q + t * p.qTokenStrideElements + qh * p.headDim;
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float acc = 0.0f;

    for (uint key = first; key < last_exclusive; ++key) {
        const uint phys_key = prefill_kv_slot(key);
        device const half* k_row = K + phys_key * p.kvTokenStrideElements + kvh * p.headDim;
        const float qv = owns ? float(q_row[d]) : 0.0f;
        const float kv = owns ? float(k_row[d]) : 0.0f;
        const uint bank = key & 1u;
        const float score = prefill_attention_tg_sum(
            qv * kv,
            lane,
            simd_group,
            simdgroups,
            partial + bank * kPrefillAttentionMaxSimdGroups) * p.scale;

        const float new_max = max(row_max, score);
        const float old_scale = row_sum > 0.0f ? fast::exp(row_max - new_max) : 0.0f;
        const float new_scale = fast::exp(score - new_max);
        if (owns) {
            device const half* v_row = V + phys_key * p.kvTokenStrideElements + kvh * p.headDim;
            acc = fma(new_scale, float(v_row[d]), acc * old_scale);
        }
        row_sum = row_sum * old_scale + new_scale;
        row_max = new_max;
    }

    if (owns) {
        device half* out_row = O + t * p.oTokenStrideElements + qh * p.headDim;
        out_row[d] = row_sum > 0.0f ? half(acc / row_sum) : half(0.0f);
    }
}

#if defined(__HAVE_TENSOR__)

// Grouped-query attention prefill on the tensor units: a threadgroup takes
// 16 rows, `16 / Heads` queries under the `Heads` query heads of one KV head,
// so each block of K and V serves all of them. Scores (Q K^T) and the value
// sum (P V) are matmul2d ops; the online softmax between them gives each row
// `threads / 16` lanes of one SIMD group and writes the weights in FP16 over
// the first half of the row's own scores, as the tensor units take FP32
// inputs much more slowly. Causal; sliding layers start at their window and
// cut key blocks where their cache ring (function constant 76) wraps. No
// bidirectional block. Ported from the Flint fork's head-size-128 kernel.
constant constexpr int kGQATensorRows = 16;

template <int HeadDim, int Heads, int Keys, int SimdGroups>
kernel void attention_prefill_gqa_tensorops(
    device const half* Q [[buffer(0)]],
    device half* K [[buffer(1)]],
    device half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    constant PrefillAttentionParams& p [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {
    constexpr int Queries = kGQATensorRows / Heads;
    constexpr uint threads = uint(32 * SimdGroups);
    threadgroup half query_tile[kGQATensorRows * HeadDim];
    threadgroup float score_tile[kGQATensorRows * Keys];
    threadgroup float row_max[kGQATensorRows];
    threadgroup float row_sum[kGQATensorRows];
    threadgroup float row_old_scale[kGQATensorRows];

    constexpr auto qk_desc = matmul2d_descriptor(kGQATensorRows, Keys, HeadDim, false, true, false);
    constexpr auto pv_desc = matmul2d_descriptor(kGQATensorRows, HeadDim, Keys, false, false, false);
    matmul2d<qk_desc, execution_simdgroups<SimdGroups>> qk_op;
    matmul2d<pv_desc, execution_simdgroups<SimdGroups>> pv_op;
    using device_half_tensor = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using threadgroup_half_tensor = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;

    const uint q0 = tg.x * uint(Queries);
    const uint kvh = tg.y;
    const uint qh0 = kvh * uint(Heads);
    const uint live_queries = min(uint(Queries), p.queryCount - q0);

    // Row r is query r / Heads under query head qh0 + r % Heads.
    for (uint i = lid; i < uint(kGQATensorRows * HeadDim); i += threads) {
        const uint row = i / uint(HeadDim), d = i % uint(HeadDim);
        const uint ql = row / uint(Heads), h = row % uint(Heads);
        query_tile[i] = ql < live_queries
            ? Q[(q0 + ql) * p.qTokenStrideElements + (qh0 + h) * uint(HeadDim) + d]
            : half(0.0h);
    }
    if (lid < uint(kGQATensorRows)) {
        row_max[lid] = -INFINITY;
        row_sum[lid] = 0.0f;
        row_old_scale[lid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup_half_tensor query_tensor(
        query_tile, dextents<int32_t, 2>(HeadDim, kGQATensorRows), array<int32_t, 2>({1, HeadDim}));
    threadgroup half* weight_tile = reinterpret_cast<threadgroup half*>(score_tile);
    threadgroup_half_tensor weight_tensor(
        weight_tile, dextents<int32_t, 2>(Keys, kGQATensorRows), array<int32_t, 2>({1, 2 * Keys}));
    // One key block's rows: `rows` cache slots from `slot`. Rows past the
    // view read as zero.
    auto kv_block = [&](device half* base, uint slot, uint rows) {
        return device_half_tensor(
            base + slot * p.kvTokenStrideElements + kvh * uint(HeadDim),
            dextents<int32_t, 2>(HeadDim, int32_t(rows)),
            array<int32_t, 2>({1, int32_t(p.kvTokenStrideElements)})).slice(0, 0);
    };

    auto query_slice = query_tensor.slice(0, 0);
    auto first_value_slice = kv_block(V, 0u, 1u);
    auto output_accumulator = pv_op.template get_destination_cooperative_tensor<
        decltype(weight_tensor), decltype(first_value_slice), float>();
    #pragma clang loop unroll(full)
    for (int e = 0; e < output_accumulator.get_capacity(); ++e) {
        if (output_accumulator.is_valid_element(e)) output_accumulator[e] = 0.0f;
    }

    // This thread's softmax row and share of each key block.
    constexpr uint row_threads = threads / uint(kGQATensorRows);
    const uint my_row = lid / row_threads, my_part = lid % row_threads;
    const uint my_query = my_row / uint(Heads);
    const uint my_end = my_query < live_queries
        ? prefill_attention_last_exclusive(p, p.startPosition + q0 + my_query) : 0u;
    const uint my_abs = p.startPosition + q0 + my_query;
    const uint my_first = p.slidingWindow && my_abs + 1u > p.slidingWindow ? my_abs + 1u - p.slidingWindow : 0u;
    uint key_end = 0u;
    for (uint ql = 0; ql < live_queries; ++ql) {
        key_end = max(key_end, prefill_attention_last_exclusive(p, p.startPosition + q0 + ql));
    }
    const uint first_abs = p.startPosition + q0;
    const uint key_begin = p.slidingWindow && first_abs + 1u > p.slidingWindow
        ? first_abs + 1u - p.slidingWindow : 0u;
    const uint ring = is_function_constant_defined(FC_PREFILL_KV_RING_CAP) ? FC_PREFILL_KV_RING_CAP : 0u;

    // Blocks of up to `Keys` keys, cut where a sliding layer's ring wraps.
    uint block_keys = 0u;
    for (uint key_start = key_begin; key_start < key_end; key_start += block_keys) {
        const uint slot = prefill_kv_slot(key_start);
        block_keys = min(uint(Keys), key_end - key_start);
        if (ring != 0u) block_keys = min(block_keys, ring - slot);
        auto key_slice = kv_block(K, slot, block_keys);
        auto score_product = qk_op.template get_destination_cooperative_tensor<
            decltype(query_slice), decltype(key_slice), float>();
        #pragma clang loop unroll(full)
        for (int e = 0; e < score_product.get_capacity(); ++e) {
            if (score_product.is_valid_element(e)) score_product[e] = 0.0f;
        }
        qk_op.run(query_slice, key_slice, score_product);
        #pragma clang loop unroll(full)
        for (int e = 0; e < score_product.get_capacity(); ++e) {
            if (!score_product.is_valid_element(e)) continue;
            const auto position = score_product.get_multidimensional_index(e);
            score_tile[uint(position[1]) * uint(Keys) + uint(position[0])] = score_product[e] * p.scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // A row's threads are lanes of one SIMD group: each holds its scores
        // in registers, so the row's weights can overwrite them.
        constexpr uint part_keys = uint(Keys) / row_threads;
        float row_scores[part_keys];
        float m = -INFINITY;
        for (uint i = 0; i < part_keys; ++i) {
            const uint j = my_part * part_keys + i;
            row_scores[i] = score_tile[my_row * uint(Keys) + j];
            if (j < block_keys && key_start + j >= my_first && key_start + j < my_end) m = max(m, row_scores[i]);
        }
        for (uint lane_bit = 1u; lane_bit < row_threads; lane_bit <<= 1) m = max(m, simd_shuffle_xor(m, lane_bit));
        const float old_max = row_max[my_row];
        m = max(m, old_max);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float sum = 0.0f;
        for (uint i = 0; i < part_keys; ++i) {
            const uint j = my_part * part_keys + i;
            const float w = j < block_keys && key_start + j >= my_first && key_start + j < my_end && m > -INFINITY
                ? fast::exp(row_scores[i] - m) : 0.0f;
            weight_tile[my_row * uint(2 * Keys) + j] = half(w);
            sum += w;
        }
        for (uint lane_bit = 1u; lane_bit < row_threads; lane_bit <<= 1) sum += simd_shuffle_xor(sum, lane_bit);
        if (my_part == 0u) {
            const float scale_old = old_max > -INFINITY ? fast::exp(old_max - m) : 0.0f;
            row_sum[my_row] = row_sum[my_row] * scale_old + sum;
            row_max[my_row] = m;
            row_old_scale[my_row] = scale_old;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto value_slice = kv_block(V, slot, block_keys);
        auto output_product = pv_op.template get_destination_cooperative_tensor<
            decltype(weight_tensor), decltype(value_slice), float>();
        #pragma clang loop unroll(full)
        for (int e = 0; e < output_product.get_capacity(); ++e) {
            if (output_product.is_valid_element(e)) output_product[e] = 0.0f;
        }
        pv_op.run(weight_tensor, value_slice, output_product);
        #pragma clang loop unroll(full)
        for (int e = 0; e < output_accumulator.get_capacity(); ++e) {
            if (!output_accumulator.is_valid_element(e) || !output_product.is_valid_element(e)) continue;
            const auto position = output_accumulator.get_multidimensional_index(e);
            output_accumulator[e] = fma(1.0f, output_product[e],
                                        output_accumulator[e] * row_old_scale[uint(position[1])]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (int e = 0; e < output_accumulator.get_capacity(); ++e) {
        if (!output_accumulator.is_valid_element(e)) continue;
        const auto position = output_accumulator.get_multidimensional_index(e);
        const uint d = uint(position[0]), row = uint(position[1]);
        const uint ql = row / uint(Heads), h = row % uint(Heads);
        if (ql >= live_queries) continue;
        const float denominator = row_sum[row];
        O[(q0 + ql) * p.oTokenStrideElements + (qh0 + h) * uint(HeadDim) + d] =
            denominator > 0.0f ? half(output_accumulator[e] / denominator) : half(0.0f);
    }
}

#define ATTENTION_PREFILL_GQA_TENSOROPS(NAME, D, H, KEYS, SG)                              \
    template [[host_name(NAME)]] kernel void attention_prefill_gqa_tensorops<D, H, KEYS, SG>( \
        device const half*, device half*, device half*, device half*,                     \
        constant PrefillAttentionParams&, uint3, uint);
// Gemma's sliding layers (head size 256, two query heads a KV head) and full
// layers (512, eight).
ATTENTION_PREFILL_GQA_TENSOROPS("attention_prefill_gqa_tensorops_d256_h2", 256, 2, 128, 4)
ATTENTION_PREFILL_GQA_TENSOROPS("attention_prefill_gqa_tensorops_d512_h8", 512, 8, 64, 8)
#undef ATTENTION_PREFILL_GQA_TENSOROPS
#endif

// ============================================================================
// Verify routed experts: one SIMD per weight row, applied to every token the
// tile routes to that expert, so a verify block reads each expert's weights
// once (the batched kernels above give each (row, pair) its own thread and
// re-read the row per pair). `segments[slot]` is the slot's (first pair,
// pair count) in `sorted_pairs`; at most FC_VERIFY_ROWS pairs per expert.
// Row math is the decode GEMV's lane layout: 4 bytes per lane per 4-group
// block, scale and bias factored per group, u16 weight loads.
// ============================================================================
constant uint FC_VERIFY_ROWS [[function_constant(90)]];
constant constexpr uint kVerifyMaxRows = 16;

static inline void verify_unpack8(uint w4, thread float* q) {
    q[0] = float(w4 & 0x0Fu);         q[1] = float((w4 >> 4) & 0x0Fu);
    q[2] = float((w4 >> 8) & 0x0Fu);  q[3] = float((w4 >> 12) & 0x0Fu);
    q[4] = float((w4 >> 16) & 0x0Fu); q[5] = float((w4 >> 20) & 0x0Fu);
    q[6] = float((w4 >> 24) & 0x0Fu); q[7] = float(w4 >> 28);
}

static inline float verify_dot8(thread const float* q, half4 xa, half4 xb, thread float& sum) {
    const float e0 = float(xa.x), e1 = float(xa.y), e2 = float(xa.z), e3 = float(xa.w);
    const float e4 = float(xb.x), e5 = float(xb.y), e6 = float(xb.z), e7 = float(xb.w);
    sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
    float dot = 0.0f;
    dot = fma(q[0], e0, dot); dot = fma(q[1], e1, dot);
    dot = fma(q[2], e2, dot); dot = fma(q[3], e3, dot);
    dot = fma(q[4], e4, dot); dot = fma(q[5], e5, dot);
    dot = fma(q[6], e6, dot); dot = fma(q[7], e7, dot);
    return dot;
}

/// One gate/up weight row `f` of `expert` applied to `n` rows: row r reads
/// `hidden` row `tokens[r]` and writes its activation to `acts` row
/// `act_rows[r]`.
static inline void verify_gate_up_act_rows(
    device const half* hidden, device half* acts, device const uint8_t* expert,
    constant PrefillGroupedRoutedMoEStreamedParamsMSL& p, uint f, uint lane, uint n,
    thread const uint* tokens, thread const uint* act_rows
) {
    const uint R = FC_VERIFY_ROWS;
    const uint D = p.D;
    const uint n_groups = D / kPrefillGroupSize;
    device const uint8_t* gW = expert + p.gate_W_off + f * (D / 2u);
    device const uint8_t* uW = expert + p.up_W_off + f * (D / 2u);
    device const bfloat* gS = reinterpret_cast<device const bfloat*>(expert + p.gate_s_off) + f * n_groups;
    device const bfloat* gB = reinterpret_cast<device const bfloat*>(expert + p.gate_b_off) + f * n_groups;
    device const bfloat* uS = reinterpret_cast<device const bfloat*>(expert + p.up_s_off) + f * n_groups;
    device const bfloat* uB = reinterpret_cast<device const bfloat*>(expert + p.up_b_off) + f * n_groups;

    device const half* xrow[kVerifyMaxRows];
    float ga[kVerifyMaxRows], ua[kVerifyMaxRows];
    for (uint r = 0; r < R; ++r) {
        xrow[r] = hidden + tokens[min(r, n - 1u)] * p.hidden_stride_elements;
        ga[r] = 0.0f; ua[r] = 0.0f;
    }

    const uint full_blocks = n_groups / 4u;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        device const ushort* gp = (device const ushort*)(gW + byte_base);
        device const ushort* up = (device const ushort*)(uW + byte_base);
        float qg[8], qu[8];
        verify_unpack8(uint(gp[0]) | (uint(gp[1]) << 16), qg);
        verify_unpack8(uint(up[0]) | (uint(up[1]) << 16), qu);
        const uint g = blk * 4u + (lane >> 3);
        const float gs = float(gS[g]), gb = float(gB[g]);
        const float us = float(uS[g]), ub = float(uB[g]);
        const uint elem = byte_base * 2u;
        for (uint r = 0; r < R; ++r) {
            if (r < n) {
                const half4 xa = *((device const half4*)(xrow[r] + elem));
                const half4 xb = *((device const half4*)(xrow[r] + elem + 4u));
                float sum;
                const float gd = verify_dot8(qg, xa, xb, sum);
                const float ud = verify_dot8(qu, xa, xb, sum);
                ga[r] = fma(gs, gd, ga[r]); ga[r] = fma(gb, sum, ga[r]);
                ua[r] = fma(us, ud, ua[r]); ua[r] = fma(ub, sum, ua[r]);
            }
        }
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const uint8_t gbv = gW[g * (kPrefillGroupSize / 2u) + lane];
        const uint8_t ubv = uW[g * (kPrefillGroupSize / 2u) + lane];
        const float gs = float(gS[g]), gb = float(gB[g]);
        const float us = float(uS[g]), ub = float(uB[g]);
        for (uint r = 0; r < R; ++r) {
            if (r < n) {
                const float x0 = float(xrow[r][g * kPrefillGroupSize + lane * 2u]);
                const float x1 = float(xrow[r][g * kPrefillGroupSize + lane * 2u + 1u]);
                const float gd = fma(float(uint(gbv >> 4)), x1, float(uint(gbv & 0x0Fu)) * x0);
                const float ud = fma(float(uint(ubv >> 4)), x1, float(uint(ubv & 0x0Fu)) * x0);
                ga[r] = fma(gs, gd, ga[r]); ga[r] = fma(gb, x0 + x1, ga[r]);
                ua[r] = fma(us, ud, ua[r]); ua[r] = fma(ub, x0 + x1, ua[r]);
            }
        }
    }
    for (uint r = 0; r < R; ++r) {
        if (r < n) {
            const float gv = simd_sum(ga[r]);
            const float uv = simd_sum(ua[r]);
            if (lane == 0) {
                acts[act_rows[r] * p.F + f] = half(prefill_gelu_pytorch_tanh(gv) * uv);
            }
        }
    }
}

/// Down weight rows `d0` and `d0 + 1` of `expert` applied to `n` rows: row r
/// reads `acts` row `act_rows[r]` and writes `route_partials` row
/// `out_rows[r]`. Two rows per SIMD share each activation load.
static inline void verify_down_rows(
    device half* route_partials, device const half* acts, device const uint8_t* expert,
    constant PrefillGroupedRoutedMoEStreamedParamsMSL& p, uint d0, uint lane, uint n,
    thread const uint* act_rows, thread const uint* out_rows
) {
    const uint R = FC_VERIFY_ROWS;
    const uint F = p.F;
    const uint n_groups = F / kPrefillGroupSize;
    const uint d1 = min(d0 + 1u, p.D - 1u);
    device const uint8_t* W0 = expert + p.down_W_off + d0 * (F / 2u);
    device const uint8_t* W1 = expert + p.down_W_off + d1 * (F / 2u);
    device const bfloat* S = reinterpret_cast<device const bfloat*>(expert + p.down_s_off);
    device const bfloat* B = reinterpret_cast<device const bfloat*>(expert + p.down_b_off);

    float acc0[kVerifyMaxRows], acc1[kVerifyMaxRows];
    for (uint r = 0; r < R; ++r) { acc0[r] = 0.0f; acc1[r] = 0.0f; }

    // F = 704 is 11 groups: the last 4-group block runs with its fourth
    // group's lanes masked, so every group takes the vector path.
    const uint blocks = (n_groups + 3u) / 4u;
    for (uint blk = 0; blk < blocks; ++blk) {
        const uint g = blk * 4u + (lane >> 3);
        if (g >= n_groups) continue;
        const uint byte_base = blk * 128u + lane * 4u;
        device const ushort* wp0 = (device const ushort*)(W0 + byte_base);
        device const ushort* wp1 = (device const ushort*)(W1 + byte_base);
        float q0[8], q1[8];
        verify_unpack8(uint(wp0[0]) | (uint(wp0[1]) << 16), q0);
        verify_unpack8(uint(wp1[0]) | (uint(wp1[1]) << 16), q1);
        const float s0 = float(S[d0 * n_groups + g]), b0 = float(B[d0 * n_groups + g]);
        const float s1 = float(S[d1 * n_groups + g]), b1 = float(B[d1 * n_groups + g]);
        const uint elem = byte_base * 2u;
        for (uint r = 0; r < R; ++r) {
            if (r < n) {
                device const half* a = acts + act_rows[r] * F;
                const half4 xa = *((device const half4*)(a + elem));
                const half4 xb = *((device const half4*)(a + elem + 4u));
                float sum;
                const float dot0 = verify_dot8(q0, xa, xb, sum);
                const float dot1 = verify_dot8(q1, xa, xb, sum);
                acc0[r] = fma(s0, dot0, acc0[r]); acc0[r] = fma(b0, sum, acc0[r]);
                acc1[r] = fma(s1, dot1, acc1[r]); acc1[r] = fma(b1, sum, acc1[r]);
            }
        }
    }
    for (uint r = 0; r < R; ++r) {
        if (r < n) {
            const float v0 = simd_sum(acc0[r]);
            const float v1 = simd_sum(acc1[r]);
            if (lane == 0) {
                device half* out = route_partials + out_rows[r] * p.D;
                out[d0] = half(v0);
                if (d0 + 1u < p.D) out[d0 + 1u] = half(v1);
            }
        }
    }
}

kernel void verify_routed_gate_up_act(
    device const half*                                   hidden       [[buffer(0)]],
    device const PrefillTokenExpertPairMSL*              sorted_pairs [[buffer(1)]],
    device half*                                         acts         [[buffer(7)]],
    device const PrefillStreamedRoutedBlobsMSL&          routed       [[buffer(9)]],
    constant PrefillGroupedRoutedMoEStreamedParamsMSL&   p            [[buffer(10)]],
    constant uint2*                                      segments     [[buffer(11)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint slot = tg.y;
    const uint f = tg.x * 8u + sg;
    if (slot >= p.live_expert_count || f >= p.F) return;
    const uint2 seg = segments[slot];
    const uint n = min(seg.y, FC_VERIFY_ROWS);
    if (n == 0) return;
    uint tokens[kVerifyMaxRows], act_rows[kVerifyMaxRows];
    for (uint r = 0; r < n; ++r) { tokens[r] = sorted_pairs[seg.x + r].token; act_rows[r] = seg.x + r; }
    verify_gate_up_act_rows(hidden, acts, routed.blob[slot], p, f, lane, n, tokens, act_rows);
}

kernel void verify_routed_down(
    device const PrefillTokenExpertPairMSL*              sorted_pairs   [[buffer(1)]],
    device half*                                         route_partials [[buffer(5)]],
    device const half*                                   acts           [[buffer(7)]],
    device const PrefillStreamedRoutedBlobsMSL&          routed         [[buffer(9)]],
    constant PrefillGroupedRoutedMoEStreamedParamsMSL&   p              [[buffer(10)]],
    constant uint2*                                      segments       [[buffer(11)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint slot = tg.y;
    const uint d0 = (tg.x * 8u + sg) * 2u;
    if (slot >= p.live_expert_count || d0 >= p.D) return;
    const uint2 seg = segments[slot];
    const uint n = min(seg.y, FC_VERIFY_ROWS);
    if (n == 0) return;
    uint act_rows[kVerifyMaxRows], out_rows[kVerifyMaxRows];
    for (uint r = 0; r < n; ++r) {
        const PrefillTokenExpertPairMSL pair = sorted_pairs[seg.x + r];
        act_rows[r] = seg.x + r;
        out_rows[r] = pair.token * p.top_k + pair.rank;
    }
    verify_down_rows(route_partials, acts, routed.blob[slot], p, d0, lane, n, act_rows, out_rows);
}

// ============================================================================
// GPU-routed verify experts: the same row math over a bound list of experts
// fixed before the routes are known. `verify_bound_plan` finds each bound
// expert's routes once and compacts the routed ones, then the two expert
// kernels run indirectly over exactly those. A route to an expert outside the
// list is left for the host to patch. Activations and partials are indexed
// by route (row * top_k + rank).
// ============================================================================
struct VerifyBoundExpertMSL {
    device const uint8_t* blob;
    uint expert;
    uint pad;
};

struct VerifyRoutedExpertMSL {
    device const uint8_t* blob;
    uint n;
    uint pad;
    uint routes[kVerifyMaxRows];
};

constant constexpr uint kVerifyRowClasses = 4;     // up to 2, 4, 8, 16 rows
constant constexpr uint kVerifyClassEntries = 128;

/// One thread per bound expert: finds the expert's routes once and lists the
/// routed ones by row class (the power of two at or above its row count, at
/// least 2), so
/// each class runs a kernel specialized for it. `counters` holds the classes'
/// counts, `routed` their lists of `kVerifyClassEntries` entries each, and
/// `args` per class the two kernels' indirect grids.
kernel void verify_bound_plan(
    device const uint*                  route_ids   [[buffer(1)]],
    device const VerifyBoundExpertMSL*  bound       [[buffer(2)]],
    constant uint&                      route_count [[buffer(3)]],
    constant uint&                      bound_count [[buffer(4)]],
    device atomic_uint*                 counters    [[buffer(6)]],
    device VerifyRoutedExpertMSL*       routed      [[buffer(8)]],
    device uint*                        args        [[buffer(12)]],
    constant PrefillGroupedRoutedMoEStreamedParamsMSL& p [[buffer(10)]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    if (tid < kVerifyRowClasses) atomic_store_explicit(&counters[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_device);
    for (uint b = tid; b < bound_count; b += threads) {
        const uint expert = bound[b].expert;
        uint routes[kVerifyMaxRows];
        uint n = 0;
        for (uint i = 0; i < route_count; ++i) {
            if (route_ids[i] == expert && n < kVerifyMaxRows) { routes[n] = i; ++n; }
        }
        if (n == 0) continue;
        const uint c = n <= 2u ? 0u : n <= 4u ? 1u : n <= 8u ? 2u : 3u;
        const uint slot = c * kVerifyClassEntries + atomic_fetch_add_explicit(&counters[c], 1u, memory_order_relaxed);
        routed[slot].blob = bound[b].blob;
        routed[slot].n = n;
        for (uint r = 0; r < n; ++r) routed[slot].routes[r] = routes[r];
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (tid < kVerifyRowClasses) {
        const uint count = atomic_load_explicit(&counters[tid], memory_order_relaxed);
        device uint* a = args + tid * 6u;
        a[0] = (p.F + 7u) / 8u;             a[1] = count; a[2] = 1u;
        a[3] = ((p.D + 1u) / 2u + 7u) / 8u; a[4] = count; a[5] = 1u;
    }
}

kernel void verify_bound_gate_up_act(
    device const half*                                   hidden      [[buffer(0)]],
    device half*                                         acts        [[buffer(7)]],
    device const VerifyRoutedExpertMSL*                  routed      [[buffer(8)]],
    constant PrefillGroupedRoutedMoEStreamedParamsMSL&   p           [[buffer(10)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint f = tg.x * 8u + sg;
    if (f >= p.F) return;
    device const VerifyRoutedExpertMSL& e = routed[tg.y];
    const uint n = min(e.n, FC_VERIFY_ROWS);
    uint tokens[kVerifyMaxRows], routes[kVerifyMaxRows];
    for (uint r = 0; r < n; ++r) { routes[r] = e.routes[r]; tokens[r] = routes[r] / p.top_k; }
    verify_gate_up_act_rows(hidden, acts, e.blob, p, f, lane, n, tokens, routes);
}

kernel void verify_bound_down(
    device half*                                         route_partials [[buffer(5)]],
    device const half*                                   acts           [[buffer(7)]],
    device const VerifyRoutedExpertMSL*                  routed         [[buffer(8)]],
    constant PrefillGroupedRoutedMoEStreamedParamsMSL&   p              [[buffer(10)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    const uint d0 = (tg.x * 8u + sg) * 2u;
    if (d0 >= p.D) return;
    device const VerifyRoutedExpertMSL& e = routed[tg.y];
    const uint n = min(e.n, FC_VERIFY_ROWS);
    uint routes[kVerifyMaxRows];
    for (uint r = 0; r < n; ++r) routes[r] = e.routes[r];
    verify_down_rows(route_partials, acts, e.blob, p, d0, lane, n, routes, routes);
}
