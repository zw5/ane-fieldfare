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

static inline uint prefill_streamed_local_expert_id(
    constant PrefillGroupedRoutedMoEStreamedParamsMSL& p,
    uint slot
) {
    switch (slot) {
        case 0: return p.local_expert_0;
        case 1: return p.local_expert_1;
        case 2: return p.local_expert_2;
        case 3: return p.local_expert_3;
        case 4: return p.local_expert_4;
        case 5: return p.local_expert_5;
        case 6: return p.local_expert_6;
        case 7: return p.local_expert_7;
        case 8: return p.local_expert_8;
        case 9: return p.local_expert_9;
        case 10: return p.local_expert_10;
        case 11: return p.local_expert_11;
        case 12: return p.local_expert_12;
        case 13: return p.local_expert_13;
        case 14: return p.local_expert_14;
        default: return p.local_expert_15;
    }
}

static inline float prefill_moe_int4_gemv_row_dev(
    device const uint8_t* W,
    device const bfloat* S,
    device const bfloat* B,
    device const half* x,
    uint row,
    uint N
) {
    const uint groups = N / kPrefillGroupSize;
    const uint row_bytes = N / 2u;
    device const uint8_t* W_row = W + row * row_bytes;
    device const bfloat* s_row = S + row * groups;
    device const bfloat* b_row = B + row * groups;

    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        const float scale = float(s_row[g]);
        const float bias = float(b_row[g]);
        device const uint8_t* Wg = W_row + g * (kPrefillGroupSize / 2u);
        device const half* xg = x + g * kPrefillGroupSize;
        float dot_qx = 0.0f;
        float sum_x = 0.0f;
        for (uint k = 0; k < kPrefillGroupSize / 2u; ++k) {
            const uint8_t packed = Wg[k];
            const float x0 = float(xg[2u * k]);
            const float x1 = float(xg[2u * k + 1u]);
            dot_qx = fma(float(uint(packed & 0x0Fu)), x0, dot_qx);
            dot_qx = fma(float(uint(packed >> 4)), x1, dot_qx);
            sum_x += x0 + x1;
        }
        acc = fma(scale, dot_qx, acc);
        acc = fma(bias, sum_x, acc);
    }
    return acc;
}

static inline float prefill_moe_int4_gemv_row_tg(
    device const uint8_t* W,
    device const bfloat* S,
    device const bfloat* B,
    threadgroup const half* x,
    uint row,
    uint N
) {
    const uint groups = N / kPrefillGroupSize;
    const uint row_bytes = N / 2u;
    device const uint8_t* W_row = W + row * row_bytes;
    device const bfloat* s_row = S + row * groups;
    device const bfloat* b_row = B + row * groups;

    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        const float scale = float(s_row[g]);
        const float bias = float(b_row[g]);
        device const uint8_t* Wg = W_row + g * (kPrefillGroupSize / 2u);
        threadgroup const half* xg = x + g * kPrefillGroupSize;
        float dot_qx = 0.0f;
        float sum_x = 0.0f;
        for (uint k = 0; k < kPrefillGroupSize / 2u; ++k) {
            const uint8_t packed = Wg[k];
            const float x0 = float(xg[2u * k]);
            const float x1 = float(xg[2u * k + 1u]);
            dot_qx = fma(float(uint(packed & 0x0Fu)), x0, dot_qx);
            dot_qx = fma(float(uint(packed >> 4)), x1, dot_qx);
            sum_x += x0 + x1;
        }
        acc = fma(scale, dot_qx, acc);
        acc = fma(bias, sum_x, acc);
    }
    return acc;
}

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

kernel void prefill_grouped_routed_moe_batched_phase1(
    device const half*                                   hidden               [[buffer(0)]],
    device const PrefillTokenExpertPairMSL*              sorted_pairs         [[buffer(1)]],
    device half*                                         gate_up_act_scratch  [[buffer(7)]],
    device const PrefillStreamedRoutedBlobsMSL&          routed               [[buffer(9)]],
    constant PrefillGroupedRoutedMoEStreamedParamsMSL&   p                    [[buffer(10)]],
    uint2                                                gid                  [[thread_position_in_grid]]
) {
    const uint f = gid.x;
    const uint pair_local = gid.y;
    if (f >= p.F || pair_local >= p.pair_count) return;

    const PrefillTokenExpertPairMSL pair = sorted_pairs[p.pair_start + pair_local];
    uint local_slot = kPrefillMaxTileExperts;
    for (uint slot = 0; slot < p.live_expert_count; ++slot) {
        if (prefill_streamed_local_expert_id(p, slot) == pair.expert) {
            local_slot = slot;
            break;
        }
    }
    if (local_slot >= p.live_expert_count) return;

    device const uint8_t* expert = routed.blob[local_slot];
    device const half* x = hidden + pair.token * p.hidden_stride_elements;
    device const uint8_t* gate_W = expert + p.gate_W_off;
    device const bfloat* gate_s = reinterpret_cast<device const bfloat*>(expert + p.gate_s_off);
    device const bfloat* gate_b = reinterpret_cast<device const bfloat*>(expert + p.gate_b_off);
    device const uint8_t* up_W = expert + p.up_W_off;
    device const bfloat* up_s = reinterpret_cast<device const bfloat*>(expert + p.up_s_off);
    device const bfloat* up_b = reinterpret_cast<device const bfloat*>(expert + p.up_b_off);

    const float gate = prefill_moe_int4_gemv_row_dev(gate_W, gate_s, gate_b, x, f, p.D);
    const float up = prefill_moe_int4_gemv_row_dev(up_W, up_s, up_b, x, f, p.D);
    const uint row_elements = p.pair_count * p.F;
    const uint index = pair_local * p.F + f;
    gate_up_act_scratch[index] = half(gate);
    gate_up_act_scratch[row_elements + index] = half(up);
    gate_up_act_scratch[2u * row_elements + index] =
        half(prefill_gelu_pytorch_tanh(gate) * up);
}

kernel void prefill_grouped_routed_moe_batched_down(
    device const PrefillTokenExpertPairMSL*              sorted_pairs         [[buffer(1)]],
    device half*                                         route_partials       [[buffer(5)]],
    device const half*                                   gate_up_act_scratch  [[buffer(7)]],
    device half*                                         down_scratch         [[buffer(8)]],
    device const PrefillStreamedRoutedBlobsMSL&          routed               [[buffer(9)]],
    constant PrefillGroupedRoutedMoEStreamedParamsMSL&   p                    [[buffer(10)]],
    uint2                                                gid                  [[thread_position_in_grid]]
) {
    const uint d = gid.x;
    const uint pair_local = gid.y;
    if (d >= p.D || pair_local >= p.pair_count) return;

    const PrefillTokenExpertPairMSL pair = sorted_pairs[p.pair_start + pair_local];
    uint local_slot = kPrefillMaxTileExperts;
    for (uint slot = 0; slot < p.live_expert_count; ++slot) {
        if (prefill_streamed_local_expert_id(p, slot) == pair.expert) {
            local_slot = slot;
            break;
        }
    }
    if (local_slot >= p.live_expert_count) return;

    device const uint8_t* expert = routed.blob[local_slot];
    device const uint8_t* down_W = expert + p.down_W_off;
    device const bfloat* down_s = reinterpret_cast<device const bfloat*>(expert + p.down_s_off);
    device const bfloat* down_b = reinterpret_cast<device const bfloat*>(expert + p.down_b_off);
    device const half* act = gate_up_act_scratch + 2u * p.pair_count * p.F + pair_local * p.F;
    const half value = half(prefill_moe_int4_gemv_row_dev(down_W, down_s, down_b, act, d, p.F));
    down_scratch[pair_local * p.D + d] = value;
    route_partials[(pair.token * p.top_k + pair.rank) * p.D + d] = value;
}

kernel void prefill_dequant_int4_qmm_f16_block(
    device const uint8_t* W      [[buffer(0)]],
    device const bfloat*  scales [[buffer(1)]],
    device const bfloat*  biases [[buffer(2)]],
    device const half*    X      [[buffer(3)]],
    device half*          Y      [[buffer(4)]],
    constant uint&        T      [[buffer(5)]],
    constant uint&        N      [[buffer(6)]],
    constant uint&        K      [[buffer(7)]],
    uint2                 tid    [[thread_position_in_threadgroup]],
    uint2                 tgid   [[threadgroup_position_in_grid]]
) {
    const uint n = tgid.x * 8u + tid.x;
    const uint t = tgid.y * 8u + tid.y;
    if (t >= T || n >= N) return;

    const uint groups = K / kPrefillGroupSize;
    const uint row_bytes = K / 2u;
    device const uint8_t* w_row = W + n * row_bytes;
    device const bfloat* s_row = scales + n * groups;
    device const bfloat* b_row = biases + n * groups;
    device const half* x_row = X + t * K;

    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        const float scale = float(s_row[g]);
        const float bias = float(b_row[g]);
        const uint group_base = g * kPrefillGroupSize;
        for (uint kk = 0; kk < kPrefillGroupSize; ++kk) {
            const uint k = group_base + kk;
            const uint8_t packed = w_row[k >> 1];
            const uint q = (k & 1u) == 0u ? uint(packed & 0x0Fu) : uint(packed >> 4);
            const float w = fma(float(q), scale, bias);
            acc = fma(w, float(x_row[k]), acc);
        }
    }
    Y[t * N + n] = half(acc);
}

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

kernel void prefill_attention_params_smoke(
    constant PrefillAttentionParams& p [[buffer(0)]],
    device uint* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= 13u) return;
    switch (id) {
    case 0u: out[id] = p.startPosition; break;
    case 1u: out[id] = p.queryCount; break;
    case 2u: out[id] = p.headDim; break;
    case 3u: out[id] = p.numQHeads; break;
    case 4u: out[id] = p.numKVHeads; break;
    case 5u: out[id] = p.kvValidCount; break;
    case 6u: out[id] = p.slidingWindow; break;
    case 7u: out[id] = p.kvTokenStrideElements; break;
    case 8u: out[id] = p.qTokenStrideElements; break;
    case 9u: out[id] = p.oTokenStrideElements; break;
    case 10u: out[id] = as_type<uint>(p.scale); break;
    case 11u: out[id] = p.bidirectionalBlockStart; break;
    default: out[id] = p.bidirectionalBlockEnd; break;
    }
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

static inline float prefill_attention_tg_sum_single_bank(
    float value,
    uint lane,
    uint simd_group,
    uint simdgroups,
    threadgroup float* partial
) {
    const float result = prefill_attention_tg_sum(
        value, lane, simd_group, simdgroups, partial);
    // A single scratch bank needs an explicit reader-to-next-writer edge.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
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

constant constexpr int kPrefillTensorOpsOutputs = 8;
constant constexpr int kPrefillTensorOpsKeys = 64;
constant constexpr int kPrefillTensorOpsHeadDim = 512;

static inline void attention_prefill_full_tensorops_2d_validity_v2_impl(
    device const half* Q,
    device half* K,
    device half* V,
    device half* O,
    constant PrefillAttentionParams& p,
    uint3 tg,
    uint lid,
    uint threads,
    threadgroup half* query_tile,
    threadgroup float* score_tile,
    threadgroup float* weight_tile,
    threadgroup float* row_max,
    threadgroup float* row_sum,
    threadgroup float* row_old_scale
) {
    constexpr auto qk_desc = matmul2d_descriptor(
        kPrefillTensorOpsOutputs,
        kPrefillTensorOpsKeys,
        kPrefillTensorOpsHeadDim,
        false, true, false);
    constexpr auto pv_desc = matmul2d_descriptor(
        kPrefillTensorOpsOutputs,
        kPrefillTensorOpsHeadDim,
        kPrefillTensorOpsKeys,
        false, false, false);
    matmul2d<qk_desc, execution_simdgroups<4>> qk_op;
    matmul2d<pv_desc, execution_simdgroups<4>> pv_op;

    using device_half_tensor =
        tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using threadgroup_half_tensor =
        tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;
    using threadgroup_float_tensor =
        tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>;

    const uint query_start = tg.x;
    const uint qh_start = tg.y * uint(kPrefillTensorOpsOutputs);
    const uint valid_query_rows =
        min(1u, p.queryCount - min(query_start, p.queryCount));
    const uint q_per_kv = p.numQHeads / p.numKVHeads;
    const uint kvh = qh_start / q_per_kv;

    for (uint linear = lid;
         linear < uint(kPrefillTensorOpsOutputs * kPrefillTensorOpsHeadDim);
         linear += threads) {
        const uint output_row =
            linear / uint(kPrefillTensorOpsHeadDim);
        const uint d = linear % uint(kPrefillTensorOpsHeadDim);
        if (valid_query_rows != 0u) {
            query_tile[linear] = Q[
                query_start * p.qTokenStrideElements
                + (qh_start + output_row) * p.headDim
                + d];
        } else {
            query_tile[linear] = half(0.0f);
        }
    }
    if (lid < uint(kPrefillTensorOpsOutputs)) {
        row_max[lid] = -INFINITY;
        row_sum[lid] = 0.0f;
        row_old_scale[lid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup_half_tensor query_tensor(
        query_tile,
        dextents<int32_t, 2>(
            kPrefillTensorOpsHeadDim,
            kPrefillTensorOpsOutputs),
        array<int32_t, 2>({1, kPrefillTensorOpsHeadDim}));
    threadgroup_float_tensor weight_tensor(
        weight_tile,
        dextents<int32_t, 2>(
            kPrefillTensorOpsKeys,
            kPrefillTensorOpsOutputs),
        array<int32_t, 2>({1, kPrefillTensorOpsKeys}));
    device_half_tensor key_tensor(
        K + kvh * p.headDim,
        dextents<int32_t, 2>(
            int32_t(p.headDim),
            int32_t(p.kvValidCount)),
        array<int32_t, 2>({1, int32_t(p.kvTokenStrideElements)}));
    device_half_tensor value_tensor(
        V + kvh * p.headDim,
        dextents<int32_t, 2>(
            int32_t(p.headDim),
            int32_t(p.kvValidCount)),
        array<int32_t, 2>({1, int32_t(p.kvTokenStrideElements)}));

    auto query_slice = query_tensor.slice(0, 0);
    auto first_value_slice = value_tensor.slice(0, 0);
    auto output_accumulator =
        pv_op.get_destination_cooperative_tensor<
            decltype(weight_tensor), decltype(first_value_slice), float>();
    #pragma clang loop unroll(full)
    for (int element = 0;
         element < output_accumulator.get_capacity();
         ++element) {
        if (output_accumulator.is_valid_element(element)) {
            output_accumulator[element] = 0.0f;
        }
    }

    // This full-attention kernel starts at key zero and ignores slidingWindow.
    // The Swift selector must dispatch it only when every prior key is visible.
    const uint last =
        min(p.kvValidCount, p.startPosition + query_start + valid_query_rows);
    for (uint key_start = 0u;
         key_start < last;
         key_start += uint(kPrefillTensorOpsKeys)) {
        auto key_slice = key_tensor.slice(0, int32_t(key_start));
        auto score_product =
            qk_op.get_destination_cooperative_tensor<
                decltype(query_slice), decltype(key_slice), float>();
        #pragma clang loop unroll(full)
        for (int element = 0;
             element < score_product.get_capacity();
             ++element) {
            if (score_product.is_valid_element(element)) {
                score_product[element] = 0.0f;
            }
        }
        qk_op.run(query_slice, key_slice, score_product);

        #pragma clang loop unroll(full)
        for (int element = 0;
             element < score_product.get_capacity();
             ++element) {
            if (!score_product.is_valid_element(element)) continue;
            const auto position =
                score_product.get_multidimensional_index(element);
            const uint key_column = uint(position[0]);
            const uint output_row = uint(position[1]);
            score_tile[
                output_row * uint(kPrefillTensorOpsKeys) + key_column] =
                score_product[element] * p.scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid < uint(kPrefillTensorOpsOutputs)) {
            const uint output_row = lid;
            const uint causal_last = valid_query_rows != 0u
                ? min(p.kvValidCount, p.startPosition + query_start + 1u)
                : 0u;
            const uint visible =
                causal_last > key_start
                    ? min(uint(kPrefillTensorOpsKeys), causal_last - key_start)
                    : 0u;

            float tile_max = -INFINITY;
            for (uint key = 0u; key < visible; ++key) {
                tile_max = max(
                    tile_max,
                    score_tile[
                        output_row * uint(kPrefillTensorOpsKeys) + key]);
            }
            const float next_max = max(row_max[output_row], tile_max);
            const float old_scale = row_sum[output_row] > 0.0f
                ? fast::exp(row_max[output_row] - next_max)
                : 0.0f;
            float tile_sum = 0.0f;
            for (uint key = 0u;
                 key < uint(kPrefillTensorOpsKeys);
                 ++key) {
                const float weight = key < visible
                    ? fast::exp(
                        score_tile[
                            output_row * uint(kPrefillTensorOpsKeys) + key]
                        - next_max)
                    : 0.0f;
                weight_tile[
                    output_row * uint(kPrefillTensorOpsKeys) + key] = weight;
                tile_sum += weight;
            }
            row_old_scale[output_row] = old_scale;
            row_sum[output_row] =
                row_sum[output_row] * old_scale + tile_sum;
            row_max[output_row] = next_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto value_slice = value_tensor.slice(0, int32_t(key_start));
        auto output_product =
            pv_op.get_destination_cooperative_tensor<
                decltype(weight_tensor), decltype(value_slice), float>();
        #pragma clang loop unroll(full)
        for (int element = 0;
             element < output_product.get_capacity();
             ++element) {
            if (output_product.is_valid_element(element)) {
                output_product[element] = 0.0f;
            }
        }
        pv_op.run(weight_tensor, value_slice, output_product);
        #pragma clang loop unroll(full)
        for (int element = 0;
             element < output_accumulator.get_capacity();
             ++element) {
            if (!output_accumulator.is_valid_element(element)
                || !output_product.is_valid_element(element)) {
                continue;
            }
            const auto position =
                output_accumulator.get_multidimensional_index(element);
            const uint output_row = uint(position[1]);
            output_accumulator[element] =
                fma(
                    1.0f,
                    output_product[element],
                    output_accumulator[element] * row_old_scale[output_row]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (int element = 0;
         element < output_accumulator.get_capacity();
         ++element) {
        if (!output_accumulator.is_valid_element(element)) continue;
        const auto position =
            output_accumulator.get_multidimensional_index(element);
        const uint d = uint(position[0]);
        const uint output_row = uint(position[1]);
        if (valid_query_rows != 0u) {
            const float denominator = row_sum[output_row];
            O[
                query_start * p.oTokenStrideElements
                + (qh_start + output_row) * p.headDim
                + d] = denominator > 0.0f
                    ? half(output_accumulator[element] / denominator)
                    : half(0.0f);
        }
    }
}

kernel void attention_prefill_full_tensorops_2d_validity_v2(
    device const half* Q [[buffer(0)]],
    device half* K [[buffer(1)]],
    device half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    constant PrefillAttentionParams& p [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint3 threads3 [[threads_per_threadgroup]]
) {
    threadgroup half query_tile[
        kPrefillTensorOpsOutputs * kPrefillTensorOpsHeadDim];
    threadgroup float score_tile[
        kPrefillTensorOpsOutputs * kPrefillTensorOpsKeys];
    threadgroup float weight_tile[
        kPrefillTensorOpsOutputs * kPrefillTensorOpsKeys];
    threadgroup float row_max[kPrefillTensorOpsOutputs];
    threadgroup float row_sum[kPrefillTensorOpsOutputs];
    threadgroup float row_old_scale[kPrefillTensorOpsOutputs];
    attention_prefill_full_tensorops_2d_validity_v2_impl(
        Q, K, V, O, p, tg, lid, threads3.x,
        query_tile, score_tile, weight_tile,
        row_max, row_sum, row_old_scale);
}

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
    const uint R = FC_VERIFY_ROWS;
    const uint slot = tg.y;
    const uint f = tg.x * 8u + sg;
    if (slot >= p.live_expert_count || f >= p.F) return;
    const uint2 seg = segments[slot];
    const uint n = min(seg.y, R);
    if (n == 0) return;

    device const uint8_t* expert = routed.blob[slot];
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
        xrow[r] = hidden + sorted_pairs[seg.x + min(r, n - 1u)].token * p.hidden_stride_elements;
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
                acts[(seg.x + r) * p.F + f] = half(prefill_gelu_pytorch_tanh(gv) * uv);
            }
        }
    }
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
    // Two down rows per SIMD share each activation load and conversion.
    const uint R = FC_VERIFY_ROWS;
    const uint slot = tg.y;
    const uint d0 = (tg.x * 8u + sg) * 2u;
    if (slot >= p.live_expert_count || d0 >= p.D) return;
    const uint2 seg = segments[slot];
    const uint n = min(seg.y, R);
    if (n == 0) return;

    device const uint8_t* expert = routed.blob[slot];
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
                device const half* a = acts + (seg.x + r) * F;
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
                const PrefillTokenExpertPairMSL pair = sorted_pairs[seg.x + r];
                device half* out = route_partials + (pair.token * p.top_k + pair.rank) * p.D;
                out[d0] = half(v0);
                if (d0 + 1u < p.D) out[d0 + 1u] = half(v1);
            }
        }
    }
}
