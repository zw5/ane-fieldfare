#include <metal_stdlib>
using namespace metal;

constant constexpr uint kMoEGroupSize = 64;
constant constexpr uint kMaxStreamedExperts = 8;
constant constexpr float kGeluSqrt2OverPi = 0.7978845608028654f;
constant constexpr float kGeluCubicCoeff = 0.044715f;

constant uint FC_ROUTER_NUM_EXPERTS [[function_constant(40)]];
constant uint FC_ROUTER_D [[function_constant(41)]];
constant bool FC_ROUTER_USE_FC [[function_constant(43)]];

constant uint FC_MOE_D [[function_constant(0)]];
constant uint FC_MOE_F [[function_constant(1)]];
constant uint FC_MOE_TOP_K [[function_constant(2)]];
constant bool FC_MOE_USE_FC [[function_constant(3)]];

static inline uint router_fc_num_experts(constant uint& num_experts) {
    return (is_function_constant_defined(FC_ROUTER_USE_FC) &&
            FC_ROUTER_USE_FC &&
            is_function_constant_defined(FC_ROUTER_NUM_EXPERTS))
        ? FC_ROUTER_NUM_EXPERTS
        : num_experts;
}

static inline uint router_fc_d(constant uint& D) {
    return (is_function_constant_defined(FC_ROUTER_USE_FC) &&
            FC_ROUTER_USE_FC &&
            is_function_constant_defined(FC_ROUTER_D))
        ? FC_ROUTER_D
        : D;
}

static inline uint moe_fc_d(constant uint& D) {
    return (is_function_constant_defined(FC_MOE_USE_FC) &&
            FC_MOE_USE_FC &&
            is_function_constant_defined(FC_MOE_D)) ? FC_MOE_D : D;
}

static inline uint moe_fc_f(constant uint& F) {
    return (is_function_constant_defined(FC_MOE_USE_FC) &&
            FC_MOE_USE_FC &&
            is_function_constant_defined(FC_MOE_F)) ? FC_MOE_F : F;
}

static inline uint moe_fc_top_k(constant uint& top_k) {
    return (is_function_constant_defined(FC_MOE_USE_FC) &&
            FC_MOE_USE_FC &&
            is_function_constant_defined(FC_MOE_TOP_K)) ? FC_MOE_TOP_K : top_k;
}

static inline float gelu_pytorch_tanh(float x) {
    const float x3 = x * x * x;
    float inner = kGeluSqrt2OverPi * (x + kGeluCubicCoeff * x3);
    // Clamping avoids Metal tanh producing NaN at large magnitudes while being
    // equivalent to the saturated result at FP32 precision.
    inner = clamp(inner, -20.0f, 20.0f);
    return 0.5f * x * (1.0f + tanh(inner));
}

struct ExpertOffsets {
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

struct RoutedBlobs {
    device const uint8_t* blob[kMaxStreamedExperts];
};

static inline void router_gemv_gemma4_body(
    device const uint8_t* W,
    device const bfloat* scales,
    device const bfloat* biases,
    device const half* hidden,
    device const bfloat* effective_scale,
    device float* out_logits,
    constant uint& num_experts,
    constant uint& D,
    uint tg_idx,
    uint sg_idx,
    uint lane
) {
    const uint NE = router_fc_num_experts(num_experts);
    const uint DD = router_fc_d(D);
    // Four SIMD groups, one expert row each, per threadgroup.
    const uint e = tg_idx * 4u + sg_idx;
    if (e >= NE) return;

    const uint n_groups = DD / kMoEGroupSize;
    device const uint8_t* W_row = W + uint(e) * DD;
    device const bfloat* s_row = scales + uint(e) * n_groups;
    device const bfloat* b_row = biases + uint(e) * n_groups;

    float acc = 0.0f;
    for (uint g = 0; g < n_groups; ++g) {
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint idx = g * kMoEGroupSize + lane * 2u;
        const float q0 = float(uint(W_row[idx]));
        const float q1 = float(uint(W_row[idx + 1u]));
        const float x0 = float(hidden[idx]) * float(effective_scale[idx]);
        const float x1 = float(hidden[idx + 1u]) * float(effective_scale[idx + 1u]);
        acc = fma(s, q0 * x0 + q1 * x1, acc);
        acc = fma(b, x0 + x1, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) out_logits[e] = acc;
}

kernel void router_gemv_gemma4_r4(
    device const uint8_t* W [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const half* hidden [[buffer(3)]],
    device const bfloat* effective_scale [[buffer(4)]],
    device float* out_logits [[buffer(5)]],
    constant uint& num_experts [[buffer(6)]],
    constant uint& D [[buffer(7)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    router_gemv_gemma4_body(W, scales, biases, hidden, effective_scale,
                            out_logits, num_experts, D, tg_idx, sg_idx, lane);
}

// The router GEMV over two weight sets in one dispatch: this layer's router
// into `out_logits`, then the next layer's guess into `out_logits2`. Each row
// is the single router GEMV's, bitwise.
kernel void router_gemv_gemma4_pair_r4(
    device const uint8_t* W [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const half* hidden [[buffer(3)]],
    device const bfloat* effective_scale [[buffer(4)]],
    device float* out_logits [[buffer(5)]],
    constant uint& num_experts [[buffer(6)]],
    constant uint& D [[buffer(7)]],
    device const uint8_t* W2 [[buffer(8)]],
    device const bfloat* scales2 [[buffer(9)]],
    device const bfloat* biases2 [[buffer(10)]],
    device const bfloat* effective_scale2 [[buffer(11)]],
    device float* out_logits2 [[buffer(12)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint groups = (router_fc_num_experts(num_experts) + 3u) / 4u;
    if (tg_idx < groups) {
        router_gemv_gemma4_body(W, scales, biases, hidden, effective_scale,
                                out_logits, num_experts, D, tg_idx, sg_idx, lane);
    } else {
        router_gemv_gemma4_body(W2, scales2, biases2, hidden, effective_scale2,
                                out_logits2, num_experts, D, tg_idx - groups, sg_idx, lane);
    }
}

// router_gemv_gemma4_pair_r4 over the rows of a block: grid row y scores
// hidden row y into `[rows][num_experts]` logits.
kernel void router_gemv_gemma4_pair_rows_r4(
    device const uint8_t* W [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const half* hidden [[buffer(3)]],
    device const bfloat* effective_scale [[buffer(4)]],
    device float* out_logits [[buffer(5)]],
    constant uint& num_experts [[buffer(6)]],
    constant uint& D [[buffer(7)]],
    device const uint8_t* W2 [[buffer(8)]],
    device const bfloat* scales2 [[buffer(9)]],
    device const bfloat* biases2 [[buffer(10)]],
    device const bfloat* effective_scale2 [[buffer(11)]],
    device float* out_logits2 [[buffer(12)]],
    constant uint& hidden_stride [[buffer(13)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint NE = router_fc_num_experts(num_experts);
    const uint groups = (NE + 3u) / 4u;
    device const half* h = hidden + tg.y * hidden_stride;
    if (tg.x < groups) {
        router_gemv_gemma4_body(W, scales, biases, h, effective_scale,
                                out_logits + tg.y * NE, num_experts, D, tg.x, sg_idx, lane);
    } else {
        router_gemv_gemma4_body(W2, scales2, biases2, h, effective_scale2,
                                out_logits2 + tg.y * NE, num_experts, D, tg.x - groups, sg_idx, lane);
    }
}

static inline void router_topk_select_k8_simd_body(
    device const float* logits,
    device const bfloat* per_expert_scale,
    device uint* out_indices,
    device half* out_weights,
    uint NE,
    uint lane
) {
    float v[8];
    bool live[8];
    for (uint j = 0; j < 8; ++j) {
        const uint e = j * 32u + lane;
        live[j] = e < NE;
        v[j] = live[j] ? logits[e] : -INFINITY;
        // The serial kernel never takes a score at or below -inf (or NaN).
        live[j] = live[j] && v[j] > -INFINITY;
    }
    uint top_idx[8];
    float top_score[8];
    for (uint r = 0; r < 8; ++r) {
        float best = -INFINITY;
        uint best_e = 0xFFFFFFFFu;
        for (uint j = 0; j < 8; ++j) {
            const uint e = j * 32u + lane;
            if (live[j] && (v[j] > best || (v[j] == best && e < best_e))) {
                best = v[j];
                best_e = e;
            }
        }
        const float top = simd_max(best);
        const uint winner = simd_min(best == top ? best_e : 0xFFFFFFFFu);
        if (winner != 0xFFFFFFFFu && (winner & 31u) == lane) live[winner >> 5] = false;
        top_idx[r] = winner == 0xFFFFFFFFu ? 0u : winner;
        top_score[r] = winner == 0xFFFFFFFFu ? -INFINITY : top;
    }
    if (lane != 0) return;

    const float max_s = top_score[0];
    float sum_exp = 0.0f;
    float exps[8];
    for (uint i = 0; i < 8; ++i) {
        const float ex = fast::exp(top_score[i] - max_s);
        exps[i] = ex;
        sum_exp += ex;
    }
    for (uint i = 0; i < 8; ++i) {
        const uint expert_idx = top_idx[i];
        const float weight = exps[i] / sum_exp;
        out_indices[i] = expert_idx;
        out_weights[i] = half(weight * float(per_expert_scale[expert_idx]));
    }
}

// router_topk_select_k8 over one SIMD: eight rounds of a SIMD-wide argmax
// (higher score, then lower expert), so the selection, its order and the
// softmax are the serial kernel's.
kernel void router_topk_select_k8_simd(
    device const float* logits [[buffer(0)]],
    device const bfloat* per_expert_scale [[buffer(1)]],
    device uint* out_indices [[buffer(2)]],
    device half* out_weights [[buffer(3)]],
    constant uint& num_experts [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]]
) {
    router_topk_select_k8_simd_body(logits, per_expert_scale, out_indices, out_weights,
                                    router_fc_num_experts(num_experts), lane);
}

// One SIMD-group select per row of `[rows][num_experts]` logits.
kernel void router_topk_select_k8_simd_rows(
    device const float* logits [[buffer(0)]],
    device const bfloat* per_expert_scale [[buffer(1)]],
    device uint* out_indices [[buffer(2)]],
    device half* out_weights [[buffer(3)]],
    constant uint& num_experts [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint NE = router_fc_num_experts(num_experts);
    router_topk_select_k8_simd_body(logits + row * NE, per_expert_scale,
                                    out_indices + row * 8u, out_weights + row * 8u, NE, lane);
}

kernel void router_topk_select_k8(
    device const float* logits [[buffer(0)]],
    device const bfloat* per_expert_scale [[buffer(1)]],
    device uint* out_indices [[buffer(2)]],
    device half* out_weights [[buffer(3)]],
    constant uint& num_experts [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (tid != 0) return;
    const uint NE = router_fc_num_experts(num_experts);
    uint top_idx[8];
    float top_score[8];
    for (uint i = 0; i < 8; ++i) {
        top_idx[i] = 0u;
        top_score[i] = -INFINITY;
    }

    for (uint e = 0; e < NE; ++e) {
        const float s = logits[e];
        if (s <= top_score[7]) continue;
        uint pos = 8u;
        for (uint i = 0; i < 8; ++i) {
            if (s > top_score[i] || (s == top_score[i] && e < top_idx[i])) {
                pos = i;
                break;
            }
        }
        if (pos >= 8u) continue;
        for (uint i = 7; i > pos; --i) {
            top_idx[i] = top_idx[i - 1];
            top_score[i] = top_score[i - 1];
        }
        top_idx[pos] = e;
        top_score[pos] = s;
    }

    const float max_s = top_score[0];
    float sum_exp = 0.0f;
    float exps[8];
    for (uint i = 0; i < 8; ++i) {
        const float ex = fast::exp(top_score[i] - max_s);
        exps[i] = ex;
        sum_exp += ex;
    }
    for (uint i = 0; i < 8; ++i) {
        const uint expert_idx = top_idx[i];
        const float weight = exps[i] / sum_exp;
        out_indices[i] = expert_idx;
        out_weights[i] = half(weight * float(per_expert_scale[expert_idx]));
    }
}

// Each SIMD computes one affine INT4 row. Four adjacent groups are loaded as
// aligned 32-bit chunks; remaining groups use one byte per lane.
static inline float moe_int4_gemv_row_simd_dev_vec(
    device const uint8_t* W,
    device const bfloat* S,
    device const bfloat* B,
    device const half* x,
    uint row,
    uint N,
    uint lane
) {
    const uint n_groups = N / kMoEGroupSize;
    const uint row_bytes = N / 2;
    device const uint8_t* W_row = W + uint(row) * row_bytes;
    device const bfloat* s_row = S + uint(row) * n_groups;
    device const bfloat* b_row = B + uint(row) * n_groups;

    float acc = 0.0f;
    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        const uint w4 = *((device const uint*)(W_row + byte_base));
        const uint g = blk * 4u + (lane >> 3);
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint elem = byte_base * 2u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        const uint b0 = w4 & 0xFFu;
        const uint b1 = (w4 >> 8) & 0xFFu;
        const uint b2 = (w4 >> 16) & 0xFFu;
        const uint b3 = (w4 >> 24) & 0xFFu;
        const float e0 = float(xa.x), e1 = float(xa.y);
        const float e2 = float(xa.z), e3 = float(xa.w);
        const float e4 = float(xb.x), e5 = float(xb.y);
        const float e6 = float(xb.z), e7 = float(xb.w);
        float dot = 0.0f;
        dot = fma(float(b0 & 0x0Fu), e0, dot); dot = fma(float(b0 >> 4), e1, dot);
        dot = fma(float(b1 & 0x0Fu), e2, dot); dot = fma(float(b1 >> 4), e3, dot);
        dot = fma(float(b2 & 0x0Fu), e4, dot); dot = fma(float(b2 >> 4), e5, dot);
        dot = fma(float(b3 & 0x0Fu), e6, dot); dot = fma(float(b3 >> 4), e7, dot);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint8_t byte = W_row[g * (kMoEGroupSize / 2) + lane];
        const float x0 = float(x[g * kMoEGroupSize + lane * 2u]);
        const float x1 = float(x[g * kMoEGroupSize + lane * 2u + 1u]);
        float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
        dot = fma(float(uint(byte >> 4)), x1, dot);
        acc = fma(s, dot, acc);
        acc = fma(b, x0 + x1, acc);
    }
    return simd_sum(acc);
}

// Gate and up rows share activation loads. Two 16-bit loads assemble each
// 4-byte weight chunk because packed sub-tensor offsets need only be 2-byte aligned.
static inline float2 moe_int4_gate_up_rows_simd_dev_vec_u16load(
    device const uint8_t* gateW,
    device const bfloat* gateS,
    device const bfloat* gateB,
    device const uint8_t* upW,
    device const bfloat* upS,
    device const bfloat* upB,
    device const half* x,
    uint row,
    uint N,
    uint lane
) {
    const uint n_groups = N / kMoEGroupSize;
    const uint row_bytes = N / 2;
    device const uint8_t* gW_row = gateW + uint(row) * row_bytes;
    device const uint8_t* uW_row = upW + uint(row) * row_bytes;
    device const bfloat* gS_row = gateS + uint(row) * n_groups;
    device const bfloat* gB_row = gateB + uint(row) * n_groups;
    device const bfloat* uS_row = upS + uint(row) * n_groups;
    device const bfloat* uB_row = upB + uint(row) * n_groups;

    float g_acc = 0.0f;
    float u_acc = 0.0f;
    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        device const ushort* gp = (device const ushort*)(gW_row + byte_base);
        device const ushort* up = (device const ushort*)(uW_row + byte_base);
        const uint gw4 = uint(gp[0]) | (uint(gp[1]) << 16);
        const uint uw4 = uint(up[0]) | (uint(up[1]) << 16);
        const uint g = blk * 4u + (lane >> 3);
        const float gs = float(gS_row[g]);
        const float gb = float(gB_row[g]);
        const float us = float(uS_row[g]);
        const float ub = float(uB_row[g]);
        const uint elem = byte_base * 2u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        const float e0 = float(xa.x), e1 = float(xa.y);
        const float e2 = float(xa.z), e3 = float(xa.w);
        const float e4 = float(xb.x), e5 = float(xb.y);
        const float e6 = float(xb.z), e7 = float(xb.w);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;

        const uint gb0 = gw4 & 0xFFu;
        const uint gb1 = (gw4 >> 8) & 0xFFu;
        const uint gb2 = (gw4 >> 16) & 0xFFu;
        const uint gb3 = (gw4 >> 24) & 0xFFu;
        float g_dot = 0.0f;
        g_dot = fma(float(gb0 & 0x0Fu), e0, g_dot); g_dot = fma(float(gb0 >> 4), e1, g_dot);
        g_dot = fma(float(gb1 & 0x0Fu), e2, g_dot); g_dot = fma(float(gb1 >> 4), e3, g_dot);
        g_dot = fma(float(gb2 & 0x0Fu), e4, g_dot); g_dot = fma(float(gb2 >> 4), e5, g_dot);
        g_dot = fma(float(gb3 & 0x0Fu), e6, g_dot); g_dot = fma(float(gb3 >> 4), e7, g_dot);

        const uint ub0 = uw4 & 0xFFu;
        const uint ub1 = (uw4 >> 8) & 0xFFu;
        const uint ub2 = (uw4 >> 16) & 0xFFu;
        const uint ub3 = (uw4 >> 24) & 0xFFu;
        float u_dot = 0.0f;
        u_dot = fma(float(ub0 & 0x0Fu), e0, u_dot); u_dot = fma(float(ub0 >> 4), e1, u_dot);
        u_dot = fma(float(ub1 & 0x0Fu), e2, u_dot); u_dot = fma(float(ub1 >> 4), e3, u_dot);
        u_dot = fma(float(ub2 & 0x0Fu), e4, u_dot); u_dot = fma(float(ub2 >> 4), e5, u_dot);
        u_dot = fma(float(ub3 & 0x0Fu), e6, u_dot); u_dot = fma(float(ub3 >> 4), e7, u_dot);

        g_acc = fma(gs, g_dot, g_acc);
        g_acc = fma(gb, sum, g_acc);
        u_acc = fma(us, u_dot, u_acc);
        u_acc = fma(ub, sum, u_acc);
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const float gs = float(gS_row[g]);
        const float gb = float(gB_row[g]);
        const float us = float(uS_row[g]);
        const float ub = float(uB_row[g]);
        const uint8_t gbv = gW_row[g * (kMoEGroupSize / 2) + lane];
        const uint8_t ubv = uW_row[g * (kMoEGroupSize / 2) + lane];
        const float x0 = float(x[g * kMoEGroupSize + lane * 2u]);
        const float x1 = float(x[g * kMoEGroupSize + lane * 2u + 1u]);
        const float sum = x0 + x1;
        float g_dot = fma(float(uint(gbv & 0x0Fu)), x0, 0.0f);
        g_dot = fma(float(uint(gbv >> 4)), x1, g_dot);
        float u_dot = fma(float(uint(ubv & 0x0Fu)), x0, 0.0f);
        u_dot = fma(float(uint(ubv >> 4)), x1, u_dot);
        g_acc = fma(gs, g_dot, g_acc);
        g_acc = fma(gb, sum, g_acc);
        u_acc = fma(us, u_dot, u_acc);
        u_acc = fma(ub, sum, u_acc);
    }
    return float2(simd_sum(g_acc), simd_sum(u_acc));
}

// One gate/up row `f` of routed slot `slot`, activated into `acts`.
static inline void moe_phase1_gate_up_act_row(
    device const RoutedBlobs& routed,
    constant ExpertOffsets& routed_offsets,
    device const half* x,
    device half* acts,
    uint slot,
    uint f,
    uint D,
    uint F,
    uint lane
) {
    device const uint8_t* base = routed.blob[slot];
    const ExpertOffsets re = routed_offsets;
    device const uint8_t* gW = base + re.gate_W_off;
    device const uint8_t* uW = base + re.up_W_off;
    device const bfloat* gS = (device const bfloat*)(base + re.gate_s_off);
    device const bfloat* uS = (device const bfloat*)(base + re.up_s_off);
    device const bfloat* gB = (device const bfloat*)(base + re.gate_b_off);
    device const bfloat* uB = (device const bfloat*)(base + re.up_b_off);

    const float2 gu = moe_int4_gate_up_rows_simd_dev_vec_u16load(
        gW, gS, gB, uW, uS, uB, x, f, D, lane);
    if (lane == 0) acts[slot * F + f] = half(gelu_pytorch_tanh(gu.x) * gu.y);
}

// Eight SIMD groups, one (slot, f) row each, per threadgroup.
kernel void moe_phase1_gate_up_act_u16load(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* x [[buffer(2)]],
    device half* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint FF = moe_fc_f(F);
    const uint rowg = tg_idx * 8u + sg_idx;
    if (rowg >= moe_fc_top_k(top_k) * FF) return;
    moe_phase1_gate_up_act_row(routed, routed_offsets, x, acts, rowg / FF, rowg % FF,
                               moe_fc_d(D), FF, lane);
}

// moe_phase1_gate_up_act_u16load over the `active_count` slots listed in
// `active_slots` only.
kernel void moe_phase1_gate_up_act_subset_u16load(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* x [[buffer(2)]],
    device half* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    device const uint* active_slots [[buffer(7)]],
    constant uint& active_count [[buffer(8)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint FF = moe_fc_f(F);
    const uint rowg = tg_idx * 8u + sg_idx;
    if (rowg >= active_count * FF) return;
    const uint slot = active_slots[rowg / FF];
    if (slot >= moe_fc_top_k(top_k)) return;
    moe_phase1_gate_up_act_row(routed, routed_offsets, x, acts, slot, rowg % FF,
                               moe_fc_d(D), FF, lane);
}

// Shared FFN gate, up and GELU in one dispatch. Each SIMD computes one gate
// row and one up row with the routed experts' row math, which is the decode
// GEMV's, and rounds both to FP16 before the activation as the separate
// GEMV + gelu_mul_fp16 path does, so the result is bitwise that path's.
kernel void shared_gate_up_act_int4(
    device const uint8_t* gW [[buffer(0)]],
    device const bfloat* gS [[buffer(1)]],
    device const bfloat* gB [[buffer(2)]],
    device const uint8_t* uW [[buffer(3)]],
    device const bfloat* uS [[buffer(4)]],
    device const bfloat* uB [[buffer(5)]],
    device const half* x [[buffer(6)]],
    device half* acts [[buffer(7)]],
    constant uint& D [[buffer(8)]],
    constant uint& F [[buffer(9)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row = tg_idx * 8u + sg_idx;
    if (row >= moe_fc_f(F)) return;
    const float2 gu = moe_int4_gate_up_rows_simd_dev_vec_u16load(
        gW, gS, gB, uW, uS, uB, x, row, moe_fc_d(D), lane);
    if (lane == 0) {
        acts[row] = half(gelu_pytorch_tanh(float(half(gu.x))) * float(half(gu.y)));
    }
}

// Decode routing without a host round trip: points each routed slot at its
// expert through the layer's address table, where 0 marks an expert the
// command buffer did not bind. Such a slot reads `fallback` (zeros) and the
// host patches the layer, so this also saves the layer input for the patch.
kernel void moe_gather_routed_slots(
    device const uint* indices [[buffer(0)]],
    device const ulong* table [[buffer(1)]],
    device ulong* slots [[buffer(2)]],
    device const half* hidden [[buffer(3)]],
    device half* saved [[buffer(4)]],
    constant uint& top_k [[buffer(5)]],
    constant uint& num_experts [[buffer(6)]],
    constant uint& D [[buffer(7)]],
    constant ulong& fallback [[buffer(8)]],
    uint tid [[thread_position_in_grid]],
    uint threads [[threads_per_grid]]
) {
    if (tid < top_k) {
        const ulong address = table[min(indices[tid], num_experts - 1)];
        slots[tid] = address != 0 ? address : fallback;
    }
    for (uint i = tid; i < D; i += threads) saved[i] = hidden[i];
}

kernel void moe_phase2_down_reduce_k8(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* acts [[buffer(2)]],
    device const half* routing_w [[buffer(3)]],
    device const half* residual [[buffer(4)]],
    device half* y [[buffer(5)]],
    constant uint& D [[buffer(6)]],
    constant uint& F [[buffer(7)]],
    uint d [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float partial[8];
    const uint DD = moe_fc_d(D);
    const uint FF = moe_fc_f(F);
    if (d >= DD) return;

    device const uint8_t* base = routed.blob[sg_idx];
    const ExpertOffsets re = routed_offsets;
    device const uint8_t* dW = base + re.down_W_off;
    device const bfloat* dS = (device const bfloat*)(base + re.down_s_off);
    device const bfloat* dB = (device const bfloat*)(base + re.down_b_off);
    device const half* act_slot = acts + sg_idx * FF;

    const float value = moe_int4_gemv_row_simd_dev_vec(
        dW, dS, dB, act_slot, d, FF, lane);
    if (lane == 0) partial[sg_idx] = float(routing_w[sg_idx]) * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_idx == 0 && lane == 0) {
        float acc = float(residual[d]);
        acc += partial[0]; acc += partial[1]; acc += partial[2]; acc += partial[3];
        acc += partial[4]; acc += partial[5]; acc += partial[6]; acc += partial[7];
        y[d] = half(acc);
    }
}
