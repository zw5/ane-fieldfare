// Adapted from Turbo Fieldfare's compact MoE Metal kernels (Apache-2.0).
// ANE Fieldfare owns this local copy so the hybrid runtime can evolve here.
#include <metal_stdlib>
using namespace metal;

constant constexpr uint kMoEGroupSize = 64;
constant constexpr uint kMaxStreamedExperts = 16;
constant constexpr float kGeluSqrt2OverPi = 0.7978845608028654f;
constant constexpr float kGeluCubicCoeff = 0.044715f;

constant uint FC_ROUTER_NUM_EXPERTS [[function_constant(40)]];
constant uint FC_ROUTER_D [[function_constant(41)]];
constant uint FC_ROUTER_TOP_K [[function_constant(42)]];
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
    uint rows_per_tg,
    uint tg_idx,
    uint sg_idx,
    uint lane
) {
    const uint NE = router_fc_num_experts(num_experts);
    const uint DD = router_fc_d(D);
    const uint e = tg_idx * rows_per_tg + sg_idx;
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
                            out_logits, num_experts, D, 4, tg_idx, sg_idx, lane);
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

// Generic affine-INT4 matrix/vector rail used by resident attention, shared
// FFN, and tied-head tensors. One SIMD group owns one output row; weights stay
// packed and BF16 scale/bias parameters are consumed directly.
kernel void affine_int4_gemv_f16(
    device const uint8_t* W [[buffer(0)]],
    device const bfloat* S [[buffer(1)]],
    device const bfloat* B [[buffer(2)]],
    device const half* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& columns [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (row >= rows) return;
    float value = moe_int4_gemv_row_simd_dev_vec(W, S, B, x, row, columns, lane);
    if (lane == 0) y[row] = half(value);
}

kernel void swiglu_f16(
    device half* gate [[buffer(0)]],
    device const half* up [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) gate[index] = half(gelu_pytorch_tanh(float(gate[index])) * float(up[index]));
}

static inline float affine_int4_row_f32(
    device const uint8_t* W,
    device const bfloat* S,
    device const bfloat* B,
    device const float* x,
    uint row,
    uint N,
    uint lane) {
    const uint groups = N / kMoEGroupSize;
    const uint rowBytes = N / 2;
    device const uint8_t* wr = W + row * rowBytes;
    device const bfloat* sr = S + row * groups;
    device const bfloat* br = B + row * groups;
    float acc = 0.0f;
    const uint fullBlocks = groups / 4u;
    for (uint block = 0; block < fullBlocks; ++block) {
        const uint byteBase = block * 128u + lane * 4u;
        // Live compact tensors guarantee two-byte, not four-byte, alignment.
        // Two ushort loads preserve that contract while exposing four groups'
        // independent arithmetic to the compiler.
        device const ushort* packed16 = (device const ushort*)(wr + byteBase);
        const uint packed = uint(packed16[0]) | (uint(packed16[1]) << 16);
        const uint group = block * 4u + (lane >> 3);
        const float scale = float(sr[group]);
        const float bias = float(br[group]);
        const uint element = byteBase * 2u;
        const float4 xa = *((device const float4*)(x + element));
        const float4 xb = *((device const float4*)(x + element + 4u));
        const uint b0 = packed & 0xffu;
        const uint b1 = (packed >> 8) & 0xffu;
        const uint b2 = (packed >> 16) & 0xffu;
        const uint b3 = (packed >> 24) & 0xffu;
        float dot = 0.0f;
        dot = fma(float(b0 & 0x0fu), xa.x, dot);
        dot = fma(float(b0 >> 4), xa.y, dot);
        dot = fma(float(b1 & 0x0fu), xa.z, dot);
        dot = fma(float(b1 >> 4), xa.w, dot);
        dot = fma(float(b2 & 0x0fu), xb.x, dot);
        dot = fma(float(b2 >> 4), xb.y, dot);
        dot = fma(float(b3 & 0x0fu), xb.z, dot);
        dot = fma(float(b3 >> 4), xb.w, dot);
        acc = fma(scale, dot, acc);
        acc = fma(bias, xa.x + xa.y + xa.z + xa.w + xb.x + xb.y + xb.z + xb.w, acc);
    }
    for (uint group = fullBlocks * 4u; group < groups; ++group) {
        const uint8_t packed = wr[group * 32u + lane];
        const float x0 = x[group * 64u + lane * 2u];
        const float x1 = x[group * 64u + lane * 2u + 1u];
        const float scale = float(sr[group]);
        const float bias = float(br[group]);
        acc = fma(scale, fma(float(packed & 0x0fu), x0,
                             float(packed >> 4) * x1), acc);
        acc = fma(bias, x0 + x1, acc);
    }
    return simd_sum(acc);
}

static inline float affine_int4_row_f32_scalar_groups(
    device const uint8_t* W,
    device const bfloat* S,
    device const bfloat* B,
    device const float* x,
    uint row,
    uint N,
    uint lane) {
    const uint groups = N / kMoEGroupSize;
    const uint rowBytes = N / 2;
    device const uint8_t* wr = W + row * rowBytes;
    device const bfloat* sr = S + row * groups;
    device const bfloat* br = B + row * groups;
    float acc = 0.0f;
    for (uint group = 0; group < groups; ++group) {
        const uint8_t packed = wr[group * 32u + lane];
        const float x0 = x[group * 64u + lane * 2u];
        const float x1 = x[group * 64u + lane * 2u + 1u];
        acc = fma(float(sr[group]),
                  fma(float(packed & 0x0fu), x0, float(packed >> 4) * x1), acc);
        acc = fma(float(br[group]), x0 + x1, acc);
    }
    return simd_sum(acc);
}

kernel void affine_int4_gemv_f32(
    device const uint8_t* W [[buffer(0)]],
    device const bfloat* S [[buffer(1)]],
    device const bfloat* B [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& columns [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (row >= rows) return;
    float value = affine_int4_row_f32(W, S, B, x, row, columns, lane);
    if (lane == 0) y[row] = value;
}

kernel void affine_int4_gemv_f32_scalar_groups(
    device const uint8_t* W [[buffer(0)]],
    device const bfloat* S [[buffer(1)]],
    device const bfloat* B [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& columns [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (row >= rows) return;
    float value = affine_int4_row_f32_scalar_groups(W, S, B, x, row, columns, lane);
    if (lane == 0) y[row] = value;
}

// Adapted from Turbo Fieldfare's Apache-2.0 row-reduced greedy head. Eight
// SIMD groups evaluate eight independent vocabulary rows, then emit one
// value/index summary per threadgroup. A second dispatch reduces those
// summaries, so greedy decode never materializes a vocabulary-sized logits
// carrier on the host.
constant constexpr uint kGreedyHeadRowsPerThreadgroup = 8;

kernel void greedy_head_int4_rows_f32(
    device const uint8_t* W [[buffer(0)]],
    device const bfloat* S [[buffer(1)]],
    device const bfloat* B [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float2* summaries [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& columns [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdGroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partialValues[kGreedyHeadRowsPerThreadgroup];
    threadgroup uint partialIndices[kGreedyHeadRowsPerThreadgroup];
    const uint row = group * kGreedyHeadRowsPerThreadgroup + simdGroup;
    float bestValue = -INFINITY;
    uint bestIndex = 0xffffffffu;
    if (row < rows) {
        float value = affine_int4_row_f32_scalar_groups(W, S, B, x, row, columns, lane);
        if (lane == 0 && isfinite(value)) {
            bestValue = value;
            bestIndex = row;
        }
    }
    if (lane == 0) {
        partialValues[simdGroup] = bestValue;
        partialIndices[simdGroup] = bestIndex;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdGroup == 0) {
        const bool active = lane < kGreedyHeadRowsPerThreadgroup;
        const float value = active ? partialValues[lane] : -INFINITY;
        const uint index = active ? partialIndices[lane] : 0xffffffffu;
        const float maximum = simd_max(value);
        uint winner = value == maximum ? index : 0xffffffffu;
        winner = simd_min(winner);
        if (lane == 0) summaries[group] = float2(maximum, as_type<float>(winner));
    }
}

kernel void greedy_head_reduce(
    device const float2* summaries [[buffer(0)]],
    device uint* outputToken [[buffer(1)]],
    device float* outputValue [[buffer(2)]],
    constant uint& groupCount [[buffer(3)]],
    uint index [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdGroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partialValues[8];
    threadgroup uint partialIndices[8];
    float bestValue = -INFINITY;
    uint bestIndex = 0xffffffffu;
    for (uint group = index; group < groupCount; group += 256) {
        float2 summary = summaries[group];
        uint candidate = as_type<uint>(summary.y);
        if (summary.x > bestValue || (summary.x == bestValue && candidate < bestIndex)) {
            bestValue = summary.x;
            bestIndex = candidate;
        }
    }
    const float simdValue = simd_max(bestValue);
    uint simdIndex = bestValue == simdValue ? bestIndex : 0xffffffffu;
    simdIndex = simd_min(simdIndex);
    if (lane == 0) {
        partialValues[simdGroup] = simdValue;
        partialIndices[simdGroup] = simdIndex;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdGroup == 0) {
        const bool active = lane < 8;
        const float value = active ? partialValues[lane] : -INFINITY;
        const uint candidate = active ? partialIndices[lane] : 0xffffffffu;
        const float maximum = simd_max(value);
        uint winner = value == maximum ? candidate : 0xffffffffu;
        winner = simd_min(winner);
        if (lane == 0) {
            outputToken[0] = winner == 0xffffffffu ? 0u : winner;
            outputValue[0] = maximum;
        }
    }
}

kernel void greedy_head_int4_rows_positions_f32(
    device const uint8_t* W [[buffer(0)]],
    device const bfloat* S [[buffer(1)]],
    device const bfloat* B [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float2* summaries [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& columns [[buffer(6)]],
    constant uint& positions [[buffer(7)]],
    uint2 grid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdGroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partialValues[kGreedyHeadRowsPerThreadgroup];
    threadgroup uint partialIndices[kGreedyHeadRowsPerThreadgroup];
    if (grid.y >= positions) return;
    const uint row = grid.x * kGreedyHeadRowsPerThreadgroup + simdGroup;
    float bestValue = -INFINITY;
    uint bestIndex = 0xffffffffu;
    if (row < rows) {
        float value = affine_int4_row_f32_scalar_groups(
            W, S, B, x + uint64_t(grid.y) * columns, row, columns, lane);
        if (lane == 0 && isfinite(value)) { bestValue = value; bestIndex = row; }
    }
    if (lane == 0) {
        partialValues[simdGroup] = bestValue;
        partialIndices[simdGroup] = bestIndex;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdGroup == 0) {
        const bool active = lane < kGreedyHeadRowsPerThreadgroup;
        const float value = active ? partialValues[lane] : -INFINITY;
        const uint candidate = active ? partialIndices[lane] : 0xffffffffu;
        const float maximum = simd_max(value);
        uint winner = value == maximum ? candidate : 0xffffffffu;
        winner = simd_min(winner);
        if (lane == 0) {
            const uint groupCount = (rows + 7u) / 8u;
            summaries[uint64_t(grid.y) * groupCount + grid.x] =
                float2(maximum, as_type<float>(winner));
        }
    }
}

kernel void greedy_head_reduce_positions(
    device const float2* summaries [[buffer(0)]],
    device uint* outputTokens [[buffer(1)]],
    device float* outputValues [[buffer(2)]],
    constant uint& groupCount [[buffer(3)]],
    uint position [[threadgroup_position_in_grid]],
    uint index [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdGroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partialValues[8];
    threadgroup uint partialIndices[8];
    float bestValue = -INFINITY;
    uint bestIndex = 0xffffffffu;
    device const float2* row = summaries + uint64_t(position) * groupCount;
    for (uint group = index; group < groupCount; group += 256) {
        float2 summary = row[group];
        uint candidate = as_type<uint>(summary.y);
        if (summary.x > bestValue || (summary.x == bestValue && candidate < bestIndex)) {
            bestValue = summary.x; bestIndex = candidate;
        }
    }
    const float simdValue = simd_max(bestValue);
    uint simdIndex = bestValue == simdValue ? bestIndex : 0xffffffffu;
    simdIndex = simd_min(simdIndex);
    if (lane == 0) { partialValues[simdGroup] = simdValue; partialIndices[simdGroup] = simdIndex; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdGroup == 0) {
        const bool active = lane < 8;
        const float value = active ? partialValues[lane] : -INFINITY;
        const uint candidate = active ? partialIndices[lane] : 0xffffffffu;
        const float maximum = simd_max(value);
        uint winner = value == maximum ? candidate : 0xffffffffu;
        winner = simd_min(winner);
        if (lane == 0) { outputTokens[position] = winner; outputValues[position] = maximum; }
    }
}

// Prefill matrix/matrix rail. One SIMD group owns one output row for one
// position; the compact weight row is shared by all positions in the command.
// This preserves the GEMV arithmetic exactly while removing one command-buffer
// round trip per prompt position.
kernel void affine_int4_gemm_f32(
    device const uint8_t* W [[buffer(0)]],
    device const bfloat* S [[buffer(1)]],
    device const bfloat* B [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& columns [[buffer(6)]],
    constant uint& positions [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint row = group.x;
    const uint position = group.y;
    if (row >= rows || position >= positions) return;
    float value = affine_int4_row_f32(
        W, S, B, x + uint64_t(position) * columns, row, columns, lane);
    if (lane == 0) y[uint64_t(position) * rows + row] = value;
}

kernel void swiglu_f32(
    device float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) gate[index] = gelu_pytorch_tanh(gate[index]) * up[index];
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

static inline void moe_phase1_gate_up_act_u16load_body(
    device const RoutedBlobs& routed,
    constant ExpertOffsets& routed_offsets,
    device const half* x,
    device half* acts,
    uint D,
    uint F,
    uint top_k,
    uint rows_per_tg,
    uint tg_idx,
    uint sg_idx,
    uint lane
) {
    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= top_k * F) return;
    const uint slot = rowg / F;
    const uint f = rowg % F;

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

static inline void moe_phase1_gate_up_act_subset_u16load_body(
    device const RoutedBlobs& routed,
    constant ExpertOffsets& routed_offsets,
    device const half* x,
    device half* acts,
    device const uint* active_slots,
    uint active_count,
    uint D,
    uint F,
    uint top_k,
    uint rows_per_tg,
    uint tg_idx,
    uint sg_idx,
    uint lane
) {
    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= active_count * F) return;
    const uint active_idx = rowg / F;
    const uint slot = active_slots[active_idx];
    if (slot >= top_k) return;
    const uint f = rowg % F;

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
    constexpr uint rows_per_tg = 8;
    moe_phase1_gate_up_act_u16load_body(
        routed, routed_offsets, x, acts, moe_fc_d(D), moe_fc_f(F),
        moe_fc_top_k(top_k), rows_per_tg, tg_idx, sg_idx, lane);
}

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
    constexpr uint rows_per_tg = 8;
    moe_phase1_gate_up_act_subset_u16load_body(
        routed, routed_offsets, x, acts, active_slots, active_count,
        moe_fc_d(D), moe_fc_f(F), moe_fc_top_k(top_k), rows_per_tg,
        tg_idx, sg_idx, lane);
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

// Grouped prefill construction adapted from Turbo Fieldfare's Apache-2.0
// token/expert grouping. Unlike the decode kernel above, each bound expert is
// reused for every prompt position that selected it before the cache slots are
// recycled for the next expert tile.
struct GroupedPrefillPair {
    uint token;
    uint slot;
    uint rank;
    uint reserved;
};

kernel void moe_grouped_prefill_phase1(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* inputs [[buffer(2)]],
    device const GroupedPrefillPair* pairs [[buffer(3)]],
    device half* acts [[buffer(4)]],
    constant uint& pair_count [[buffer(5)]],
    constant uint& D [[buffer(6)]],
    constant uint& F [[buffer(7)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint rows_per_tg = 8;
    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= pair_count * F) return;
    const uint pair_index = rowg / F;
    const uint f = rowg % F;
    const GroupedPrefillPair pair = pairs[pair_index];
    if (pair.slot >= kMaxStreamedExperts) return;
    device const uint8_t* base = routed.blob[pair.slot];
    device const half* x = inputs + pair.token * D;
    const ExpertOffsets re = routed_offsets;
    const float2 gu = moe_int4_gate_up_rows_simd_dev_vec_u16load(
        base + re.gate_W_off,
        (device const bfloat*)(base + re.gate_s_off),
        (device const bfloat*)(base + re.gate_b_off),
        base + re.up_W_off,
        (device const bfloat*)(base + re.up_s_off),
        (device const bfloat*)(base + re.up_b_off),
        x, f, D, lane);
    if (lane == 0)
        acts[pair_index * F + f] = half(gelu_pytorch_tanh(gu.x) * gu.y);
}

kernel void moe_grouped_prefill_phase2(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const GroupedPrefillPair* pairs [[buffer(2)]],
    device const half* acts [[buffer(3)]],
    device const half* routing_weights [[buffer(4)]],
    device half* partials [[buffer(5)]],
    constant uint& pair_count [[buffer(6)]],
    constant uint& D [[buffer(7)]],
    constant uint& F [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint d = tg.x;
    const uint pair_index = tg.y;
    if (d >= D || pair_index >= pair_count) return;
    const GroupedPrefillPair pair = pairs[pair_index];
    if (pair.slot >= kMaxStreamedExperts || pair.rank >= 8) return;
    device const uint8_t* base = routed.blob[pair.slot];
    const ExpertOffsets re = routed_offsets;
    const float value = moe_int4_gemv_row_simd_dev_vec(
        base + re.down_W_off,
        (device const bfloat*)(base + re.down_s_off),
        (device const bfloat*)(base + re.down_b_off),
        acts + pair_index * F, d, F, lane);
    if (lane == 0) {
        const uint route = pair.token * 8u + pair.rank;
        partials[route * D + d] = half(float(routing_weights[route]) * value);
    }
}

kernel void moe_grouped_prefill_reduce(
    device const half* partials [[buffer(0)]],
    device half* outputs [[buffer(1)]],
    constant uint& position_count [[buffer(2)]],
    constant uint& D [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x;
    const uint token = gid.y;
    if (d >= D || token >= position_count) return;
    float value = 0.0f;
    for (uint rank = 0; rank < 8; ++rank)
        value += float(partials[(token * 8u + rank) * D + d]);
    outputs[token * D + d] = half(value);
}

constant constexpr uint kPackedExpertStride = 3358720;

kernel void moe_phase1_layer_gate_up_act(
    device const uint8_t* layer [[buffer(0)]],
    device const uint* expert_ids [[buffer(1)]],
    constant ExpertOffsets& re [[buffer(2)]],
    device const half* x [[buffer(3)]],
    device half* acts [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    constant uint& F [[buffer(6)]],
    constant uint& top_k [[buffer(7)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint rows_per_tg = 8;
    const uint DD = moe_fc_d(D), FF = moe_fc_f(F), K = moe_fc_top_k(top_k);
    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= K * FF) return;
    const uint slot = rowg / FF, f = rowg % FF;
    device const uint8_t* base = layer + expert_ids[slot] * kPackedExpertStride;
    const float2 gu = moe_int4_gate_up_rows_simd_dev_vec_u16load(
        base + re.gate_W_off, (device const bfloat*)(base + re.gate_s_off),
        (device const bfloat*)(base + re.gate_b_off), base + re.up_W_off,
        (device const bfloat*)(base + re.up_s_off),
        (device const bfloat*)(base + re.up_b_off), x, f, DD, lane);
    if (lane == 0) acts[slot * FF + f] = half(gelu_pytorch_tanh(gu.x) * gu.y);
}

kernel void moe_phase2_layer_down_reduce(
    device const uint8_t* layer [[buffer(0)]],
    device const uint* expert_ids [[buffer(1)]],
    constant ExpertOffsets& re [[buffer(2)]],
    device const half* acts [[buffer(3)]],
    device const half* routing_w [[buffer(4)]],
    device const half* residual [[buffer(5)]],
    device half* y [[buffer(6)]],
    constant uint& D [[buffer(7)]],
    constant uint& F [[buffer(8)]],
    uint d [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[8];
    const uint DD = moe_fc_d(D), FF = moe_fc_f(F);
    if (d >= DD) return;
    device const uint8_t* base = layer + expert_ids[sg_idx] * kPackedExpertStride;
    const float value = moe_int4_gemv_row_simd_dev_vec(
        base + re.down_W_off, (device const bfloat*)(base + re.down_s_off),
        (device const bfloat*)(base + re.down_b_off), acts + sg_idx * FF, d, FF, lane);
    if (lane == 0) partial[sg_idx] = float(routing_w[sg_idx]) * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_idx == 0 && lane == 0) {
        float acc = float(residual[d]);
        acc += partial[0]; acc += partial[1]; acc += partial[2]; acc += partial[3];
        acc += partial[4]; acc += partial[5]; acc += partial[6]; acc += partial[7];
        y[d] = half(acc);
    }
}
