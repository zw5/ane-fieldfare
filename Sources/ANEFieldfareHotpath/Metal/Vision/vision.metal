#include <metal_stdlib>
using namespace metal;

constant constexpr uint kVisionHeadDim = 72;
constant constexpr uint kVisionPaddedHeadDim = 80;

kernel void vision_normalize_patches(
    device const bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint& realRows [[buffer(2)]],
    constant uint& paddedRows [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = paddedRows * 768u;
    if (index >= count) return;
    output[index] = index < realRows * 768u
        ? bfloat(2.0f * (float(input[index]) - 0.5f))
        : bfloat(0.0f);
}

kernel void vision_add_position(
    device bfloat* hidden [[buffer(0)]],
    device const bfloat* table [[buffer(1)]],
    device const int2* positions [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = rows * 1152u;
    if (index >= count) return;
    const uint row = index / 1152u;
    const uint dimension = index % 1152u;
    const int2 position = positions[row];
    const uint xIndex = uint(position.x) * 1152u + dimension;
    const uint yIndex = (10240u + uint(position.y)) * 1152u + dimension;
    hidden[index] = bfloat(float(hidden[index])
                           + float(table[xIndex]) + float(table[yIndex]));
}

inline float vision_simdgroup_sum(float value,
                                  uint lane,
                                  uint simdgroup,
                                  threadgroup float* partials) {
    const float subtotal = simd_sum(value);
    if (lane == 0) partials[simdgroup] = subtotal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        float merged = lane < 8u ? partials[lane] : 0.0f;
        merged = simd_sum(merged);
        if (lane == 0) partials[0] = merged;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return partials[0];
}

kernel void vision_rmsnorm_rows(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device bfloat* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    if (row >= rows) return;
    threadgroup float partials[32];
    threadgroup float inverseRMS[1];
    float values[4];
    float sum = 0.0f;
    const uint base = tid * 4u;
    for (uint index = 0; index < 4u; ++index) {
        const uint dimension = base + index;
        values[index] = dimension < width
            ? float(input[row * width + dimension]) : 0.0f;
        sum += values[index] * values[index];
    }
    sum = simd_sum(sum);
    if (simdgroup == 0) partials[lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) partials[simdgroup] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        sum = simd_sum(partials[lane]);
        if (lane == 0) {
            inverseRMS[0] = metal::precise::rsqrt(sum / float(width) + 1e-6f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = 0; index < 4u; ++index) {
        const uint dimension = base + index;
        if (dimension < width) {
            const bfloat normalized = bfloat(values[index] * inverseRMS[0]);
            output[row * width + dimension] = bfloat(
                float(weight[dimension]) * float(normalized));
        }
    }
}

kernel void vision_postnorm_residual(
    device const bfloat* residual [[buffer(0)]],
    device const bfloat* branch [[buffer(1)]],
    device const bfloat* weight [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    if (row >= rows) return;
    threadgroup float partials[32];
    threadgroup float inverseRMS[1];
    float values[4];
    float sum = 0.0f;
    const uint base = tid * 4u;
    for (uint index = 0; index < 4u; ++index) {
        const uint dimension = base + index;
        values[index] = dimension < width
            ? float(branch[row * width + dimension]) : 0.0f;
        sum += values[index] * values[index];
    }
    sum = simd_sum(sum);
    if (simdgroup == 0) partials[lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) partials[simdgroup] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        sum = simd_sum(partials[lane]);
        if (lane == 0) {
            inverseRMS[0] = metal::precise::rsqrt(sum / float(width) + 1e-6f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint element = 0; element < 4u; ++element) {
        const uint dimension = base + element;
        if (dimension < width) {
            const uint index = row * width + dimension;
            const bfloat scaled = bfloat(values[element] * inverseRMS[0]);
            const bfloat normalized = bfloat(
                float(weight[dimension]) * float(scaled));
            output[index] = bfloat(float(residual[index]) + float(normalized));
        }
    }
}

inline float2 vision_rope_pair(float2 value, int position, uint pairInAxis) {
    const float exponent = (2.0f / 36.0f) * float(pairInAxis);
    const float angle = float(position) / powr(100.0f, exponent);
    const float cosine = cos(angle);
    const float sine = sin(angle);
    return float2(value.x * cosine - value.y * sine,
                  value.y * cosine + value.x * sine);
}

/// Per-head RMSNorm of Q and K (weighted) and V (unweighted), then 2-D RoPE
/// of Q and K, in place on the head-major padded layout the QKV projections
/// write: vector `(head, row)` starts at `(head * rowStride + row) * 80`.
/// Threads walk rows within a head so the strided-by-80 accesses coalesce.
kernel void vision_qkv_norm_rope_padded(
    device bfloat* q [[buffer(0)]],
    device bfloat* k [[buffer(1)]],
    device bfloat* v [[buffer(2)]],
    device const bfloat* qWeight [[buffer(3)]],
    device const bfloat* kWeight [[buffer(4)]],
    device const int2* positions [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& heads [[buffer(7)]],
    constant uint& rowStride [[buffer(8)]],
    uint vectorIndex [[thread_position_in_grid]]) {
    if (vectorIndex >= rows * heads) return;
    const uint head = vectorIndex / rows;
    const uint row = vectorIndex % rows;
    const uint base = (head * rowStride + row) * kVisionPaddedHeadDim;
    float qSquares = 0.0f;
    float kSquares = 0.0f;
    float vSquares = 0.0f;
    for (uint dimension = 0; dimension < kVisionHeadDim; ++dimension) {
        const float qValue = float(q[base + dimension]);
        const float kValue = float(k[base + dimension]);
        const float vValue = float(v[base + dimension]);
        qSquares = fma(qValue, qValue, qSquares);
        kSquares = fma(kValue, kValue, kSquares);
        vSquares = fma(vValue, vValue, vSquares);
    }
    const float qInverse = rsqrt(qSquares / 72.0f + 1e-6f);
    const float kInverse = rsqrt(kSquares / 72.0f + 1e-6f);
    const float vInverse = rsqrt(vSquares / 72.0f + 1e-6f);
    const int2 position = positions[row];
    for (uint axis = 0; axis < 2u; ++axis) {
        const uint axisBase = axis * 36u;
        const int coordinate = axis == 0u ? position.x : position.y;
        for (uint pair = 0; pair < 18u; ++pair) {
            const uint first = base + axisBase + pair;
            const uint second = first + 18u;
            float2 qPair = float2(float(q[first]) * qInverse * float(qWeight[axisBase + pair]),
                                  float(q[second]) * qInverse * float(qWeight[axisBase + pair + 18u]));
            float2 kPair = float2(float(k[first]) * kInverse * float(kWeight[axisBase + pair]),
                                  float(k[second]) * kInverse * float(kWeight[axisBase + pair + 18u]));
            qPair = vision_rope_pair(qPair, coordinate, pair);
            kPair = vision_rope_pair(kPair, coordinate, pair);
            q[first] = bfloat(qPair.x);
            q[second] = bfloat(qPair.y);
            k[first] = bfloat(kPair.x);
            k[second] = bfloat(kPair.y);
        }
    }
    for (uint dimension = 0; dimension < kVisionHeadDim; ++dimension) {
        v[base + dimension] = bfloat(float(v[base + dimension]) * vInverse);
    }
}

kernel void vision_pool_standardize_norm(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* stdBias [[buffer(1)]],
    device const bfloat* stdScale [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    constant uint& patchWidth [[buffer(4)]],
    constant uint& patchHeight [[buffer(5)]],
    uint outputRow [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    const uint outputWidth = patchWidth / 3u;
    const uint outputHeight = patchHeight / 3u;
    if (outputRow >= outputWidth * outputHeight) return;
    const uint outputX = outputRow % outputWidth;
    const uint outputY = outputRow / outputWidth;
    threadgroup float partials[8];
    float squares = 0.0f;
    for (uint dimension = tid; dimension < 1152u; dimension += 256u) {
        float pooled = 0.0f;
        for (uint dy = 0; dy < 3u; ++dy) {
            for (uint dx = 0; dx < 3u; ++dx) {
                const uint inputRow = (outputY * 3u + dy) * patchWidth
                    + outputX * 3u + dx;
                pooled += float(input[inputRow * 1152u + dimension]);
            }
        }
        pooled *= sqrt(1152.0f) / 9.0f;
        const float standardized = (pooled - float(stdBias[dimension]))
            * float(stdScale[dimension]);
        output[outputRow * 1152u + dimension] = bfloat(standardized);
        squares = fma(standardized, standardized, squares);
    }
    const float total = vision_simdgroup_sum(squares, lane, simdgroup, partials);
    const float inverse = rsqrt(total / 1152.0f + 1e-6f);
    for (uint dimension = tid; dimension < 1152u; dimension += 256u) {
        const uint index = outputRow * 1152u + dimension;
        output[index] = bfloat(float(output[index]) * inverse);
    }
}

kernel void vision_bfloat_to_half(
    device const bfloat* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) output[index] = half(input[index]);
}
