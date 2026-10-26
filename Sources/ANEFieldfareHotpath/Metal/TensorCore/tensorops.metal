#include <metal_stdlib>
using namespace metal;

#if defined(__HAVE_TENSOR__)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

constant constexpr uint kGroupSize = 64;

constant constexpr int kMPPAffineTileM = 64;
constant constexpr int kMPPAffineTileN = 32;
constant constexpr int kMPPAffineTileK = 64;

kernel void mpp_prefill_affine_threadgroup_f16(
    device const uint8_t* packedWeights [[buffer(0)]],
    device const bfloat* scales         [[buffer(1)]],
    device const bfloat* biases         [[buffer(2)]],
    device half* activations            [[buffer(3)]],
    device half* output                 [[buffer(4)]],
    constant uint& M                    [[buffer(5)]],
    constant uint& N                    [[buffer(6)]],
    constant uint& K                    [[buffer(7)]],
    uint3 tgid                          [[threadgroup_position_in_grid]],
    uint3 lid3                          [[thread_position_in_threadgroup]],
    uint3 threads3                      [[threads_per_threadgroup]]) {
    constexpr auto descriptor = matmul2d_descriptor(
        kMPPAffineTileM, kMPPAffineTileN, kMPPAffineTileK,
        false, true, false);
    matmul2d<descriptor, execution_simdgroups<4>> operation;

    using device_half_tensor = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using threadgroup_half_tensor = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;

    threadgroup half weightTile[kMPPAffineTileN * kMPPAffineTileK];
    threadgroup_half_tensor tileB(
        weightTile,
        dextents<int32_t, 2>(kMPPAffineTileK, kMPPAffineTileN),
        array<int32_t, 2>({1, kMPPAffineTileK}));
    device_half_tensor firstA(
        activations,
        dextents<int32_t, 2>(kMPPAffineTileK, M),
        array<int32_t, 2>({1, int32_t(K)}));
    auto firstTileA = firstA.slice(
        0,
        int32_t(tgid.y) * kMPPAffineTileM);
    auto accumulator = operation.get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB), float>();
    auto groupProduct = operation.get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB), float>();
    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        accumulator[element] = 0.0f;
    }

    const uint rowBytes = K / 2u;
    const uint groupsPerRow = K / kGroupSize;
    const uint lid = lid3.x;
    const uint threads = threads3.x;
    for (uint group = 0; group < groupsPerRow; ++group) {
        for (int element = 0; element < groupProduct.get_capacity(); ++element) {
            groupProduct[element] = 0.0f;
        }
        for (uint linear = lid;
             linear < uint(kMPPAffineTileN * kMPPAffineTileK);
             linear += threads) {
            const uint localN = linear / uint(kMPPAffineTileK);
            const uint localK = linear % uint(kMPPAffineTileK);
            const uint globalN = tgid.x * uint(kMPPAffineTileN) + localN;
            if (globalN < N) {
                const uint globalK = group * uint(kMPPAffineTileK) + localK;
                const uint8_t packed = packedWeights[globalN * rowBytes + (globalK >> 1)];
                const uint q = (globalK & 1u) == 0u
                    ? uint(packed & 0x0fu)
                    : uint(packed >> 4);
                const float scale = float(scales[globalN * groupsPerRow + group]);
                const float bias = float(biases[globalN * groupsPerRow + group]);
                weightTile[linear] = half(fma(float(q), scale, bias));
            } else {
                weightTile[linear] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device_half_tensor groupA(
            activations + group * uint(kMPPAffineTileK),
            dextents<int32_t, 2>(kMPPAffineTileK, M),
            array<int32_t, 2>({1, int32_t(K)}));
        auto tileA = groupA.slice(
            0,
            int32_t(tgid.y) * kMPPAffineTileM);
        operation.run(tileA, tileB, groupProduct);
        for (int element = 0; element < accumulator.get_capacity(); ++element) {
            accumulator[element] += groupProduct[element];
        }
        // TensorOps does not make threadgroup-memory lifetime visible across
        // SIMD groups; keep the next group from overwriting B while another
        // SIMD group can still be consuming the current tile.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        if (!accumulator.is_valid_element(element)) continue;
        const auto position = accumulator.get_multidimensional_index(element);
        const uint globalN = tgid.x * uint(kMPPAffineTileN) + uint(position[0]);
        const uint globalM = tgid.y * uint(kMPPAffineTileM) + uint(position[1]);
        if (globalM < M && globalN < N) {
            output[globalM * N + globalN] = half(accumulator[element]);
        }
    }
}

// The same product with 128x64 output tiles, so each dequantized weight
// serves 128 tokens. Per 64-wide step along K each thread dequantizes 32
// weights of one row from four word loads and one scale and bias, and the
// tensor op accumulates in place; 8 KB of threadgroup memory lets several
// threadgroups share a core. Needs N a multiple of 64, K of 64, and
// 4-byte-aligned weight rows.
constant constexpr int kMPPAffineWideTileM = 128;
constant constexpr int kMPPAffineWideTileN = 64;
constant constexpr int kMPPAffineWideTileK = 64;

kernel void mpp_prefill_affine_wide_f16(
    device const uint8_t* packedWeights [[buffer(0)]],
    device const bfloat* scales         [[buffer(1)]],
    device const bfloat* biases         [[buffer(2)]],
    device half* activations            [[buffer(3)]],
    device half* output                 [[buffer(4)]],
    constant uint& M                    [[buffer(5)]],
    constant uint& N                    [[buffer(6)]],
    constant uint& K                    [[buffer(7)]],
    uint3 tgid                          [[threadgroup_position_in_grid]],
    uint3 lid3                          [[thread_position_in_threadgroup]]) {
    constexpr auto descriptor = matmul2d_descriptor(
        kMPPAffineWideTileM, kMPPAffineWideTileN, kMPPAffineWideTileK,
        false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<4>> operation;

    using device_half_tensor = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using threadgroup_half_tensor = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;

    threadgroup half weightTile[kMPPAffineWideTileN * kMPPAffineWideTileK];
    threadgroup_half_tensor tileB(
        weightTile,
        dextents<int32_t, 2>(kMPPAffineWideTileK, kMPPAffineWideTileN),
        array<int32_t, 2>({1, kMPPAffineWideTileK}));
    device_half_tensor matrixA(
        activations,
        dextents<int32_t, 2>(int32_t(K), int32_t(M)),
        array<int32_t, 2>({1, int32_t(K)}));
    auto firstTileA = matrixA.slice(0, int32_t(tgid.y) * kMPPAffineWideTileM);
    auto accumulator = operation.get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB), float>();
    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        accumulator[element] = 0.0f;
    }

    // Thread t dequantizes row t % 64, 32-weight chunk t / 64.
    const uint lid = lid3.x;
    const uint localN = lid % uint(kMPPAffineWideTileN);
    const uint chunk0 = lid / uint(kMPPAffineWideTileN);
    const uint globalN = tgid.x * uint(kMPPAffineWideTileN) + localN;
    const uint groupsPerRow = K / kGroupSize;
    device const uint* row = reinterpret_cast<device const uint*>(packedWeights + globalN * (K / 2u));
    device const bfloat* rowScales = scales + globalN * groupsPerRow;
    device const bfloat* rowBiases = biases + globalN * groupsPerRow;
    threadgroup half* dst = weightTile + localN * uint(kMPPAffineWideTileK) + chunk0 * 32u;
    for (uint tileK = 0; tileK < K; tileK += uint(kMPPAffineWideTileK)) {
        const uint group = tileK / kGroupSize;
        const float scale = float(rowScales[group]);
        const float bias = float(rowBiases[group]);
        device const uint* words = row + (tileK + chunk0 * 32u) / 8u;
        for (uint w = 0; w < 4u; ++w) {
            const uint word = words[w];
            for (uint j = 0; j < 8u; ++j) {
                dst[8u * w + j] = half(fma(float((word >> (4u * j)) & 0xFu), scale, bias));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto tileA = matrixA.slice(int32_t(tileK), int32_t(tgid.y) * kMPPAffineWideTileM);
        operation.run(tileA, tileB, accumulator);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        if (!accumulator.is_valid_element(element)) continue;
        const auto position = accumulator.get_multidimensional_index(element);
        const uint outN = tgid.x * uint(kMPPAffineWideTileN) + uint(position[0]);
        const uint outM = tgid.y * uint(kMPPAffineWideTileM) + uint(position[1]);
        if (outM < M) {
            output[outM * N + outN] = half(accumulator[element]);
        }
    }
}

constant constexpr int kMPPAffineApple10TileM = 64;
constant constexpr int kMPPAffineApple10TileN = 64;
constant constexpr int kMPPAffineApple10TileK = 128;

kernel void mpp_prefill_affine_threadgroup_bf16_apple10(
    device const uint8_t* packedWeights [[buffer(0)]],
    device const bfloat* scales         [[buffer(1)]],
    device const bfloat* biases         [[buffer(2)]],
    device bfloat* activations          [[buffer(3)]],
    device half* output                 [[buffer(4)]],
    constant uint& M                    [[buffer(5)]],
    constant uint& N                    [[buffer(6)]],
    constant uint& K                    [[buffer(7)]],
    uint3 tgid                          [[threadgroup_position_in_grid]],
    uint3 lid3                          [[thread_position_in_threadgroup]],
    uint3 threads3                      [[threads_per_threadgroup]]) {
    constexpr auto descriptor = matmul2d_descriptor(
        kMPPAffineApple10TileM, kMPPAffineApple10TileN,
        kMPPAffineApple10TileK, false, true, false);
    matmul2d<descriptor, execution_simdgroups<4>> operation;

    using device_bfloat_tensor =
        tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>;
    using threadgroup_bfloat_tensor =
        tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline>;

    threadgroup bfloat weightTile[
        kMPPAffineApple10TileN * kMPPAffineApple10TileK];
    threadgroup_bfloat_tensor tileB(
        weightTile,
        dextents<int32_t, 2>(kMPPAffineApple10TileK, kMPPAffineApple10TileN),
        array<int32_t, 2>({1, kMPPAffineApple10TileK}));
    device_bfloat_tensor firstA(
        activations,
        dextents<int32_t, 2>(kMPPAffineApple10TileK, M),
        array<int32_t, 2>({1, int32_t(K)}));
    auto firstTileA = firstA.slice(
        0, int32_t(tgid.y) * kMPPAffineApple10TileM);
    auto accumulator = operation.get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB), float>();
    auto groupProduct = operation.get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB), float>();
    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        accumulator[element] = 0.0f;
    }

    const uint rowBytes = K / 2u;
    const uint groupsPerRow = K / kGroupSize;
    const uint lid = lid3.x;
    const uint threads = threads3.x;
    for (uint tileKStart = 0; tileKStart < K;
         tileKStart += uint(kMPPAffineApple10TileK)) {
        for (int element = 0; element < groupProduct.get_capacity(); ++element) {
            groupProduct[element] = 0.0f;
        }
        for (uint linear = lid;
             linear < uint(kMPPAffineApple10TileN * kMPPAffineApple10TileK);
             linear += threads) {
            const uint localN = linear / uint(kMPPAffineApple10TileK);
            const uint localK = linear % uint(kMPPAffineApple10TileK);
            const uint globalN = tgid.x * uint(kMPPAffineApple10TileN) + localN;
            if (globalN < N) {
                const uint globalK = tileKStart + localK;
                const uint8_t packed = packedWeights[
                    globalN * rowBytes + (globalK >> 1)];
                const uint q = (globalK & 1u) == 0u
                    ? uint(packed & 0x0fu)
                    : uint(packed >> 4);
                const uint affineGroup = globalK / uint(kGroupSize);
                const float scale = float(scales[
                    globalN * groupsPerRow + affineGroup]);
                const float bias = float(biases[
                    globalN * groupsPerRow + affineGroup]);
                weightTile[linear] = bfloat(fma(float(q), scale, bias));
            } else {
                weightTile[linear] = bfloat(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device_bfloat_tensor groupA(
            activations + tileKStart,
            dextents<int32_t, 2>(kMPPAffineApple10TileK, M),
            array<int32_t, 2>({1, int32_t(K)}));
        auto tileA = groupA.slice(
            0, int32_t(tgid.y) * kMPPAffineApple10TileM);
        operation.run(tileA, tileB, groupProduct);
        for (int element = 0; element < accumulator.get_capacity(); ++element) {
            accumulator[element] += groupProduct[element];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        if (!accumulator.is_valid_element(element)) continue;
        const auto position = accumulator.get_multidimensional_index(element);
        const uint globalN = tgid.x * uint(kMPPAffineApple10TileN)
            + uint(position[0]);
        const uint globalM = tgid.y * uint(kMPPAffineApple10TileM)
            + uint(position[1]);
        if (globalM < M && globalN < N) {
            output[globalM * N + globalN] = half(accumulator[element]);
        }
    }
}

constant constexpr int kVisionAttentionQueries = 16;
constant constexpr int kVisionAttentionKeys = 64;
constant constexpr int kVisionAttentionHeadDim = 80;
// The QK contraction needs a K that is a multiple of 16, so it keeps 80. The PV
// matmul carries the head dimension as N, which only needs a multiple of 8, so
// it contracts over the 72 real lanes; lanes 72 to 79 hold zero V.
constant constexpr int kVisionAttentionRealHeadDim = 72;

/// Mirrors `VisionAttentionLayout` in VisionAttention.swift. `inputRowStride`
/// (nonzero) is the per-head row count of the padded Q/K/V buffers, which may
/// exceed `sequenceLength` when the projections write every padded GEMM row.
/// The output is stored row-major `(row * numHeads + head) * headDim + lane`
/// for lanes below `unpaddedOutputHeadDim`.
struct VisionAttentionLayout {
    uint inputRowStride;
    uint unpaddedOutputHeadDim;
};

/// Threadgroup memory is passed in because MSL forbids declaring it in a
/// non-kernel function.
inline void mpp_vision_attention_bf16_impl(
    device bfloat* Q, device bfloat* K, device bfloat* V, device bfloat* O,
    constant uint& sequenceLength,
    constant uint& numHeads,
    constant VisionAttentionLayout& layout,
    uint3 tg, uint lid, ushort simdLane, ushort simdIndex,
    threadgroup float* scoreTile,
    threadgroup float* weightTile,
    threadgroup float* rowMax,
    threadgroup float* rowSum,
    threadgroup float* oldScale,
    threadgroup float* nextMaxTile) {
    constexpr auto qkDescriptor = matmul2d_descriptor(
        kVisionAttentionQueries,
        kVisionAttentionKeys,
        kVisionAttentionHeadDim,
        false, true, false);
    constexpr auto pvDescriptor = matmul2d_descriptor(
        kVisionAttentionQueries,
        kVisionAttentionRealHeadDim,
        kVisionAttentionKeys,
        false, false, false);
    matmul2d<qkDescriptor, execution_simdgroups<4>> qkOperation;
    matmul2d<pvDescriptor, execution_simdgroups<4>> pvOperation;

    using device_bfloat_tensor =
        tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>;
    using threadgroup_float_tensor =
        tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>;

    const uint queryStart = tg.x * uint(kVisionAttentionQueries);
    const uint head = tg.y;
    if (queryStart >= sequenceLength || head >= numHeads) return;
    const uint rowStride = uint(kVisionAttentionHeadDim);
    const uint validQueries = min(
        uint(kVisionAttentionQueries), sequenceLength - queryStart);

    const uint headBase =
        head * layout.inputRowStride * uint(kVisionAttentionHeadDim);
    device_bfloat_tensor queryTensor(
        Q + headBase,
        dextents<int32_t, 2>(kVisionAttentionHeadDim, int32_t(sequenceLength)),
        array<int32_t, 2>({1, int32_t(rowStride)}));
    device_bfloat_tensor keyTensor(
        K + headBase,
        dextents<int32_t, 2>(kVisionAttentionHeadDim, int32_t(sequenceLength)),
        array<int32_t, 2>({1, int32_t(rowStride)}));
    device_bfloat_tensor valueTensor(
        V + headBase,
        dextents<int32_t, 2>(kVisionAttentionRealHeadDim, int32_t(sequenceLength)),
        array<int32_t, 2>({1, int32_t(rowStride)}));
    threadgroup_float_tensor weights(
        weightTile,
        dextents<int32_t, 2>(
            kVisionAttentionKeys, kVisionAttentionQueries),
        array<int32_t, 2>({1, kVisionAttentionKeys}));

    auto querySlice = queryTensor.slice(0, int32_t(queryStart));
    auto firstValueSlice = valueTensor.slice(0, 0);
    auto outputAccumulator =
        pvOperation.template get_destination_cooperative_tensor<
            decltype(weights), decltype(firstValueSlice), float>();
    for (int element = 0;
         element < outputAccumulator.get_capacity();
         ++element) {
        if (outputAccumulator.is_valid_element(element)) {
            outputAccumulator[element] = 0.0f;
        }
    }
    if (lid < uint(kVisionAttentionQueries)) {
        rowMax[lid] = -INFINITY;
        rowSum[lid] = 0.0f;
        oldScale[lid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint keyStart = 0; keyStart < sequenceLength;
         keyStart += uint(kVisionAttentionKeys)) {
        auto keySlice = keyTensor.slice(0, int32_t(keyStart));
        auto scores = qkOperation.template get_destination_cooperative_tensor<
            decltype(querySlice), decltype(keySlice), float>();
        for (int element = 0; element < scores.get_capacity(); ++element) {
            if (scores.is_valid_element(element)) scores[element] = 0.0f;
        }
        qkOperation.run(querySlice, keySlice, scores);
        for (int element = 0; element < scores.get_capacity(); ++element) {
            if (!scores.is_valid_element(element)) continue;
            const auto position = scores.get_multidimensional_index(element);
            scoreTile[
                uint(position[1]) * uint(kVisionAttentionKeys)
                + uint(position[0])] = scores[element];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint visible = min(
            uint(kVisionAttentionKeys), sequenceLength - keyStart);
        {
            // Eight lanes per query, eight consecutive keys each. The lane groups
            // sit inside one simdgroup, so an XOR butterfly over masks 1, 2 and 4
            // reduces within a group without touching its neighbours.
            constexpr uint keysPerLane =
                uint(kVisionAttentionKeys) / 8u;
            const uint lane = uint(simdLane);
            const uint query = uint(simdIndex) * 4u + lane / 8u;
            const uint keyBase = (lane % 8u) * keysPerLane;
            const uint row = query * uint(kVisionAttentionKeys);

            float tileMax = -INFINITY;
            if (query < validQueries) {
                for (uint offset = 0; offset < keysPerLane; ++offset) {
                    const uint key = keyBase + offset;
                    if (key < visible) tileMax = max(tileMax, scoreTile[row + key]);
                }
            }
            for (uint mask = 1u; mask < 8u; mask <<= 1) {
                tileMax = max(tileMax, simd_shuffle_xor(tileMax, ushort(mask)));
            }
            const float nextMax = max(rowMax[query], tileMax);
            // The serial summation threads below own different queries than
            // the lanes that computed the running max.
            if (lane % 8u == 0u) {
                nextMaxTile[query] = nextMax;
            }
            for (uint offset = 0; offset < keysPerLane; ++offset) {
                const uint key = keyBase + offset;
                const float weight = query < validQueries && key < visible
                    ? fast::exp(scoreTile[row + key] - nextMax)
                    : 0.0f;
                weightTile[row + key] = weight;
            }
        }
        // Sum the weights serially in key order, so the tile sum does not
        // depend on the lane split above.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < uint(kVisionAttentionQueries)) {
            const uint serialRow = lid * uint(kVisionAttentionKeys);
            float serialSum = 0.0f;
            for (uint key = 0; key < uint(kVisionAttentionKeys); ++key) {
                serialSum += weightTile[serialRow + key];
            }
            oldScale[lid] = rowSum[lid] > 0.0f
                ? fast::exp(rowMax[lid] - nextMaxTile[lid])
                : 0.0f;
            rowSum[lid] = rowSum[lid] * oldScale[lid] + serialSum;
            rowMax[lid] = nextMaxTile[lid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto valueSlice = valueTensor.slice(0, int32_t(keyStart));
        auto outputProduct = pvOperation.template get_destination_cooperative_tensor<
            decltype(weights), decltype(valueSlice), float>();
        for (int element = 0;
             element < outputProduct.get_capacity();
             ++element) {
            if (outputProduct.is_valid_element(element)) {
                outputProduct[element] = 0.0f;
            }
        }
        pvOperation.run(weights, valueSlice, outputProduct);
        for (int element = 0;
             element < outputAccumulator.get_capacity();
             ++element) {
            if (!outputAccumulator.is_valid_element(element)
                || !outputProduct.is_valid_element(element)) continue;
            const auto position =
                outputAccumulator.get_multidimensional_index(element);
            const uint query = uint(position[1]);
            // fma(1, ...) rounds the rescaled accumulator before the add;
            // fast math could otherwise contract the add into one fma.
            outputAccumulator[element] = fma(
                1.0f,
                outputProduct[element],
                outputAccumulator[element] * oldScale[query]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int element = 0;
         element < outputAccumulator.get_capacity();
         ++element) {
        if (!outputAccumulator.is_valid_element(element)) continue;
        const auto position =
            outputAccumulator.get_multidimensional_index(element);
        const uint dimension = uint(position[0]);
        const uint query = uint(position[1]);
        const uint outputHeadDim = layout.unpaddedOutputHeadDim;
        if (query < validQueries && dimension < outputHeadDim) {
            O[(queryStart + query) * numHeads * outputHeadDim
              + head * outputHeadDim + dimension] =
                bfloat(outputAccumulator[element] / rowSum[query]);
        }
    }
}

kernel void mpp_vision_attention_bf16(
    device bfloat* Q [[buffer(0)]],
    device bfloat* K [[buffer(1)]],
    device bfloat* V [[buffer(2)]],
    device bfloat* O [[buffer(3)]],
    constant uint& sequenceLength [[buffer(4)]],
    constant uint& numHeads [[buffer(5)]],
    constant VisionAttentionLayout& layout [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    ushort simdLane [[thread_index_in_simdgroup]],
    ushort simdIndex [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scoreTile[
        kVisionAttentionQueries * kVisionAttentionKeys];
    threadgroup float weightTile[
        kVisionAttentionQueries * kVisionAttentionKeys];
    threadgroup float rowMax[kVisionAttentionQueries];
    threadgroup float rowSum[kVisionAttentionQueries];
    threadgroup float oldScale[kVisionAttentionQueries];
    threadgroup float nextMaxTile[kVisionAttentionQueries];
    mpp_vision_attention_bf16_impl(
        Q, K, V, O, sequenceLength, numHeads, layout, tg, lid, simdLane,
        simdIndex, scoreTile, weightTile, rowMax, rowSum, oldScale, nextMaxTile);
}

#endif // __HAVE_TENSOR__
