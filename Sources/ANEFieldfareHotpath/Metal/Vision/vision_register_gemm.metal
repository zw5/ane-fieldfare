#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

/// Head-major padded store descriptor. `rowStride == 0` selects the plain
/// row-major store `output[m * columns + n]`; otherwise column `n` is split into
/// head `n / headDim` and lane `n % headDim` and written at
/// `(head * rowStride + m) * paddedHeadDim + lane`, which is the layout the
/// padded attention kernel reads. Lanes in `[headDim, paddedHeadDim)` are never
/// written here; they are zeroed once per image and stay zero.
struct VisionGEMMPaddedStore {
    uint rowStride;
    uint headDim;
    uint paddedHeadDim;
};

/// tanh-approximated GELU(gate) * up, with `up` rounded to bfloat first, as
/// an unfused path storing it in a bfloat buffer would.
inline bfloat vision_geglu_combine(bfloat gate, float up) {
    const float x = float(gate);
    const float tanhInput = clamp(
        0.7978845608f * (x + 0.044715f * x * x * x), -20.0f, 20.0f);
    const float activated = 0.5f * x * (1.0f + tanh(tanhInput));
    return bfloat(activated * float(bfloat(up)));
}

/// `FuseGeGLU` folds GeGLU into this GEMM's epilogue: the kernel computes the
/// `up` projection and combines it with the `gate` values a previous dispatch
/// already wrote to `output`.
template <bool FuseGeGLU>
inline void vision_register_gemm_impl(
    device bfloat* input,
    device bfloat* weights,
    device bfloat* output,
    constant uint& rows,
    constant uint& columns,
    constant uint& inner,
    constant VisionGEMMPaddedStore& paddedStore,
    uint2 group,
    ushort simdgroup) {
    constexpr int tileM = 64;
    constexpr int tileN = 128;
    constexpr int simdTileM = 16;
    constexpr int simdTileN = 32;
    constexpr int simdTileK = 32;
    constexpr int simdgroupsN = tileN / simdTileN;
    constexpr auto descriptor = matmul2d_descriptor(
        simdTileM, simdTileN, simdTileK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroup> operation;
    using device_bfloat_tensor = tensor<
        device bfloat, dextents<int32_t, 2>, tensor_inline>;
    device_bfloat_tensor matrixA(
        input,
        dextents<int32_t, 2>(int32_t(inner), int32_t(rows)),
        array<int32_t, 2>({1, int32_t(inner)}));
    device_bfloat_tensor matrixB(
        weights,
        dextents<int32_t, 2>(int32_t(inner), int32_t(columns)),
        array<int32_t, 2>({1, int32_t(inner)}));

    const uint localM = uint(simdgroup) / uint(simdgroupsN) * 32;
    const uint localN = uint(simdgroup) % uint(simdgroupsN) * uint(simdTileN);
    const uint startM = group.y * uint(tileM);
    const uint startN = group.x * uint(tileN);
    auto a0 = operation.get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto a1 = operation.get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto b = operation.get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    auto accumulator0 = operation.get_destination_cooperative_tensor<
        remove_addrspace_t<decltype(a0)>, remove_addrspace_t<decltype(b)>, float>();
    auto accumulator1 = operation.get_destination_cooperative_tensor<
        remove_addrspace_t<decltype(a1)>, remove_addrspace_t<decltype(b)>, float>();
    // The sums arrays hold the (simdTileM * simdTileN) / 32 elements each
    // thread owns of the 16x32 tile. get_capacity() is only known at run
    // time; a larger one would overrun them, so the tile is left unwritten.
    constexpr int accumulatorSlots = (simdTileM * simdTileN) / 32;
    if (int(accumulator0.get_capacity()) > accumulatorSlots ||
        int(accumulator1.get_capacity()) > accumulatorSlots) {
        return;
    }
    float sums0[accumulatorSlots];
    float sums1[accumulatorSlots];
    for (int element = 0; element < accumulator0.get_capacity(); ++element) {
        sums0[element] = 0.0f;
        sums1[element] = 0.0f;
    }

    for (uint groupK = 0; groupK < inner; groupK += 64u) {
        for (int element = 0; element < accumulator0.get_capacity(); ++element) {
            accumulator0[element] = 0.0f;
            accumulator1[element] = 0.0f;
        }
        for (uint startK = groupK;
             startK < min(groupK + 64u, inner);
             startK += uint(simdTileK)) {
            auto tileA0 = matrixA.slice(
                int32_t(startK), int32_t(startM + localM));
            auto tileA1 = matrixA.slice(
                int32_t(startK), int32_t(startM + localM + 16));
            auto tileB = matrixB.slice(
                int32_t(startK), int32_t(startN + localN));
            a0.load(tileA0);
            a1.load(tileA1);
            b.load(tileB);
            operation.run(a0, b, accumulator0);
            operation.run(a1, b, accumulator1);
        }
        for (int element = 0; element < accumulator0.get_capacity(); ++element) {
            sums0[element] += accumulator0[element];
            sums1[element] += accumulator1[element];
        }
    }

    const bool storePadded = paddedStore.rowStride != 0u;
    for (int element = 0; element < accumulator0.get_capacity(); ++element) {
        const auto position = accumulator0.get_multidimensional_index(element);
        const uint m0 = startM + localM + uint(position[1]);
        const uint m1 = m0 + 16;
        const uint n = startN + localN + uint(position[0]);
        if (n >= columns) continue;
        uint offset0;
        uint offset1;
        if (storePadded) {
            const uint head = n / paddedStore.headDim;
            const uint lane = n % paddedStore.headDim;
            const uint headBase = head * paddedStore.rowStride;
            offset0 = (headBase + m0) * paddedStore.paddedHeadDim + lane;
            offset1 = (headBase + m1) * paddedStore.paddedHeadDim + lane;
        } else {
            offset0 = m0 * columns + n;
            offset1 = m1 * columns + n;
        }
        if (FuseGeGLU) {
            if (m0 < rows) output[offset0] = vision_geglu_combine(
                output[offset0], sums0[element]);
            if (m1 < rows) output[offset1] = vision_geglu_combine(
                output[offset1], sums1[element]);
        } else {
            if (m0 < rows) output[offset0] = bfloat(sums0[element]);
            if (m1 < rows) output[offset1] = bfloat(sums1[element]);
        }
    }
}

kernel void vision_register_gemm(
    device bfloat* input [[buffer(0)]],
    device bfloat* weights [[buffer(1)]],
    device bfloat* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& columns [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    constant VisionGEMMPaddedStore& paddedStore [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    vision_register_gemm_impl<false>(
        input, weights, output, rows, columns, inner, paddedStore,
        group, simdgroup);
}

kernel void vision_register_gemm_geglu(
    device bfloat* input [[buffer(0)]],
    device bfloat* weights [[buffer(1)]],
    device bfloat* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& columns [[buffer(4)]],
    constant uint& inner [[buffer(5)]],
    constant VisionGEMMPaddedStore& paddedStore [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    vision_register_gemm_impl<true>(
        input, weights, output, rows, columns, inner, paddedStore,
        group, simdgroup);
}
