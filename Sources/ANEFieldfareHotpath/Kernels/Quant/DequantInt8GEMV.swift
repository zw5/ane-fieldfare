import Metal

/// `dequant_int8_gemv_simd`: y = W * x, where W (M rows, N cols) is MLX-affine
/// 8-bit (group=64) with BF16 scale + BF16 bias per group, x is FP16 [N], y is
/// FP16 [M]. One SIMD group (32 threads) runs per output row. The 8-bit shared
/// expert's down projection is its caller.
final class DequantInt8GEMV {
    private let pso: any MTLComputePipelineState
    /// Specialized to the 26B-A4B shared expert down projection, 2816 x 2112.
    private let psoDown: any MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("dequant_int8_gemv_simd")
        self.psoDown = try context.pipeline(
            "dequant_int8_gemv_simd",
            constants: [
                MetalFunctionConstant(index: 70, value: .uint32(2816)),
                MetalFunctionConstant(index: 71, value: .uint32(2112)),
                MetalFunctionConstant(index: 72, value: .bool(true)),
            ])
    }

    /// Encodes the GEMV onto `commandBuffer`. Offsets allow passing the same
    /// mmap'd `MTLBuffer` for weights / scales / biases when they share a
    /// page-aligned blob.
    func encode(commandBuffer: any MTLCommandBuffer,
                       weights: any MTLBuffer, weightsOffset: Int = 0,
                       scales:  any MTLBuffer, scalesOffset:  Int = 0,
                       biases:  any MTLBuffer, biasesOffset:  Int = 0,
                       x:       any MTLBuffer,
                       xOffset: Int = 0,
                       y:       any MTLBuffer,
                       yOffset: Int = 0,
                       m: UInt32,
                       n: UInt32) {
        precondition(n % UInt32(Quantization.groupSize) == 0,
                     "N must be a multiple of \(Quantization.groupSize)")
        precondition(xOffset >= 0 && yOffset >= 0, "buffer offsets must be non-negative")
        guard let enc = commandBuffer.makeTracedComputeEncoder() else { return }
        enc.setComputePipelineState(m == 2816 && n == 2112 ? psoDown : pso)
        enc.setBuffer(weights, offset: weightsOffset, index: 0)
        enc.setBuffer(scales,  offset: scalesOffset,  index: 1)
        enc.setBuffer(biases,  offset: biasesOffset,  index: 2)
        enc.setBuffer(x,       offset: xOffset,       index: 3)
        enc.setBuffer(y,       offset: yOffset,       index: 4)
        enc.setValue(m, index: 5)
        enc.setValue(n, index: 6)

        let rowsPerTG = 8
        let tgSize  = MTLSize(width: 32 * rowsPerTG, height: 1, depth: 1)
        let tgCount = MTLSize(width: (Int(m) + rowsPerTG - 1) / rowsPerTG,
                              height: 1, depth: 1)
        enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
    }
}
