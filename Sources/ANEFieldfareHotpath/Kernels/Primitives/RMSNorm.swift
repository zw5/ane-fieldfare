import Metal

/// BF16-weight RMSNorm of one row (`rmsnorm_bf16w`):
///
///   y[i] = x[i] * rsqrt(mean(x^2) + eps) * weight[i]
final class RMSNorm {
    private let pso: any MTLComputePipelineState
    /// Specialized to the hidden size, D = 2816.
    private let psoD2816: any MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("rmsnorm_bf16w")
        self.psoD2816 = try context.pipeline(
            "rmsnorm_bf16w",
            constants: [
                MetalFunctionConstant(index: 30, value: .uint32(2816)),
                MetalFunctionConstant(index: 31, value: .bool(true)),
            ])
    }

    func encodeBF16W(commandBuffer: any MTLCommandBuffer,
                     x: any MTLBuffer, xOffset: Int = 0,
                     weight: any MTLBuffer, weightOffset: Int = 0,
                     out: any MTLBuffer, outOffset: Int = 0,
                     d: UInt32,
                     eps: Float) {
        guard let enc = commandBuffer.makeTracedComputeEncoder() else { return }
        let pipeline = d == 2816 ? psoD2816 : pso
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(x,      offset: xOffset,      index: 0)
        enc.setBuffer(weight, offset: weightOffset, index: 1)
        enc.setBuffer(out,    offset: outOffset,    index: 2)
        enc.setValue(d, index: 3)
        enc.setValue(eps, index: 4)
        // One threadgroup of 256 threads, a multiple of the 32-lane SIMD
        // width, so the reduction has no partial SIMD group.
        let threads = MTLSize(width: min(pipeline.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1)
        enc.dispatchThreads(threads, threadsPerThreadgroup: threads)
        enc.endEncoding()
    }
}
