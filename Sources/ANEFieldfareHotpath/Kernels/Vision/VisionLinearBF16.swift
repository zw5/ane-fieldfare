import Metal

/// BF16 linear layers of the image tower, on the register-tiled GEMM. The
/// QKV projections can store straight into attention's padded head-major
/// layout, and the up projection can fold GeGLU into its epilogue.
public final class VisionLinearBF16 {
    /// Head-major padded store for the QKV projections. `rowStride` is the
    /// per-head row count of the destination; lanes in `[headDim, paddedHeadDim)`
    /// are left untouched and must already be zero.
    public struct PaddedHeadStore: Sendable, Equatable {
        public let rowStride: Int
        public let headDim: Int
        public let paddedHeadDim: Int

        public init(rowStride: Int, headDim: Int, paddedHeadDim: Int) {
            self.rowStride = rowStride
            self.headDim = headDim
            self.paddedHeadDim = paddedHeadDim
        }
    }

    private let registerPipeline: MTLComputePipelineState
    private let registerGeGLUPipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        let library = try MetalContext.privateLibrary(
            device: context.device, module: "vision_register_gemm")
        guard let function = library.makeFunction(name: "tff_vision_register_gemm"),
              let geglu = library.makeFunction(
                name: "tff_vision_register_gemm_geglu") else {
            throw MetalError.missingFunction("tff_vision_register_gemm")
        }
        registerPipeline = try context.device.makeComputePipelineState(function: function)
        registerGeGLUPipeline = try context.device.makeComputePipelineState(
            function: geglu)
    }

    public func encode(commandBuffer: MTLCommandBuffer,
                       input: MTLBuffer, inputOffset: Int = 0,
                       weights: MTLBuffer, weightsOffset: Int = 0,
                       output: MTLBuffer, outputOffset: Int = 0,
                       m: Int, n: Int, k: Int,
                       paddedHeadStore: PaddedHeadStore? = nil,
                       fusesGeGLU: Bool = false) {
        precondition(m > 0 && n > 0 && k > 0)
        precondition(!(fusesGeGLU && paddedHeadStore != nil))
        encodeRegisterGEMM(
            commandBuffer: commandBuffer,
            input: input, inputOffset: inputOffset,
            weights: weights, weightsOffset: weightsOffset,
            output: output, outputOffset: outputOffset,
            m: m, n: n, k: k,
            paddedHeadStore: paddedHeadStore,
            pipeline: fusesGeGLU ? registerGeGLUPipeline : registerPipeline)
    }

    private func encodeRegisterGEMM(
        commandBuffer: MTLCommandBuffer,
        input: MTLBuffer, inputOffset: Int,
        weights: MTLBuffer, weightsOffset: Int,
        output: MTLBuffer, outputOffset: Int,
        m: Int, n: Int, k: Int,
        paddedHeadStore: PaddedHeadStore?,
        pipeline: MTLComputePipelineState
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setBuffer(input, offset: inputOffset, index: 0)
        encoder.setBuffer(weights, offset: weightsOffset, index: 1)
        encoder.setBuffer(output, offset: outputOffset, index: 2)
        var rows = UInt32(m), columns = UInt32(n), inner = UInt32(k)
        encoder.setBytes(&rows, length: 4, index: 3)
        encoder.setBytes(&columns, length: 4, index: 4)
        encoder.setBytes(&inner, length: 4, index: 5)
        var store = VisionGEMMPaddedStore(
            rowStride: UInt32(paddedHeadStore?.rowStride ?? 0),
            headDim: UInt32(paddedHeadStore?.headDim ?? 0),
            paddedHeadDim: UInt32(paddedHeadStore?.paddedHeadDim ?? 0))
        encoder.setBytes(&store,
                         length: MemoryLayout<VisionGEMMPaddedStore>.stride,
                         index: 6)
        encoder.setComputePipelineState(pipeline)
        encoder.dispatchThreadgroups(
            MTLSize(width: (n + 127) / 128, height: (m + 63) / 64, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }
}

private struct VisionGEMMPaddedStore {
    var rowStride: UInt32
    var headDim: UInt32
    var paddedHeadDim: UInt32
}
