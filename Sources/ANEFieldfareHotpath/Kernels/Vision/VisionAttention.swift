public import Metal

/// Vision self-attention on the MPP tensor-op kernel, over Q/K/V in the
/// padded head-major layout the QKV projections write directly.
public final class VisionAttention {
    public static let headDimension = 72
    public static let paddedHeadDimension = 80
    private let pipeline: any MTLComputePipelineState
    /// Padded head-major destinations the QKV projections write directly, so
    /// no pad or unpad dispatch runs.
    public let fusedQ: any MTLBuffer
    public let fusedK: any MTLBuffer
    public let fusedV: any MTLBuffer

    /// Zeroes the padded lanes the projections never write. Called once per image;
    /// nothing writes lanes 72 to 79 afterwards, so they stay zero across layers.
    public func encodeClearPadding(commandBuffer: any MTLCommandBuffer) {
        guard let blit = commandBuffer.makeTracedBlitEncoder() else { return }
        for buffer in [fusedQ, fusedK, fusedV] {
            blit.fill(buffer: buffer, range: 0..<buffer.length, value: 0)
        }
        blit.endEncoding()
    }

    public init(context: MetalContext) throws {
        let library = try MetalContext.privateLibrary(
            device: context.device, module: "tensorops",
            includeVisionTensorOps: true)
        guard let function = library.makeFunction(name: "mpp_vision_attention_bf16") else {
            throw MetalError.missingFunction("mpp_vision_attention_bf16")
        }
        pipeline = try context.device.makeComputePipelineState(function: function)
        // The fused layout is addressed by padded GEMM rows, not patch rows.
        let paddedBytes = Self.paddedRowCount(VisionConfig.maximumPatches)
            * 16 * Self.paddedHeadDimension * MemoryLayout<UInt16>.stride
        fusedQ = try Self.makePrivateBuffer(device: context.device, bytes: paddedBytes)
        fusedK = try Self.makePrivateBuffer(device: context.device, bytes: paddedBytes)
        fusedV = try Self.makePrivateBuffer(device: context.device, bytes: paddedBytes)
    }

    package static func paddedRowCount(_ rows: Int) -> Int {
        ((rows + 63) / 64) * 64
    }

    /// Q/K/V are in the padded head-major store, `inputRowStride` rows per
    /// head; the output is row-major and unpadded.
    public func encode(commandBuffer: any MTLCommandBuffer,
                       q: any MTLBuffer, k: any MTLBuffer, v: any MTLBuffer,
                       output: any MTLBuffer,
                       sequenceLength: Int, numHeads: Int,
                       inputRowStride: Int) {
        precondition(sequenceLength > 0 && numHeads > 0)
        precondition(
            inputRowStride >= sequenceLength,
            "the fused layout reads head-major padded Q/K/V and needs their row stride")
        guard let encoder = commandBuffer.makeTracedComputeEncoder() else { return }
        let rows = UInt32(sequenceLength), heads = UInt32(numHeads)
        let layout = VisionAttentionLayout(
            inputRowStride: UInt32(inputRowStride),
            unpaddedOutputHeadDim: UInt32(Self.headDimension))
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(q, offset: 0, index: 0)
        encoder.setBuffer(k, offset: 0, index: 1)
        encoder.setBuffer(v, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setValue(rows, index: 4)
        encoder.setValue(heads, index: 5)
        encoder.setValue(layout, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: (sequenceLength + 15) / 16,
                    height: numHeads, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private static func makePrivateBuffer(device: any MTLDevice, bytes: Int) throws -> any MTLBuffer {
        guard let buffer = device.makeBuffer(length: bytes, options: .storageModePrivate) else {
            throw MetalError.noDevice
        }
        return buffer
    }
}

private struct VisionAttentionLayout {
    var inputRowStride: UInt32
    var unpaddedOutputHeadDim: UInt32
}
