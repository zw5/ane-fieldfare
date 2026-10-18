import Foundation
import Metal

public final class VisionAttention {
    public static let headDimension = 72
    public static let paddedHeadDimension = 80
    /// Native kernel for row-major Q/K/V; 8 query rows per threadgroup.
    private let nativePipeline: MTLComputePipelineState
    /// MPP tensor-op kernel over the padded head-major layout.
    private let mppPipeline: MTLComputePipelineState
    /// Padded head-major destinations the QKV projections write directly, so
    /// no pad or unpad dispatch runs.
    public let fusedQ: MTLBuffer
    public let fusedK: MTLBuffer
    public let fusedV: MTLBuffer

    public var additionalScratchBytes: Int {
        fusedQ.length + fusedK.length + fusedV.length
    }

    /// Zeroes the padded lanes the projections never write. Called once per image;
    /// nothing writes lanes 72 to 79 afterwards, so they stay zero across layers.
    public func encodeClearPadding(commandBuffer: MTLCommandBuffer) {
        guard let blit = commandBuffer.makeBlitCommandEncoder() else { return }
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
        mppPipeline = try context.device.makeComputePipelineState(function: function)
        nativePipeline = try context.pipeline("vision_attention_online_72")
        // The fused layout is addressed by padded GEMM rows, not patch rows.
        let paddedBytes = Self.paddedRowCount(VisionConfig().maximumPatches)
            * 16 * Self.paddedHeadDimension * MemoryLayout<UInt16>.stride
        fusedQ = try Self.makePrivateBuffer(device: context.device, bytes: paddedBytes)
        fusedK = try Self.makePrivateBuffer(device: context.device, bytes: paddedBytes)
        fusedV = try Self.makePrivateBuffer(device: context.device, bytes: paddedBytes)
    }

    package static func paddedRowCount(_ rows: Int) -> Int {
        ((rows + 63) / 64) * 64
    }

    /// `inputRowStride > 0` means Q/K/V are in the padded head-major store
    /// (`fusedQ`/`fusedK`/`fusedV`) with that per-head row count, and runs the
    /// MPP kernel. Stride 0 means row-major Q/K/V and runs the native kernel;
    /// reading a row-major call as 80-stride head-major would be silent garbage.
    public func encode(commandBuffer: MTLCommandBuffer,
                       q: MTLBuffer, k: MTLBuffer, v: MTLBuffer,
                       output: MTLBuffer,
                       sequenceLength: Int, numHeads: Int = 16,
                       inputRowStride: Int = 0) {
        precondition(sequenceLength > 0 && numHeads > 0)
        guard inputRowStride > 0 else {
            encodeNative(commandBuffer: commandBuffer, q: q, k: k, v: v,
                         output: output,
                         sequenceLength: sequenceLength, numHeads: numHeads)
            return
        }
        precondition(
            inputRowStride >= sequenceLength,
            "the fused layout reads head-major padded Q/K/V and needs their row stride")
        encodeMPPTensorOps(
            commandBuffer: commandBuffer,
            sequenceLength: sequenceLength, numHeads: numHeads,
            paddedQ: q, paddedK: k, paddedV: v,
            output: output,
            inputRowStride: inputRowStride)
    }

    private func encodeNative(commandBuffer: MTLCommandBuffer,
                              q: MTLBuffer, k: MTLBuffer, v: MTLBuffer,
                              output: MTLBuffer,
                              sequenceLength: Int, numHeads: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(nativePipeline)
        encoder.setBuffer(q, offset: 0, index: 0)
        encoder.setBuffer(k, offset: 0, index: 1)
        encoder.setBuffer(v, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        var length = UInt32(sequenceLength)
        var heads = UInt32(numHeads)
        encoder.setBytes(&length, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&heads, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.dispatchThreadgroups(
            MTLSize(width: (sequenceLength + 7) / 8,
                    height: numHeads, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeMPPTensorOps(
        commandBuffer: MTLCommandBuffer,
        sequenceLength: Int, numHeads: Int,
        paddedQ: MTLBuffer, paddedK: MTLBuffer, paddedV: MTLBuffer,
        output: MTLBuffer,
        inputRowStride: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        var rows = UInt32(sequenceLength), heads = UInt32(numHeads)
        // Row-major unpadded output; parallelExp softmax keeps the tile sum
        // sequential per query, bit-identical to the serial phase and faster.
        var layout = VisionAttentionLayout(
            inputRowStride: UInt32(inputRowStride),
            unpaddedOutputHeadDim: UInt32(Self.headDimension),
            parallelSoftmax: 2)
        encoder.setComputePipelineState(mppPipeline)
        encoder.setBuffer(paddedQ, offset: 0, index: 0)
        encoder.setBuffer(paddedK, offset: 0, index: 1)
        encoder.setBuffer(paddedV, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setBytes(&rows, length: 4, index: 4)
        encoder.setBytes(&heads, length: 4, index: 5)
        encoder.setBytes(&layout,
                         length: MemoryLayout<VisionAttentionLayout>.stride,
                         index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: (sequenceLength + 15) / 16,
                    height: numHeads, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private static func makePrivateBuffer(device: MTLDevice, bytes: Int) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: bytes, options: .storageModePrivate) else {
            throw MetalError.noDevice
        }
        return buffer
    }

}

private struct VisionAttentionLayout {
    var inputRowStride: UInt32
    var unpaddedOutputHeadDim: UInt32
    var parallelSoftmax: UInt32
}
