import Foundation
import Metal

/// Single-kernel Gemma 4 Q/K/V epilogue, per head:
///   Q = rope_neox(rmsnorm(Q) * q_norm)
///   K = rope_neox(rmsnorm(K) * k_norm)
///   V = rmsnorm(V)
final class FusedQKVEpilogue {
    private struct Shape: Hashable {
        var headDim: UInt32
        var numQHeads: UInt32
        var numKVHeads: UInt32
        var rotatedPairs: UInt32
    }

    private let pso: any MTLComputePipelineState
    private let specializedPSOs: [Shape: any MTLComputePipelineState]
    private let rowsPSOs: [Shape: any MTLComputePipelineState]
    private static let realDecodeShapes: [Shape] = [
        Shape(headDim: 256, numQHeads: 16, numKVHeads: 8, rotatedPairs: 128),
        Shape(headDim: 512, numQHeads: 16, numKVHeads: 2, rotatedPairs: 64),
    ]

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("fused_qkv_epilogue")
        var variants: [Shape: any MTLComputePipelineState] = [:]
        for shape in Self.realDecodeShapes {
            variants[shape] = try context.pipeline(
                "fused_qkv_epilogue",
                constants: [
                    MetalFunctionConstant(index: 82, value: .uint32(shape.headDim)),
                    MetalFunctionConstant(index: 83, value: .uint32(shape.numQHeads)),
                    MetalFunctionConstant(index: 84, value: .uint32(shape.numKVHeads)),
                    MetalFunctionConstant(index: 85, value: .uint32(shape.rotatedPairs)),
                ])
        }
        self.specializedPSOs = variants
        var rowVariants: [Shape: any MTLComputePipelineState] = [:]
        for shape in Self.realDecodeShapes {
            rowVariants[shape] = try context.pipeline(
                "fused_qkv_epilogue_rows",
                constants: [
                    MetalFunctionConstant(index: 82, value: .uint32(shape.headDim)),
                    MetalFunctionConstant(index: 83, value: .uint32(shape.numQHeads)),
                    MetalFunctionConstant(index: 84, value: .uint32(shape.numKVHeads)),
                    MetalFunctionConstant(index: 85, value: .uint32(shape.rotatedPairs)),
                ])
        }
        self.rowsPSOs = rowVariants
    }

    /// `encode` over `rows` tokens of a block in one dispatch: token r is at
    /// `position + r`, its Q `qStrideElements` and K/V `kvStrideElements`
    /// past token r - 1. Real decode shapes only.
    func encodeRows(commandBuffer cb: any MTLCommandBuffer,
                    q: any MTLBuffer, k: any MTLBuffer, v: any MTLBuffer,
                    qWeight: any MTLBuffer, qWeightOffset: Int,
                    kWeight: any MTLBuffer, kWeightOffset: Int,
                    headDim: UInt32, numQHeads: UInt32, numKVHeads: UInt32,
                    position: UInt32, rows: Int,
                    qStrideElements: UInt32, kvStrideElements: UInt32,
                    theta: Float, rotatedPairs: UInt32, eps: Float) throws -> Bool {
        guard let pipeline = rowsPSOs[Shape(headDim: headDim, numQHeads: numQHeads,
                                            numKVHeads: numKVHeads, rotatedPairs: rotatedPairs)]
        else { return false }
        let enc = try cb.makeTracedComputeEncoder()
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(q, offset: 0, index: 0)
        enc.setBuffer(k, offset: 0, index: 1)
        enc.setBuffer(v, offset: 0, index: 2)
        enc.setBuffer(qWeight, offset: qWeightOffset, index: 3)
        enc.setBuffer(kWeight, offset: kWeightOffset, index: 4)
        let values: [UInt32] = [headDim, numQHeads, numKVHeads, position]
        for i in values.indices { enc.setValue(values[i], index: 5 + i) }
        enc.setValue(theta, index: 9)
        enc.setValue(rotatedPairs, index: 10)
        enc.setValue(eps, index: 11)
        enc.setValue(qStrideElements, index: 12)
        enc.setValue(kvStrideElements, index: 13)
        let threads = min(Int(pipeline.maxTotalThreadsPerThreadgroup), 256)
        enc.dispatchThreadgroups(MTLSize(width: Int(numQHeads + 2 * numKVHeads), height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
        return true
    }

    func encode(commandBuffer cb: any MTLCommandBuffer,
                       q: any MTLBuffer,
                       qOffset: Int = 0,
                       k: any MTLBuffer,
                       kOffset: Int = 0,
                       v: any MTLBuffer,
                       vOffset: Int = 0,
                       qWeight: any MTLBuffer,
                       qWeightOffset: Int = 0,
                       kWeight: any MTLBuffer,
                       kWeightOffset: Int = 0,
                       headDim: UInt32,
                       numQHeads: UInt32,
                       numKVHeads: UInt32,
                       position: UInt32,
                       theta: Float,
                       rotatedPairs: UInt32,
                       eps: Float) throws {
        precondition(headDim <= 512,
                     "headDim > 512 exceeds the fused QKV epilogue scratch")
        precondition(rotatedPairs * 2 <= headDim,
                     "rotatedPairs must fit inside one NeoX head")
        let enc = try cb.makeTracedComputeEncoder()
        let pipeline = specializedPSOs[Shape(headDim: headDim,
                                             numQHeads: numQHeads,
                                             numKVHeads: numKVHeads,
                                             rotatedPairs: rotatedPairs)] ?? pso
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(q,       offset: qOffset,       index: 0)
        enc.setBuffer(k,       offset: kOffset,       index: 1)
        enc.setBuffer(v,       offset: vOffset,       index: 2)
        enc.setBuffer(qWeight, offset: qWeightOffset, index: 3)
        enc.setBuffer(kWeight, offset: kWeightOffset, index: 4)
        enc.setValue(headDim, index: 5)
        enc.setValue(numQHeads, index: 6)
        enc.setValue(numKVHeads, index: 7)
        enc.setValue(position, index: 8)
        enc.setValue(theta, index: 9)
        enc.setValue(rotatedPairs, index: 10)
        enc.setValue(eps, index: 11)

        let threads = min(Int(pipeline.maxTotalThreadsPerThreadgroup), 256)
        let groups = Int(numQHeads + 2 * numKVHeads)
        enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }
}
