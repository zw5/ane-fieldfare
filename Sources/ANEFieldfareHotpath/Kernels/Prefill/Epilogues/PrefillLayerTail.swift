import Foundation
import Metal

final class PrefillLayerTail {
    private let pso: any MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("prefill_layer_tail_block")
    }

    func encode(commandBuffer: any MTLCommandBuffer,
                       h2: any MTLBuffer,
                       h2Offset: Int = 0,
                       h1: any MTLBuffer,
                       h1Offset: Int = 0,
                       hidden: any MTLBuffer,
                       hiddenOffset: Int = 0,
                       postFFN2Weight: any MTLBuffer,
                       postFFN2WeightOffset: Int = 0,
                       postFFNWeight: any MTLBuffer,
                       postFFNWeightOffset: Int = 0,
                       queryCount: UInt32,
                       d: UInt32,
                       h2StrideElements: UInt32,
                       h1StrideElements: UInt32,
                       hiddenStrideElements: UInt32,
                       eps: Float,
                       layerScalar: Float) {
        precondition(queryCount > 0, "queryCount must be positive")
        precondition(d > 0 && d <= 4096, "D must be in 1...4096")
        precondition(h2StrideElements >= d, "h2 stride is too small")
        precondition(h1StrideElements >= d, "h1 stride is too small")
        precondition(hiddenStrideElements >= d, "hidden stride is too small")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(h2, offset: h2Offset, index: 0)
        enc.setBuffer(h1, offset: h1Offset, index: 1)
        enc.setBuffer(hidden, offset: hiddenOffset, index: 2)
        enc.setBuffer(postFFN2Weight, offset: postFFN2WeightOffset, index: 3)
        enc.setBuffer(postFFNWeight, offset: postFFNWeightOffset, index: 4)
        let tVar = queryCount
        let dVar = d
        let h2Stride = h2StrideElements
        let h1Stride = h1StrideElements
        let hiddenStride = hiddenStrideElements
        let epsVar = eps
        let scaleVar = layerScalar
        enc.setValue(tVar, index: 5)
        enc.setValue(dVar, index: 6)
        enc.setValue(h2Stride, index: 7)
        enc.setValue(h1Stride, index: 8)
        enc.setValue(hiddenStride, index: 9)
        enc.setValue(epsVar, index: 10)
        enc.setValue(scaleVar, index: 11)

        let threads = min(pso.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreadgroups(MTLSize(width: Int(queryCount), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }
}
