import Foundation
import Metal

final class PrefillRoPE {
    private let psoDefaultNeox: any MTLComputePipelineState
    private let psoProportionalNeox: any MTLComputePipelineState

    init(context: MetalContext) throws {
        self.psoDefaultNeox = try context.pipeline("prefill_rope_default_neox_block")
        self.psoProportionalNeox = try context.pipeline("prefill_rope_proportional_neox_block")
    }

    func encodeDefaultNeox(commandBuffer: any MTLCommandBuffer,
                                  data: any MTLBuffer,
                                  dataOffset: Int = 0,
                                  startPosition: UInt32,
                                  queryCount: UInt32,
                                  headDim: UInt32,
                                  numHeads: UInt32,
                                  tokenStrideElements: UInt32,
                                  theta: Float = 10_000.0) {
        precondition(queryCount > 0, "queryCount must be positive")
        precondition(headDim % 2 == 0, "headDim must be even")
        precondition(tokenStrideElements >= numHeads * headDim,
                     "token stride is too small")
        guard let enc = commandBuffer.makeTracedComputeEncoder() else { return }
        enc.setComputePipelineState(psoDefaultNeox)
        enc.setBuffer(data, offset: dataOffset, index: 0)
        let start = startPosition
        let hd = headDim
        let heads = numHeads
        let stride = tokenStrideElements
        let thetaVar = theta
        enc.setValue(start, index: 1)
        enc.setValue(hd, index: 2)
        enc.setValue(heads, index: 3)
        enc.setValue(stride, index: 4)
        enc.setValue(thetaVar, index: 5)

        let pairs = Int(headDim) / 2
        enc.dispatchThreads(
            MTLSize(width: pairs, height: Int(numHeads), depth: Int(queryCount)),
            threadsPerThreadgroup: MTLSize(width: min(pairs, psoDefaultNeox.maxTotalThreadsPerThreadgroup),
                                           height: 1,
                                           depth: 1))
        enc.endEncoding()
    }

    func encodeProportionalNeox(commandBuffer: any MTLCommandBuffer,
                                       data: any MTLBuffer,
                                       dataOffset: Int = 0,
                                       startPosition: UInt32,
                                       queryCount: UInt32,
                                       headDim: UInt32,
                                       numHeads: UInt32,
                                       rotatedPairs: UInt32,
                                       tokenStrideElements: UInt32,
                                       theta: Float = 1_000_000.0) {
        precondition(queryCount > 0, "queryCount must be positive")
        precondition(headDim % 2 == 0, "headDim must be even")
        precondition(rotatedPairs * 2 <= headDim,
                     "rotatedPairs * 2 must not exceed headDim")
        precondition(tokenStrideElements >= numHeads * headDim,
                     "token stride is too small")
        guard let enc = commandBuffer.makeTracedComputeEncoder() else { return }
        enc.setComputePipelineState(psoProportionalNeox)
        enc.setBuffer(data, offset: dataOffset, index: 0)
        let start = startPosition
        let hd = headDim
        let heads = numHeads
        let stride = tokenStrideElements
        let thetaVar = theta
        let rp = rotatedPairs
        enc.setValue(start, index: 1)
        enc.setValue(hd, index: 2)
        enc.setValue(heads, index: 3)
        enc.setValue(stride, index: 4)
        enc.setValue(thetaVar, index: 5)
        enc.setValue(rp, index: 6)

        enc.dispatchThreads(
            MTLSize(width: Int(rotatedPairs), height: Int(numHeads), depth: Int(queryCount)),
            threadsPerThreadgroup: MTLSize(width: min(Int(rotatedPairs),
                                                      psoProportionalNeox.maxTotalThreadsPerThreadgroup),
                                           height: 1,
                                           depth: 1))
        enc.endEncoding()
    }
}
