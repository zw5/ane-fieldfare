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
                           startPosition: UInt32,
                           queryCount: UInt32,
                           headDim: UInt32,
                           numHeads: UInt32,
                           tokenStrideElements: UInt32,
                           theta: Float) {
        precondition(queryCount > 0, "queryCount must be positive")
        precondition(headDim % 2 == 0, "headDim must be even")
        precondition(tokenStrideElements >= numHeads * headDim,
                     "token stride is too small")
        guard let enc = commandBuffer.makeTracedComputeEncoder() else { return }
        enc.setComputePipelineState(psoDefaultNeox)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setValue(startPosition, index: 1)
        enc.setValue(headDim, index: 2)
        enc.setValue(numHeads, index: 3)
        enc.setValue(tokenStrideElements, index: 4)
        enc.setValue(theta, index: 5)

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
                                startPosition: UInt32,
                                queryCount: UInt32,
                                headDim: UInt32,
                                numHeads: UInt32,
                                rotatedPairs: UInt32,
                                tokenStrideElements: UInt32,
                                theta: Float) {
        precondition(queryCount > 0, "queryCount must be positive")
        precondition(headDim % 2 == 0, "headDim must be even")
        precondition(rotatedPairs * 2 <= headDim,
                     "rotatedPairs * 2 must not exceed headDim")
        precondition(tokenStrideElements >= numHeads * headDim,
                     "token stride is too small")
        guard let enc = commandBuffer.makeTracedComputeEncoder() else { return }
        enc.setComputePipelineState(psoProportionalNeox)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setValue(startPosition, index: 1)
        enc.setValue(headDim, index: 2)
        enc.setValue(numHeads, index: 3)
        enc.setValue(tokenStrideElements, index: 4)
        enc.setValue(theta, index: 5)
        enc.setValue(rotatedPairs, index: 6)

        enc.dispatchThreads(
            MTLSize(width: Int(rotatedPairs), height: Int(numHeads), depth: Int(queryCount)),
            threadsPerThreadgroup: MTLSize(width: min(Int(rotatedPairs),
                                                      psoProportionalNeox.maxTotalThreadsPerThreadgroup),
                                           height: 1,
                                           depth: 1))
        enc.endEncoding()
    }
}
