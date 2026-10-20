import Foundation
import Metal

final class PrefillMoE {
    private let reducePSO: any MTLComputePipelineState

    init(context: MetalContext) throws {
        self.reducePSO = try context.pipeline("prefill_moe_reduce_token_major")
    }

    func encodeReduceTokenMajor(commandBuffer: any MTLCommandBuffer,
                                routePartials: any MTLBuffer,
                                routeWeights: any MTLBuffer,
                                h2: any MTLBuffer,
                                queryCount: UInt32,
                                topK: UInt32,
                                d: UInt32) throws {
        precondition(queryCount > 0, "queryCount must be positive")
        precondition(topK > 0, "topK must be positive")
        precondition(d > 0, "D must be positive")
        let enc = try commandBuffer.makeTracedComputeEncoder()
        enc.setComputePipelineState(reducePSO)
        enc.setBuffer(routePartials, offset: 0, index: 0)
        enc.setBuffer(routeWeights, offset: 0, index: 1)
        enc.setBuffer(h2, offset: 0, index: 2)
        enc.setValue(queryCount, index: 3)
        enc.setValue(topK, index: 4)
        enc.setValue(d, index: 5)
        enc.dispatchThreads(MTLSize(width: Int(d), height: Int(queryCount), depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
    }
}
