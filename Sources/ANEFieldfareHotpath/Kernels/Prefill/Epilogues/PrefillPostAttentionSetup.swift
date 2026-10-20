import Foundation
import Metal

final class PrefillPostAttentionSetup {
    private let pso: any MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("prefill_post_attn_setup_block")
    }

    func encode(commandBuffer: any MTLCommandBuffer,
                hidden: any MTLBuffer,
                attn: any MTLBuffer,
                denseX: any MTLBuffer,
                routedX: any MTLBuffer,
                routerX: any MTLBuffer,
                postAttentionWeight: any MTLBuffer,
                postAttentionWeightOffset: Int = 0,
                preFFNWeight: any MTLBuffer,
                preFFNWeightOffset: Int = 0,
                preFFN2Weight: any MTLBuffer,
                preFFN2WeightOffset: Int = 0,
                queryCount: UInt32,
                d: UInt32,
                hiddenStrideElements: UInt32,
                attnStrideElements: UInt32,
                denseStrideElements: UInt32,
                routedStrideElements: UInt32,
                routerStrideElements: UInt32,
                eps: Float) {
        precondition(queryCount > 0, "queryCount must be positive")
        precondition(d > 0 && d <= 4096, "D must be in 1...4096")
        precondition(hiddenStrideElements >= d, "hidden stride is too small")
        precondition(attnStrideElements >= d, "attention stride is too small")
        precondition(denseStrideElements >= d, "dense stride is too small")
        precondition(routedStrideElements >= d, "routed stride is too small")
        precondition(routerStrideElements >= d, "router stride is too small")
        guard let enc = commandBuffer.makeTracedComputeEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(hidden, offset: 0, index: 0)
        enc.setBuffer(attn, offset: 0, index: 1)
        enc.setBuffer(denseX, offset: 0, index: 2)
        enc.setBuffer(routedX, offset: 0, index: 3)
        enc.setBuffer(routerX, offset: 0, index: 4)
        enc.setBuffer(postAttentionWeight, offset: postAttentionWeightOffset, index: 5)
        enc.setBuffer(preFFNWeight, offset: preFFNWeightOffset, index: 6)
        enc.setBuffer(preFFN2Weight, offset: preFFN2WeightOffset, index: 7)
        enc.setValue(queryCount, index: 8)
        enc.setValue(d, index: 9)
        enc.setValue(hiddenStrideElements, index: 10)
        enc.setValue(attnStrideElements, index: 11)
        enc.setValue(denseStrideElements, index: 12)
        enc.setValue(routedStrideElements, index: 13)
        enc.setValue(routerStrideElements, index: 14)
        enc.setValue(eps, index: 15)

        let threads = min(pso.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreadgroups(MTLSize(width: Int(queryCount), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }
}
