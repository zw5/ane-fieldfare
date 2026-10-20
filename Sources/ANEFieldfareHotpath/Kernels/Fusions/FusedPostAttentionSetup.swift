import Foundation
import Metal

/// Single-kernel Gemma 4 post-attention setup.
///
/// Equivalent to:
///   attn = rmsnorm_bf16w(attn, post_attention_norm)
///   hidden = hidden + attn
///   dense_x = rmsnorm_bf16w(hidden, pre_feedforward_layernorm)
///   routed_x = rmsnorm_bf16w(hidden, pre_feedforward_layernorm_2)
///   router_x = rmsnorm(hidden), no weight
final class FusedPostAttentionSetup {
    private let pso: any MTLComputePipelineState
    private let specializedPSO: any MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("fused_post_attn_setup")
        self.specializedPSO = try context.pipeline(
            "fused_post_attn_setup",
            constants: [
                MetalFunctionConstant(index: 80, value: .uint32(2816)),
                MetalFunctionConstant(index: 86, value: .bool(true)),
            ])
    }

    func encode(commandBuffer cb: any MTLCommandBuffer,
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
                       d: UInt32,
                       eps: Float) throws {
        precondition(d <= 4096,
                     "D > 4096 exceeds the fused post-attention setup scratch")
        let enc = try cb.makeTracedComputeEncoder()
        let pipeline = d == 2816 ? specializedPSO : pso
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(hidden,              offset: 0,                         index: 0)
        enc.setBuffer(attn,                offset: 0,                         index: 1)
        enc.setBuffer(denseX,              offset: 0,                         index: 2)
        enc.setBuffer(routedX,             offset: 0,                         index: 3)
        enc.setBuffer(routerX,             offset: 0,                         index: 4)
        enc.setBuffer(postAttentionWeight, offset: postAttentionWeightOffset, index: 5)
        enc.setBuffer(preFFNWeight,        offset: preFFNWeightOffset,        index: 6)
        enc.setBuffer(preFFN2Weight,       offset: preFFN2WeightOffset,       index: 7)
        enc.setValue(d, index: 8)
        enc.setValue(eps, index: 9)

        let threads = min(Int(pipeline.maxTotalThreadsPerThreadgroup), 256)
        enc.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }
}
