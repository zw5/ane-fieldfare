import Foundation
import Metal

/// Single-kernel Gemma 4 decoder-layer tail.
///
/// Equivalent to:
///   h2 = rmsnorm_bf16w(h2, post_feedforward_layernorm_2)
///   h12 = h1 + h2
///   h12 = rmsnorm_bf16w(h12, post_feedforward_layernorm)
///   hidden = (hidden + h12) * layer_scalar
///
/// `encodeWithNorms` also takes the two decode norms on either side of it:
/// the shared expert's post-FFN-1 norm of `h1` and the next layer's input
/// norm of the new `hidden`.
final class FusedLayerTail {
    private let pso: any MTLComputePipelineState
    private let specializedPSO: any MTLComputePipelineState
    private let normsPSO: any MTLComputePipelineState
    private let normsSpecializedPSO: any MTLComputePipelineState

    /// The norms reproduce `rmsnorm_bf16w` bitwise only with its 256-thread
    /// row; a pipeline that cannot run 256 threads leaves the norms unfused.
    var fusesNorms: Bool { normsPSO.maxTotalThreadsPerThreadgroup >= Self.normThreads }
    private static let normThreads = 256

    init(context: MetalContext) throws {
        let d2816: [MetalFunctionConstant] = [
            MetalFunctionConstant(index: 80, value: .uint32(2816)),
            MetalFunctionConstant(index: 86, value: .bool(true)),
        ]
        self.pso = try context.pipeline("fused_layer_tail")
        self.specializedPSO = try context.pipeline("fused_layer_tail", constants: d2816)
        self.normsPSO = try context.pipeline("fused_layer_tail_norms")
        self.normsSpecializedPSO = try context.pipeline("fused_layer_tail_norms", constants: d2816)
    }

    func encode(commandBuffer cb: any MTLCommandBuffer,
                       h2: any MTLBuffer,
                       h1: any MTLBuffer,
                       hidden: any MTLBuffer,
                       postFFN2Weight: any MTLBuffer,
                       postFFN2WeightOffset: Int = 0,
                       postFFNWeight: any MTLBuffer,
                       postFFNWeightOffset: Int = 0,
                       d: UInt32,
                       eps: Float,
                       layerScalar: Float) throws {
        precondition(d <= 4096,
                     "D > 4096 exceeds the fused-tail threadgroup scratch (kFusedMaxD)")
        let enc = try cb.makeTracedComputeEncoder()
        let pipeline = d == 2816 ? specializedPSO : pso
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(h2,             offset: 0,                    index: 0)
        enc.setBuffer(h1,             offset: 0,                    index: 1)
        enc.setBuffer(hidden,         offset: 0,                    index: 2)
        enc.setBuffer(postFFN2Weight, offset: postFFN2WeightOffset, index: 3)
        enc.setBuffer(postFFNWeight,  offset: postFFNWeightOffset,  index: 4)
        enc.setValue(d, index: 5)
        enc.setValue(eps, index: 6)
        enc.setValue(layerScalar, index: 7)

        let threads = min(Int(pipeline.maxTotalThreadsPerThreadgroup), 256)
        enc.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// The tail with `h1` taken before its post-FFN-1 norm (`postFFN1Weight`)
    /// and, when `next` is given, the next layer's input norm of the new
    /// `hidden` written to `next.normed`. Bitwise the same as
    /// `rmsnorm_bf16w(h1)`, `encode`, then `rmsnorm_bf16w(hidden)`; callers
    /// check `fusesNorms` first.
    func encodeWithNorms(commandBuffer cb: any MTLCommandBuffer,
                         h2: any MTLBuffer,
                         h1: any MTLBuffer,
                         hidden: any MTLBuffer,
                         postFFN1Weight: TensorView,
                         postFFN2Weight: TensorView,
                         postFFNWeight: TensorView,
                         next: (inputNorm: TensorView, normed: any MTLBuffer)?,
                         d: UInt32,
                         eps: Float,
                         layerScalar: Float) throws {
        precondition(d <= 4096,
                     "D > 4096 exceeds the fused-tail threadgroup scratch (kFusedMaxD)")
        let enc = try cb.makeTracedComputeEncoder()
        enc.setComputePipelineState(d == 2816 ? normsSpecializedPSO : normsPSO)
        enc.setBuffer(h2, offset: 0, index: 0)
        enc.setBuffer(h1, offset: 0, index: 1)
        enc.setBuffer(hidden, offset: 0, index: 2)
        enc.setBuffer(postFFN2Weight.buffer, offset: Int(postFFN2Weight.offset), index: 3)
        enc.setBuffer(postFFNWeight.buffer, offset: Int(postFFNWeight.offset), index: 4)
        enc.setValue(d, index: 5)
        enc.setValue(eps, index: 6)
        enc.setValue(layerScalar, index: 7)
        enc.setBuffer(postFFN1Weight.buffer, offset: Int(postFFN1Weight.offset), index: 8)
        // Without a next layer the kernel never reads 9 or 10; bind valid
        // buffers anyway.
        let nextNorm = next?.inputNorm ?? postFFN1Weight
        enc.setBuffer(nextNorm.buffer, offset: Int(nextNorm.offset), index: 9)
        enc.setBuffer(next?.normed ?? hidden, offset: 0, index: 10)
        enc.setValue(UInt32(next == nil ? 0 : 1), index: 11)
        let threads = Self.normThreads
        enc.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }
}
