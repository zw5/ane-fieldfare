import Foundation
import Metal

enum PrefillAttentionLayerKind: Sendable, Equatable {
    case full
    case slidingWindow
}

/// Mirrored field for field by `PrefillAttentionParams` in prefill.metal.
struct PrefillAttentionParams: Sendable, Equatable {
    var startPosition: UInt32
    var queryCount: UInt32
    var headDim: UInt32
    var numQHeads: UInt32
    var numKVHeads: UInt32
    var kvValidCount: UInt32
    var slidingWindow: UInt32
    var kvTokenStrideElements: UInt32
    var qTokenStrideElements: UInt32
    var oTokenStrideElements: UInt32
    var scale: Float
    var bidirectionalBlockStart: UInt32
    var bidirectionalBlockEnd: UInt32
}

final class PrefillAttention {
    private let context: MetalContext
    private let psoCausalTiled: any MTLComputePipelineState

    init(context: MetalContext) throws {
        self.context = context
        self.psoCausalTiled = try context.pipeline("attention_prefill_causal_tiled")
    }

    /// Gemma's shapes without a bidirectional block run the grouped-query
    /// tensor kernel, other head sizes 256 and 512 the SIMD kernel, and the
    /// rest the tiled kernel.
    func encodeCausal(commandBuffer: any MTLCommandBuffer,
                      q: any MTLBuffer,
                      k: any MTLBuffer,
                      v: any MTLBuffer,
                      out: any MTLBuffer,
                      params: PrefillAttentionParams,
                      kvRingCapacity: UInt32,
                      layerKind: PrefillAttentionLayerKind) throws {
        let params = Self.effective(params, layerKind: layerKind)
        if let shape = Self.gqaTensorShape(params) {
            let constants = kvRingCapacity > 0
                ? [MetalFunctionConstant(index: 76, value: .uint32(kvRingCapacity))] : []
            let pipeline = try context.pipeline(shape.name, constants: constants)
            let queries = 16 / shape.heads
            try dispatch(commandBuffer, pipeline, q: q, k: k, v: v, out: out, params: params,
                         threads: shape.threads, headsPerGroup: shape.heads,
                         queryGroups: (Int(params.queryCount) + queries - 1) / queries)
            return
        }
        if [256, 512].contains(params.headDim) {
            let pipeline = try context.pipeline("attention_prefill_causal_simd",
                constants: [MetalFunctionConstant(index: 76, value: .uint32(kvRingCapacity)),
                            MetalFunctionConstant(index: 77, value: .uint32(params.headDim))])
            try dispatch(commandBuffer, pipeline, q: q, k: k, v: v, out: out, params: params,
                         threads: 32, headsPerGroup: 1)
        } else {
            try encodeCausalTiled(commandBuffer: commandBuffer, q: q, k: k, v: v, out: out, params: params,
                                  kvRingCapacity: kvRingCapacity, layerKind: layerKind)
        }
    }

    /// The grouped-query tensor kernel for this shape: head size 256 with two
    /// query heads a KV head, or 512 with eight; none with a bidirectional
    /// block.
    private static func gqaTensorShape(_ p: PrefillAttentionParams) -> (name: String, heads: Int, threads: Int)? {
        guard p.bidirectionalBlockStart == p.bidirectionalBlockEnd else { return nil }
        switch (p.headDim, p.numQHeads / p.numKVHeads) {
        case (256, 2): return ("attention_prefill_gqa_tensorops_d256_h2", 2, 128)
        case (512, 8): return ("attention_prefill_gqa_tensorops_d512_h8", 8, 256)
        default: return nil
        }
    }

    /// The tiled kernel, for any head size; the reference the others match.
    func encodeCausalTiled(commandBuffer: any MTLCommandBuffer,
                           q: any MTLBuffer,
                           k: any MTLBuffer,
                           v: any MTLBuffer,
                           out: any MTLBuffer,
                           params: PrefillAttentionParams,
                           kvRingCapacity: UInt32,
                           layerKind: PrefillAttentionLayerKind) throws {
        let params = Self.effective(params, layerKind: layerKind)
        let pipeline = kvRingCapacity > 0
            ? try context.pipeline("attention_prefill_causal_tiled",
                                   constants: [MetalFunctionConstant(index: 76, value: .uint32(kvRingCapacity))])
            : psoCausalTiled
        let threadWidth = max(1, pipeline.threadExecutionWidth)
        try dispatch(commandBuffer, pipeline, q: q, k: k, v: v, out: out, params: params,
                     threads: roundUp(max(threadWidth, Int(params.headDim)), toMultipleOf: threadWidth),
                     headsPerGroup: 1)
    }

    /// `params` validated, with the image block dropped on full-attention
    /// layers: only sliding-window layers make it bidirectional.
    private static func effective(_ params: PrefillAttentionParams,
                                  layerKind: PrefillAttentionLayerKind) -> PrefillAttentionParams {
        var params = params
        if layerKind == .full {
            params.bidirectionalBlockStart = 0
            params.bidirectionalBlockEnd = 0
        }
        validate(params)
        return params
    }

    private func dispatch(_ commandBuffer: any MTLCommandBuffer, _ pipeline: any MTLComputePipelineState,
                          q: any MTLBuffer, k: any MTLBuffer, v: any MTLBuffer, out: any MTLBuffer,
                          params: PrefillAttentionParams, threads: Int, headsPerGroup: Int,
                          queryGroups: Int? = nil) throws {
        precondition(threads <= pipeline.maxTotalThreadsPerThreadgroup,
                     "prefill attention needs \(threads) threads per threadgroup")
        let enc = try commandBuffer.makeTracedComputeEncoder()
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(q, offset: 0, index: 0)
        enc.setBuffer(k, offset: 0, index: 1)
        enc.setBuffer(v, offset: 0, index: 2)
        enc.setBuffer(out, offset: 0, index: 3)
        enc.setValue(params, index: 4)
        enc.dispatchThreadgroups(
            MTLSize(width: queryGroups ?? Int(params.queryCount), height: Int(params.numQHeads) / headsPerGroup, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }

    private static func validate(_ params: PrefillAttentionParams) {
        precondition(params.headDim > 0, "headDim must be positive")
        precondition(params.queryCount > 0, "queryCount must be positive")
        precondition(params.numQHeads > 0, "numQHeads must be positive")
        precondition(params.numKVHeads > 0, "numKVHeads must be positive")
        precondition(params.numQHeads % params.numKVHeads == 0,
                     "numQHeads must be divisible by numKVHeads")
        precondition(params.qTokenStrideElements >= params.numQHeads * params.headDim,
                     "q token stride is too small")
        precondition(params.oTokenStrideElements >= params.numQHeads * params.headDim,
                     "output token stride is too small")
        precondition(params.kvTokenStrideElements >= params.numKVHeads * params.headDim,
                     "KV token stride is too small")
        precondition(params.startPosition + params.queryCount <= params.kvValidCount,
                     "kvValidCount must include all in-flight query rows")
        precondition(params.bidirectionalBlockStart <= params.bidirectionalBlockEnd,
                     "bidirectional block range is invalid")
        precondition(params.bidirectionalBlockEnd <= params.kvValidCount,
                     "bidirectional block exceeds valid KV rows")
    }

    private func roundUp(_ value: Int, toMultipleOf multiple: Int) -> Int {
        ((value + multiple - 1) / multiple) * multiple
    }
}
