import Foundation
import Metal

/// Gemma 4 real-forward decode pass.
///
/// Composes the production kernels against the `.gturbo` model:
///
///   embed_lookup_int4(token) * sqrt(H)
///   for L in 0..<30:
///     a = rmsnorm_bf16w(h, input_layernorm)
///     Q = q_proj(a)    K = k_proj(a)    V = (SWA) v_proj(a) | (full) k_proj(a)
///     per-head q/k_norm (bf16w), per-head v_norm (no_scale)
///     NeoX RoPE on Q + K (default for SWA, proportional for full)
///     write K and V into separate cache slots
///     attn = attention(scale=1.0, SWA window or full causal)
///     attn = o_proj(attn)
///     h = h + rmsnorm_bf16w(attn, post_attention_layernorm)
///     h1 = rmsnorm_bf16w(h, pre_feedforward_layernorm)
///     h1 = SharedExpertInt8(h1)
///     h1 = rmsnorm_bf16w(h1, post_feedforward_layernorm_1)
///     // router + routed branch
///     xr   = rmsnorm_no_scale(h)
///     idx, w = router_topk_gemma4(xr, effective_scale[L], per_expert_scale[L])
///     h2 = rmsnorm_bf16w(h, pre_feedforward_layernorm_2)
///     h2 = moe_fused_ffn_streamed_routed(h2, residual=0, routedBlobs=fetch(idx), w)
///     h2 = rmsnorm_bf16w(h2, post_feedforward_layernorm_2)
///     h = h + rmsnorm_bf16w(h1 + h2, post_feedforward_layernorm)
///     h = h * layer_scalar[L]
///   logits = DequantInt4GEMV(rmsnorm_bf16w(h, model.norm), embed_table^T)
///   // final softcap and softmax happen in the Sampler.
///
/// Direct against `Model`; this is the only production decode forward path.
internal enum PrefillProjectionFamily: Sendable, Equatable {
    case q
    case kv
    case o
}

internal enum PrefillProjectionDispatch: Sendable, Equatable {
    case repeatedGEMV
    case qmm
}

internal enum PrefillProjectionDispatchPolicy {
    static func selectedDispatch(for family: PrefillProjectionFamily,
                                 chunkTokens: Int) -> PrefillProjectionDispatch {
        guard chunkTokens >= 32 else {
            return .repeatedGEMV
        }
        switch family {
        case .q:
            return .repeatedGEMV
        case .kv, .o:
            return .qmm
        }
    }
}

public final class RealForwardRunner: ChunkedPrefillRunner, MultimodalPrefillRunner, ContextWindowReporting, @unchecked Sendable {
    private struct LayerSharedExpertProjections {
        let gate: SharedExpertInt8Proj
        let up: SharedExpertInt8Proj
        let down: SharedExpertInt8Proj
        let postF1: TensorView
    }

    private let model: Model
    private let ctx: MetalContext
    private let kv: KVCacheManager
    private let cfg: ArchConfig

    // Kernels
    private let embedInt4: EmbedLookupInt4
    private let rms: RMSNorm
    private let int4: DequantInt4GEMV
    private let attention: Attention
    private let shared: SharedExpertRuntime
    private let moe: MoE
    private let fusionHead: LMHeadChainInt4
    private let fusedQKVGEMV: FusedQKVGEMV
    private let fusedQKVEpilogue: FusedQKVEpilogue
    private let fusedPostAttentionSetup: FusedPostAttentionSetup
    private let fusedTail: FusedLayerTail

    // Prefill kernels. These are initialized once per runner so the chunk path
    // cannot accidentally rebuild PSOs inside a per-layer loop.
    private let prefillEmbed: PrefillEmbedLookupInt4
    private let prefillRMS: PrefillRMSNorm
    private let prefillQMM: PrefillInt4QMM
    private let prefillMPPAffineInt4: MPPPrefillInt4QMM?
    private let prefillQKVEpilogue: PrefillQKVEpilogue
    private let prefillAttention: PrefillAttention
    private let prefillPostAttention: PrefillPostAttentionSetup
    private let prefillRouter: PrefillRouter
    private let prefillSharedExpert: PrefillSharedExpert
    private let prefillGroupedMoE: PrefillGroupedRoutedMoE
    private let prefillMoE: PrefillMoE
    private let prefillLayerTail: PrefillLayerTail
    private let prefillFinalRowHead: PrefillFinalRowHeadInt4

    // Scratch — preallocated per spec'd D / F / vocab.
    private let hidden: MTLBuffer        // [D] FP16
    private let normed: MTLBuffer        // [D] FP16
    private let attnOut: MTLBuffer       // [N_HEADS * head_dim] FP16
    private let qScratch: MTLBuffer      // [N_HEADS * head_dim] FP16
    private let oOut: MTLBuffer          // [D] FP16
    private let h1Buf: MTLBuffer         // [D] FP16 (dense MLP output)
    private let h2Buf: MTLBuffer         // [D] FP16 (routed output)
    private let routedX: MTLBuffer       // [D] FP16 (pre_feedforward_layernorm_2 output)
    private let denseX: MTLBuffer        // [D] FP16 (pre_feedforward_layernorm output)
    private let denseScratchGate: MTLBuffer // [F=2112] FP16
    private let denseScratchUp: MTLBuffer   // [F=2112] FP16
    private let denseScratchAct: MTLBuffer  // [F=2112] FP16
    private let routerInput: MTLBuffer   // [D] FP16 (rmsnorm_no_scale(h))
    private let zeroResidual: MTLBuffer  // [D] FP16 zeros — for routed branch base
    private let outIndices: MTLBuffer    // [topK] UInt32
    private let outWeights: MTLBuffer    // [topK] FP16
    // Persistent MoE scratch, allocated once; about 56 KiB at production shape.
    private let moeActs: MTLBuffer       // [topK * FmoE] FP16
    private let moeHitActiveSlots: MTLBuffer // [topK] UInt32
    private let moeMissActiveSlots: MTLBuffer // [topK] UInt32
    private let greedyTokenBuf: MTLBuffer // 4 B UInt32 fused-head output
    private var prefillChunkState = PrefillChunkCommitState()
    private var prefillScratch: PrefillChunkScratchBuffers?
    /// Owns the ANE guest; the forward loop asks it per layer whether the
    /// guest runs routed prefill.
    let supervisor: DeviceSupervisor
    public var aneRoutedPrefillProfile: [String: Any] { supervisor.routedPrefillProfile }
    public var supervisorProfile: [String: Any] { supervisor.profile }

    private static let prefillRoutedTileSchedulerConfig = PrefillRoutedTileSchedulerConfig()

    /// Per-layer `router.scale * D^-0.5` pre-folded into one BF16 buffer
    /// allocation per layer. ~168 KB total at 30 layers × 2816 BF16 — bounded
    /// host work done once at init.
    private let effectiveScaleBuffers: [MTLBuffer]
    private let sharedExpertProjections: [LayerSharedExpertProjections]

    public let maxContext: Int

    /// Per-instance head and RDADVISE modes. The fused head (default) skips the
    /// 512 KB logits write and leaves a greedy argmax in `lastGreedyToken`;
    /// callers that sample from the logits buffer (non-greedy configs) must pass
    /// `forceLogitsHead: true` or they read a never-written buffer.
    private let useFusedGreedyHead: Bool
    private let prefillAttentionPath: RuntimePrefillAttentionPath
    private let activeRoutedExperts: Int
    public let prefillExpertUnionCap: Int?
    /// Set to record selected layer outputs and every layer's routes during
    /// chunked prefill.
    public var prefillCapture: PrefillCapture?
    /// Set only for the duration of `verify`: greedy tokens for every row,
    /// tensor-op projections at any row count, routed experts on Metal.
    private var verifyTokens: MTLBuffer?
    public convenience init(model: Model, context: MetalContext, maxContext: Int,
                runtimeConfiguration: RuntimeConfiguration = .production) throws {
        try self.init(model: model, context: context, maxContext: maxContext,
                      runtimeConfiguration: runtimeConfiguration, supervisor: nil)
    }

    /// `supervisor` replaces placement resolved from the configuration and
    /// environment. Tests inject guests through it.
    init(model: Model, context: MetalContext, maxContext: Int,
         runtimeConfiguration: RuntimeConfiguration,
         supervisor: DeviceSupervisor?) throws {
        self.model = model
        self.ctx = context
        self.cfg = model.config
        self.maxContext = maxContext
        self.useFusedGreedyHead = runtimeConfiguration.headPath == .fusedRows
        self.prefillAttentionPath = runtimeConfiguration.prefillAttentionPath
        self.activeRoutedExperts = runtimeConfiguration.activeRoutedExperts
        self.prefillExpertUnionCap = runtimeConfiguration.prefillExpertUnionCap
        let useFP16Ring = runtimeConfiguration.fp16RingEnabled
        self.supervisor = try supervisor ?? DeviceSupervisor(
            configuration: runtimeConfiguration, model: model, context: context)
        self.kv = try KVCacheManager(device: context.device,
                                     config: cfg,
                                     maxContext: maxContext,
                                     fp16RingEnabled: useFP16Ring,
                                     slidingWindow: cfg.slidingWindow,
                                     maxPrefillChunkTokens: max(
                                        runtimeConfiguration.prefillChunkTokens,
                                        VisionConfig().maximumPooledTokens))

        self.embedInt4 = try EmbedLookupInt4(context: context)
        self.rms       = try RMSNorm(context: context)
        self.int4      = try DequantInt4GEMV(context: context)
        self.attention = try Attention(context: context)
        self.shared    = try SharedExpertRuntime(context: context,
                                                  weightBits: model.sharedExpertWeightBits)
        self.moe       = try MoE(context: context)
        self.fusionHead = try LMHeadChainInt4(context: context,
                                              maxD: cfg.hiddenSize,
                                              maxVocab: cfg.vocabSize)
        self.fusedQKVGEMV = try FusedQKVGEMV(context: context)
        self.fusedQKVEpilogue = try FusedQKVEpilogue(context: context)
        self.fusedPostAttentionSetup = try FusedPostAttentionSetup(context: context)
        self.fusedTail = try FusedLayerTail(context: context)
        self.prefillEmbed = try PrefillEmbedLookupInt4(context: context)
        self.prefillRMS = try PrefillRMSNorm(context: context)
        self.prefillQMM = try PrefillInt4QMM(context: context)
        self.prefillMPPAffineInt4 = MPPPrefillInt4QMM(context: context)
        self.prefillQKVEpilogue = try PrefillQKVEpilogue(context: context)
        self.prefillAttention = try PrefillAttention(context: context)
        self.prefillPostAttention = try PrefillPostAttentionSetup(context: context)
        self.prefillRouter = try PrefillRouter(context: context)
        self.prefillSharedExpert = try PrefillSharedExpert(
            context: context,
            weightBits: model.sharedExpertWeightBits,
            batched: true)
        self.prefillGroupedMoE = try PrefillGroupedRoutedMoE(context: context)
        self.prefillMoE = try PrefillMoE(context: context)
        self.prefillLayerTail = try PrefillLayerTail(context: context)
        self.prefillFinalRowHead = try PrefillFinalRowHeadInt4(context: context,
                                                               maxD: cfg.hiddenSize)

        let device = context.device
        let D = cfg.hiddenSize
        let F = cfg.intermediateSize
        let maxQ = cfg.numHeads * max(cfg.headDim, cfg.fullHeadDim)

        func buf(_ count: Int, _ stride: Int = MemoryLayout<Float16>.size) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: max(count, 1) * stride,
                                            options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            return b
        }
        self.hidden        = try buf(D)
        self.normed        = try buf(D)
        self.attnOut       = try buf(maxQ)
        self.qScratch      = try buf(maxQ)
        self.oOut          = try buf(D)
        self.h1Buf         = try buf(D)
        self.h2Buf         = try buf(D)
        self.routedX       = try buf(D)
        self.denseX        = try buf(D)
        self.denseScratchGate = try buf(F)
        self.denseScratchUp   = try buf(F)
        self.denseScratchAct  = try buf(F)
        self.routerInput   = try buf(D)
        self.zeroResidual  = try buf(D)
        // The routed MoE kernel seeds y[d] = residual[d]; pinning this buffer
        // to zero once at init makes the routed branch's residual contribution
        // exactly zero (it's combined with the dense MLP downstream).
        memset(self.zeroResidual.contents(), 0, self.zeroResidual.length)
        self.outIndices    = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size)
        self.outWeights    = try buf(cfg.topKExperts)
        self.moeActs       = try buf(cfg.topKExperts * cfg.moeIntermediateSize)
        self.moeHitActiveSlots = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size)
        self.moeMissActiveSlots = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size)
        guard let tok = device.makeBuffer(length: MemoryLayout<UInt32>.size,
                                          options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        self.greedyTokenBuf = tok

        func sharedProj(_ view: TensorView, rows: UInt32, cols: UInt32) -> SharedExpertProjection {
            SharedExpertProjection(weights: view.buffer,
                                 scales: view.buffer,
                                 biases: view.buffer,
                                 weightsOffset: Int(view.offset),
                                 scalesOffset: Int(view.scaleOffset),
                                 biasesOffset: Int(view.biasOffset),
                                 rows: rows,
                                 cols: cols)
        }
        var sharedViews: [LayerSharedExpertProjections] = []
        sharedViews.reserveCapacity(cfg.numLayers)
        for L in 0..<cfg.numLayers {
            let gate = try model.sharedExpertGate(layer: L)
            let up = try model.sharedExpertUp(layer: L)
            let down = try model.sharedExpertDown(layer: L)
            sharedViews.append(LayerSharedExpertProjections(
                gate: sharedProj(gate, rows: UInt32(F), cols: UInt32(D)),
                up: sharedProj(up, rows: UInt32(F), cols: UInt32(D)),
                down: sharedProj(down, rows: UInt32(D), cols: UInt32(F)),
                postF1: try model.postFFN1(layer: L)))
        }
        self.sharedExpertProjections = sharedViews

        // Pre-fold 1/sqrt(D) into router.scale per layer. Each layer gets its
        // own BF16 [D] buffer — the kernel reads `effective_scale[i]` and we
        // pay for the multiply once per generation, not per token.
        var perLayer: [MTLBuffer] = []
        perLayer.reserveCapacity(cfg.numLayers)
        let invSqrtD = Float(1.0) / Float(D).squareRoot()
        let dInts = D
        for L in 0..<cfg.numLayers {
            let scaleView = try model.routerScale(layer: L)
            guard let buf = device.makeBuffer(length: dInts * MemoryLayout<UInt16>.size,
                                              options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            let src = scaleView.buffer.contents()
                .advanced(by: Int(scaleView.offset))
                .assumingMemoryBound(to: UInt16.self)
            let dst = buf.contents().assumingMemoryBound(to: UInt16.self)
            for i in 0..<dInts {
                let v = Quantization.bf16ToFloat(src[i]) * invSqrtD
                dst[i] = Quantization.bf16Bits(v)
            }
            buf.label = "effective_scale.L\(L)"
            perLayer.append(buf)
        }
        self.effectiveScaleBuffers = perLayer
    }

    public func reset() {
        kv.reset()
        resetTransientState()
    }

    /// Maximum rows one `verify` call accepts.
    public static let maxVerifyTokens = LMHeadChainInt4.maxGreedyRows

    /// Runs `tokens` at the KV cursor as one chunk and returns the target's
    /// greedy next token after each row. The cursor advances by
    /// `tokens.count`; the caller rewinds rows it rejects. `prefillCapture`,
    /// when set, records the chunk as in prefill.
    public func verify(tokens: [Int32], startPosition: Int) async throws -> [Int32] {
        try prefillChunkState.requireClean(operation: "verify")
        guard !tokens.isEmpty, tokens.count <= Self.maxVerifyTokens else {
            throw PrefillError.chunkedUnsupported(
                "verify takes 1...\(Self.maxVerifyTokens) tokens, got \(tokens.count)")
        }
        let scratch = try prefillScratch
            ?? ensurePrefillScratch(config: .production(chunkTokens: 32))
        guard let out = ctx.device.makeBuffer(length: tokens.count * MemoryLayout<UInt32>.stride,
                                              options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        verifyTokens = out
        defer { verifyTokens = nil }
        try await executePrefillChunk(
            tokens: tokens[...], startPosition: startPosition, outputMode: .greedyIfAvailable,
            logits: out, scratch: scratch,
            config: .production(chunkTokens: scratch.layout.chunkTokens),
            writeFinalHead: true)
        let values = out.contents().bindMemory(to: UInt32.self, capacity: tokens.count)
        return (0..<tokens.count).map { Int32(bitPattern: values[$0]) }
    }

    /// Abandons the newest KV tokens, e.g. rejected speculative drafts. Rows
    /// past the new cursor are never read and are overwritten on the next write.
    public func rewind(to position: Int) throws {
        guard position > 0, position <= kv.position,
              kv.highWaterPosition - position <= kv.maxRewindTokens else {
            throw PrefillError.prefillCursorMismatch(
                "cannot rewind KV from \(kv.position) to \(position); "
                    + "high water \(kv.highWaterPosition), "
                    + "ring slack \(kv.maxRewindTokens)")
        }
        kv.rewind(to: position)
        resetTransientState()
    }

    private func resetTransientState() {
        prefillChunkState.reset()
    }

    public private(set) var totalIoNanos: UInt64 = 0
    public private(set) var prefillPreRouterGPUNanos: UInt64 = 0
    public private(set) var prefillSharedGPUNanos: UInt64 = 0
    public private(set) var prefillRoutedGPUNanos: UInt64 = 0
    public private(set) var totalCb1Nanos: UInt64 = 0
    public private(set) var totalCb2Nanos: UInt64 = 0
    public private(set) var totalHeadNanos: UInt64 = 0
    public private(set) var totalHeadFusedNanos: UInt64 = 0
    public private(set) var lastGreedyToken: UInt32 = 0
    public var usesFusedGreedyHead: Bool { useFusedGreedyHead }
    public private(set) var routedExpertCacheHits: UInt64 = 0
    public private(set) var routedExpertCacheMisses: UInt64 = 0
    public private(set) var routedWeightMassByRank = Array(repeating: 0.0, count: 8)
    public private(set) var routedWeightMassSamples: UInt64 = 0
    public private(set) var prefillUnionExpertsBeforeCap: UInt64 = 0
    public private(set) var prefillUnionExpertsAfterCap: UInt64 = 0
    public private(set) var prefillUnionLayerTiles: UInt64 = 0
    public private(set) var prefillRoutedWallNanos: UInt64 = 0
    /// Host time awaiting routed tile fetches and tile drains in chunked
    /// prefill, and the tiles' planned cache hits and misses.
    public private(set) var prefillTileFetchNanos: UInt64 = 0
    public private(set) var prefillTileDrainNanos: UInt64 = 0
    public private(set) var prefillTileHits: UInt64 = 0
    public private(set) var prefillTileMisses: UInt64 = 0
    public private(set) var exactRouteDigest: UInt64 = 1_469_598_103_934_665_603
    public private(set) var exactRouteSamples: UInt64 = 0
    public private(set) var allRouteValuesFinite: Bool = true

    @inline(__always)
    private func recordExactRoute(expert: UInt32, weight: Float16) {
        exactRouteDigest = (exactRouteDigest ^ UInt64(expert)) &* 1_099_511_628_211
        exactRouteDigest = (exactRouteDigest ^ UInt64(weight.bitPattern)) &* 1_099_511_628_211
        exactRouteSamples &+= 1
        if !weight.isFinite { allRouteValuesFinite = false }
    }

    private func snapshotFP16(_ buffer: MTLBuffer, count: Int) -> [Float16] {
        let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    public func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        try prefillChunkState.requireClean(operation: "produce")
        try await produceToken(token: token,
                               position: position,
                               into: logits,
                               outputMode: .greedyIfAvailable)
    }

    public func prefillChunked(tokens: ArraySlice<Int32>,
                               startPosition: Int,
                               outputMode: PrefillOutputMode,
                               config: PrefillRuntimeConfig,
                               into logits: MTLBuffer,
                               onProgress: (Int) -> Void) async throws -> PrefillResult {
        try prefillChunkState.requireClean(operation: "prefillChunked")
        guard config.mode == .chunked else {
            throw PrefillError.chunkedUnsupported(
                "prefillChunked requires PrefillRuntimeConfig.mode == .chunked")
        }
        guard startPosition >= 0 else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill startPosition must be non-negative")
        }
        let kvPosition = kv.position
        guard kvPosition == startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill cursor \(kvPosition) != startPosition \(startPosition)")
        }
        guard tokens.count <= maxContext - startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill range starting at \(startPosition) with \(tokens.count) tokens exceeds maxContext \(maxContext)")
        }
        guard !tokens.isEmpty else {
            return PrefillResult(newPosition: startPosition, seed: .logitsWritten)
        }

        let scratch = try ensurePrefillScratch(config: config)
        let spans = PrefillChunkPlanner.spans(tokenCount: tokens.count,
                                              startPosition: startPosition,
                                              config: config)
        try await PrefillSpanIteration.forEachSpan(spans) { spanIndex, span in
            let lower = tokens.index(tokens.startIndex, offsetBy: span.tokenOffset)
            let upper = tokens.index(lower, offsetBy: span.tokenCount)
            try await executePrefillChunk(
                tokens: tokens[lower..<upper],
                startPosition: span.startPosition,
                outputMode: outputMode,
                logits: logits,
                scratch: scratch,
                config: config,
                writeFinalHead: spanIndex == spans.count - 1)
            onProgress(span.completedCount)
        }
        if outputMode == .greedyIfAvailable, useFusedGreedyHead {
            return PrefillResult(newPosition: startPosition + tokens.count,
                                 seed: .greedyToken(lastGreedyToken))
        }
        return PrefillResult(newPosition: startPosition + tokens.count,
                             seed: .logitsWritten)
    }

    public func prefillMultimodal(input: MultimodalPrefillInput,
                                  startPosition: Int,
                                  outputMode: PrefillOutputMode,
                                  config: PrefillRuntimeConfig,
                                  into logits: MTLBuffer,
                                  onProgress: (Int) -> Void) async throws -> PrefillResult {
        try prefillChunkState.requireClean(operation: "prefillMultimodal")
        // The same property `coercedForImagePrompt()` repairs, so the guard and
        // the coercion cannot drift apart.
        guard config.servesImagePrompt else {
            throw PrefillError.chunkedUnsupported(
                "multimodal prefill requires the complete chunked prefill path")
        }
        let tokens = input.embeddingTokenIDs
        guard startPosition >= 0,
              startPosition + tokens.count <= maxContext else {
            throw PrefillError.chunkedUnsupported(
                "multimodal prefill range exceeds maxContext \(maxContext)")
        }
        let kvPosition = kv.position
        guard kvPosition == startPosition else {
            throw PrefillError.chunkedUnsupported(
                "multimodal prefill cursor \(kvPosition) != startPosition \(startPosition)")
        }
        guard !tokens.isEmpty else {
            return PrefillResult(newPosition: startPosition, seed: .logitsWritten)
        }

        // The planner applies the same clamp the scratch layout does. Cutting at
        // the raw `config.chunkTokens` let a caller size a 280-token scratch and
        // then hand it a larger chunk, which died on the guard.
        let textChunkTokens = min(config.chunkTokens, PrefillRuntimeConfig.maxChunkTokens)
        let work = PrefillChunkPlanner.multimodalWork(
            tokenCount: tokens.count,
            imageRanges: input.imageSpans.map(\.tokenRange),
            chunkTokens: config.chunkTokens)
        let imageBuffers = input.imageSpans.map(\.features.buffer)

        // One layout for the whole turn, sized for the largest item in it.
        // Keying the single-slot cache on each item's exact chunk size made a
        // text-image-text prompt free and reallocate the entire chunk scratch at
        // every boundary, twice per image span, every turn.
        let pooledTokens = VisionConfig().maximumPooledTokens
        let hasImages = work.contains(where: \.isImage)
        let layoutTokens = hasImages
            ? max(config.chunkTokens, pooledTokens)
            : config.chunkTokens
        let layoutLimit = hasImages
            ? max(PrefillRuntimeConfig.maxChunkTokens, pooledTokens)
            : PrefillRuntimeConfig.maxChunkTokens
        let scratch = try ensurePrefillScratch(
            config: config.replacingChunkTokens(layoutTokens),
            chunkTokenLimit: layoutLimit)

        var completed = 0
        for (index, item) in work.enumerated() {
            try Task.checkCancellation()
            let isImage = item.isImage
            let imageFeatures = item.imageIndex.map { imageBuffers[$0] }
            // Every item executes under the chunk size it was planned at.
            let chunkConfig = config.replacingChunkTokens(
                isImage ? item.range.count : textChunkTokens)
            let lower = tokens.index(tokens.startIndex, offsetBy: item.range.lowerBound)
            let upper = tokens.index(tokens.startIndex, offsetBy: item.range.upperBound)
            try await executePrefillChunk(
                tokens: tokens[lower..<upper],
                startPosition: startPosition + item.range.lowerBound,
                outputMode: outputMode,
                logits: logits,
                scratch: scratch,
                config: chunkConfig,
                writeFinalHead: index == work.count - 1,
                embeddingOverride: imageFeatures,
                bidirectionalBlock: isImage
                    ? (startPosition + item.range.lowerBound)..<(startPosition + item.range.upperBound)
                    : nil)
            completed += item.range.count
            onProgress(completed)
        }
        if outputMode == .greedyIfAvailable, useFusedGreedyHead {
            return PrefillResult(newPosition: startPosition + tokens.count,
                                 seed: .greedyToken(lastGreedyToken))
        }
        return PrefillResult(newPosition: startPosition + tokens.count,
                             seed: .logitsWritten)
    }

    @discardableResult
    private func ensurePrefillScratch(
        config: PrefillRuntimeConfig,
        chunkTokenLimit: Int = PrefillRuntimeConfig.maxChunkTokens
    ) throws -> PrefillChunkScratchBuffers {
        let layout = PrefillChunkScratchLayout(config: cfg,
                                               chunkTokens: config.chunkTokens,
                                               chunkTokenLimit: chunkTokenLimit)
        if let scratch = prefillScratch, scratch.layout == layout {
            return scratch
        }
        let scratch = try PrefillChunkScratchBuffers.allocate(device: ctx.device, layout: layout)
        prefillScratch = scratch
        return scratch
    }

    private func executePrefillChunk(tokens: ArraySlice<Int32>,
                                     startPosition: Int,
                                     outputMode: PrefillOutputMode,
                                     logits: MTLBuffer,
                                     scratch: PrefillChunkScratchBuffers,
                                     config: PrefillRuntimeConfig,
                                     writeFinalHead: Bool,
                                     embeddingOverride: MTLBuffer? = nil,
                                     bidirectionalBlock: Range<Int>? = nil) async throws {
        guard !tokens.isEmpty else { return }
        let kvPosition = kv.position
        guard kvPosition == startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill cursor \(kvPosition) != startPosition \(startPosition)")
        }
        guard startPosition >= 0, startPosition + tokens.count <= maxContext else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill range [\(startPosition), \(startPosition + tokens.count)) exceeds maxContext \(maxContext)")
        }
        guard tokens.count <= scratch.layout.chunkTokens else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill token count \(tokens.count) exceeds scratch chunk size \(scratch.layout.chunkTokens)")
        }
        if kv.fp16RingEnabled, let ringLayer = (0..<cfg.numLayers).first(where: {
            kv.ringCapacity(layer: $0) > 0
        }) {
            let requiredCapacity = min(maxContext, cfg.slidingWindow + config.chunkTokens)
            let ringCapacity = kv.ringCapacity(layer: ringLayer)
            guard requiredCapacity <= ringCapacity else {
                throw PrefillError.chunkedUnsupported(
                    "FP16 KV ring capacity \(ringCapacity) cannot hold required capacity \(requiredCapacity) for maxContext \(maxContext), slidingWindow \(cfg.slidingWindow), and prefillChunkTokens \(config.chunkTokens)")
            }
        }

        struct LayerPrefillQKVViews {
            let inputNorm: TensorView
            let q: TensorView
            let k: TensorView
            let v: TensorView
            let o: TensorView
            let postAttention: TensorView
            let preFFN: TensorView
            let preFFN2: TensorView
            let postFFN2: TensorView
            let postFFN: TensorView
            let layerScalar: TensorView
            let qNorm: TensorView
            let kNorm: TensorView
            let router: TensorView
            let routerPerExpertScale: TensorView
        }

        let layerViews = try (0..<cfg.numLayers).map { L in
            let isFull = cfg.fullAttentionLayerMask[L] != 0
            return LayerPrefillQKVViews(
                inputNorm: try model.inputNorm(layer: L),
                q: try model.qProj(layer: L),
                k: try model.kProj(layer: L),
                v: isFull ? (try model.kProj(layer: L)) : (try model.vProj(layer: L)),
                o: try model.oProj(layer: L),
                postAttention: try model.postAttnNorm(layer: L),
                preFFN: try model.preFFN(layer: L),
                preFFN2: try model.preFFN2(layer: L),
                postFFN2: try model.postFFN2(layer: L),
                postFFN: try model.postFFN(layer: L),
                layerScalar: try model.layerScalar(layer: L),
                qNorm: try model.qNorm(layer: L),
                kNorm: try model.kNorm(layer: L),
                router: try model.router(layer: L),
                routerPerExpertScale: try model.routerPerExpertScale(layer: L))
        }

        let tokenIDs = tokens.map { UInt32(bitPattern: $0) }
        guard let tokenBuffer = ctx.device.makeBuffer(bytes: tokenIDs,
                                                      length: tokenIDs.count * MemoryLayout<UInt32>.stride,
                                                      options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        let D = cfg.hiddenSize
        let eps: Float = 1e-6
        let sqrtHidden = Float(D).squareRoot()
        let t = tokens.count
        let emb = model.embedding

        func encodeInt4Projection(commandBuffer: MTLCommandBuffer,
                                  family: PrefillProjectionFamily,
                                  weights: TensorView,
                                  x: MTLBuffer,
                                  y: MTLBuffer,
                                  rows: Int,
                                  columns: Int,
                                  tokenCount: Int,
                                  xStrideElements: Int,
                                  yStrideElements: Int) {
            if verifyTokens != nil, tokenCount <= DequantInt4GEMV.maxBatchRows {
                int4.encodeRows(commandBuffer: commandBuffer,
                                weights: weights.buffer, weightsOffset: Int(weights.offset),
                                scales: weights.buffer, scalesOffset: Int(weights.scaleOffset),
                                biases: weights.buffer, biasesOffset: Int(weights.biasOffset),
                                x: x, y: y, m: UInt32(rows), n: UInt32(columns), rows: tokenCount,
                                xStrideElements: xStrideElements, yStrideElements: yStrideElements)
                return
            }
            if tokenCount >= 32,
               family == .q || family == .kv || family == .o,
               let candidate = prefillMPPAffineInt4 {
                let metadata = candidate.encode(
                    commandBuffer: commandBuffer,
                    weights: weights.buffer,
                    weightsOffset: Int(weights.offset),
                    scales: weights.buffer,
                    scalesOffset: Int(weights.scaleOffset),
                    biases: weights.buffer,
                    biasesOffset: Int(weights.biasOffset),
                    x: x,
                    y: y,
                    m: tokenCount,
                    n: rows,
                    k: columns)
                if metadata.path == .affineThreadgroupF16 {
                    return
                }
            }
            if PrefillProjectionDispatchPolicy.selectedDispatch(for: family,
                                                                chunkTokens: tokenCount) == .qmm {
                prefillQMM.encode(commandBuffer: commandBuffer,
                                  weights: weights.buffer,
                                  weightsOffset: Int(weights.offset),
                                  scales: weights.buffer,
                                  scalesOffset: Int(weights.scaleOffset),
                                  biases: weights.buffer,
                                  biasesOffset: Int(weights.biasOffset),
                                  x: x,
                                  y: y,
                                  t: tokenCount,
                                  n: rows,
                                  k: columns)
                return
            }
            for row in 0..<tokenCount {
                int4.encode(commandBuffer: commandBuffer,
                            weights: weights.buffer,
                            weightsOffset: Int(weights.offset),
                            scales: weights.buffer,
                            scalesOffset: Int(weights.scaleOffset),
                            biases: weights.buffer,
                            biasesOffset: Int(weights.biasOffset),
                            x: x,
                            xOffset: row * xStrideElements * MemoryLayout<Float16>.stride,
                            y: y,
                            yOffset: row * yStrideElements * MemoryLayout<Float16>.stride,
                            m: UInt32(rows),
                            n: UInt32(columns))
            }
        }

        func copyPrefillKV(commandBuffer: MTLCommandBuffer,
                           source: MTLBuffer,
                           destination: (buffer: MTLBuffer, offset: Int, stride: Int),
                           sourceTokenOffset: Int,
                           tokenCount: Int,
                           bytesPerToken: Int) throws {
            guard tokenCount > 0 else { return }
            guard let blit = commandBuffer.makeBlitCommandEncoder() else {
                throw ModelError.residentBufferWrapFailed
            }
            blit.copy(from: source,
                      sourceOffset: sourceTokenOffset * bytesPerToken,
                      to: destination.buffer,
                      destinationOffset: destination.offset,
                      size: tokenCount * bytesPerToken)
            blit.endEncoding()
        }

        func copyPrefillKVToCache(commandBuffer: MTLCommandBuffer,
                                  kv: KVCacheManager,
                                  layer: Int,
                                  startPosition: Int,
                                  tokenCount: Int,
                                  keySource: MTLBuffer,
                                  valueSource: MTLBuffer,
                                  bytesPerToken: Int) throws {
            let capacity = kv.capacity(layer: layer)
            let physicalStart = startPosition % capacity
            let firstSpan = min(tokenCount, capacity - physicalStart)
            let keyFirst = kv.kRange(layer: layer, start: startPosition, count: firstSpan)
            let valueFirst = kv.vRange(layer: layer, start: startPosition, count: firstSpan)
            try copyPrefillKV(commandBuffer: commandBuffer,
                              source: keySource,
                              destination: keyFirst,
                              sourceTokenOffset: 0,
                              tokenCount: firstSpan,
                              bytesPerToken: bytesPerToken)
            try copyPrefillKV(commandBuffer: commandBuffer,
                              source: valueSource,
                              destination: valueFirst,
                              sourceTokenOffset: 0,
                              tokenCount: firstSpan,
                              bytesPerToken: bytesPerToken)
            guard firstSpan < tokenCount else { return }

            let secondCount = tokenCount - firstSpan
            let secondStart = startPosition + firstSpan
            let keySecond = kv.kRange(layer: layer, start: secondStart, count: secondCount)
            let valueSecond = kv.vRange(layer: layer, start: secondStart, count: secondCount)
            try copyPrefillKV(commandBuffer: commandBuffer,
                              source: keySource,
                              destination: keySecond,
                              sourceTokenOffset: firstSpan,
                              tokenCount: secondCount,
                              bytesPerToken: bytesPerToken)
            try copyPrefillKV(commandBuffer: commandBuffer,
                              source: valueSource,
                              destination: valueSecond,
                              sourceTokenOffset: firstSpan,
                              tokenCount: secondCount,
                              bytesPerToken: bytesPerToken)
        }

        prefillChunkState.markDirty(startPosition: startPosition, tokenCount: tokens.count)

        guard var cb = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        cb.label = "prefill start=\(startPosition) count=\(t) phase=embed"
        if let embeddingOverride {
            // The override spans the whole chunk — the planner emits image spans
            // as standalone chunks — so the INT4 gather it would cover is
            // skipped rather than encoded and fully overwritten.
            let bytes = t * D * MemoryLayout<Float16>.stride
            guard embeddingOverride.length >= bytes,
                  let blit = cb.makeBlitCommandEncoder() else {
                throw VisionRuntimeError.invalidInput(
                    "projected image feature buffer is too small")
            }
            blit.copy(from: embeddingOverride,
                      sourceOffset: 0,
                      to: scratch.hidden,
                      destinationOffset: 0,
                      size: bytes)
            blit.endEncoding()
        } else {
        prefillEmbed.encode(commandBuffer: cb,
                            table: emb.buffer,
                            tableOffset: Int(emb.offset),
                            scales: emb.buffer,
                            scalesOffset: Int(emb.scaleOffset),
                            biases: emb.buffer,
                            biasesOffset: Int(emb.biasOffset),
                            tokens: tokenBuffer,
                            out: scratch.hidden,
                            t: UInt32(t),
                            d: UInt32(D),
                            outScale: sqrtHidden)
        }

        let capture = prefillCapture
        try capture?.prepare(device: ctx.device, chunkTokens: scratch.layout.chunkTokens,
                             hiddenSize: D, numLayers: cfg.numLayers)

        for L in 0..<cfg.numLayers {
            model.beginOpeningRoutedExpertStreamer(layer: L)
            let views = layerViews[L]
            let isFull = cfg.fullAttentionLayerMask[L] != 0
            let headDim = isFull ? cfg.fullHeadDim : cfg.headDim
            let numKVHeads = isFull ? cfg.numFullKVHeads : cfg.numKVHeads
            let qDim = cfg.numHeads * headDim
            let kvDim = numKVHeads * headDim

            prefillRMS.encodeBF16W(commandBuffer: cb,
                                   x: scratch.hidden,
                                   weight: views.inputNorm.buffer,
                                   weightOffset: Int(views.inputNorm.offset),
                                   out: scratch.normed,
                                   t: UInt32(t),
                                   d: UInt32(D),
                                   eps: eps)
            encodeInt4Projection(commandBuffer: cb,
                                 family: .q,
                                 weights: views.q,
                                 x: scratch.normed,
                                 y: scratch.q,
                                 rows: qDim,
                                 columns: D,
                                 tokenCount: t,
                                 xStrideElements: D,
                                 yStrideElements: qDim)
            encodeInt4Projection(commandBuffer: cb,
                                 family: .kv,
                                 weights: views.k,
                                 x: scratch.normed,
                                 y: scratch.kStage,
                                 rows: kvDim,
                                 columns: D,
                                 tokenCount: t,
                                 xStrideElements: D,
                                 yStrideElements: kvDim)

            encodeInt4Projection(commandBuffer: cb,
                                 family: .kv,
                                 weights: views.v,
                                 x: scratch.normed,
                                 y: scratch.vStage,
                                 rows: kvDim,
                                 columns: D,
                                 tokenCount: t,
                                 xStrideElements: D,
                                 yStrideElements: kvDim)

            let rotatedPairs = isFull
                ? UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor / 2.0)
                : UInt32(headDim / 2)
            prefillQKVEpilogue.encode(commandBuffer: cb,
                                       q: scratch.q,
                                       k: scratch.kStage,
                                       v: scratch.vStage,
                                       qWeight: views.qNorm.buffer,
                                       qWeightOffset: Int(views.qNorm.offset),
                                       kWeight: views.kNorm.buffer,
                                       kWeightOffset: Int(views.kNorm.offset),
                                       startPosition: UInt32(startPosition),
                                       queryCount: UInt32(t),
                                       headDim: UInt32(headDim),
                                       numQHeads: UInt32(cfg.numHeads),
                                       numKVHeads: UInt32(numKVHeads),
                                       qTokenStrideElements: UInt32(qDim),
                                       kvTokenStrideElements: UInt32(kvDim),
                                       theta: isFull ? Float(cfg.fullRopeTheta) : Float(cfg.ropeTheta),
                                       rotatedPairs: rotatedPairs,
                                       eps: eps)

            let bytes = t * kvDim * MemoryLayout<Float16>.stride
            try copyPrefillKVToCache(commandBuffer: cb,
                                     kv: kv,
                                     layer: L,
                                     startPosition: startPosition,
                                     tokenCount: t,
                                     keySource: scratch.kStage,
                                     valueSource: scratch.vStage,
                                     bytesPerToken: bytes / t)
            let params = PrefillAttentionParams(
                    startPosition: UInt32(startPosition),
                    queryCount: UInt32(t),
                    headDim: UInt32(headDim),
                    numQHeads: UInt32(cfg.numHeads),
                    numKVHeads: UInt32(numKVHeads),
                    kvValidCount: UInt32(startPosition + t),
                    slidingWindow: isFull ? UInt32(startPosition + t) : UInt32(cfg.slidingWindow),
                    kvTokenStrideElements: UInt32(kvDim),
                    qTokenStrideElements: UInt32(qDim),
                    oTokenStrideElements: UInt32(qDim),
                    scale: 1.0,
                    bidirectionalBlockStart: UInt32(
                        isFull ? 0 : bidirectionalBlock?.lowerBound ?? 0),
                    bidirectionalBlockEnd: UInt32(
                        isFull ? 0 : bidirectionalBlock?.upperBound ?? 0))
            let keyBuffer = kv.keyBuffer(layer: L, validTokenCount: startPosition + t)
            let valueBuffer = kv.valueBuffer(layer: L, validTokenCount: startPosition + t)
            let ringCapacity = kv.ringCapacity(layer: L)
            let activeRingCapacity = ringCapacity > 0 && startPosition + t > ringCapacity
                ? UInt32(ringCapacity)
                : 0
            prefillAttention.encodeCausal(commandBuffer: cb,
                                          q: scratch.q,
                                          k: keyBuffer,
                                          v: valueBuffer,
                                          out: scratch.attentionOutput,
                                          params: params,
                                          kvRingCapacity: activeRingCapacity,
                                          layerKind: isFull ? .full : .slidingWindow,
                                          path: prefillAttentionPath)
            encodeInt4Projection(commandBuffer: cb,
                                     family: .o,
                                     weights: views.o,
                                     x: scratch.attentionOutput,
                                     y: scratch.h1,
                                     rows: D,
                                     columns: qDim,
                                     tokenCount: t,
                                     xStrideElements: qDim,
                                     yStrideElements: D)
            prefillPostAttention.encode(commandBuffer: cb,
                                            hidden: scratch.hidden,
                                            attn: scratch.h1,
                                            denseX: scratch.denseX,
                                            routedX: scratch.routedX,
                                            routerX: scratch.routerX,
                                            postAttentionWeight: views.postAttention.buffer,
                                            postAttentionWeightOffset: Int(views.postAttention.offset),
                                            preFFNWeight: views.preFFN.buffer,
                                            preFFNWeightOffset: Int(views.preFFN.offset),
                                            preFFN2Weight: views.preFFN2.buffer,
                                            preFFN2WeightOffset: Int(views.preFFN2.offset),
                                            queryCount: UInt32(t),
                                            d: UInt32(D),
                                            hiddenStrideElements: UInt32(D),
                                            attnStrideElements: UInt32(D),
                                            denseStrideElements: UInt32(D),
                                            routedStrideElements: UInt32(D),
                                            routerStrideElements: UInt32(D),
                                            eps: eps)
            prefillRouter.encodeGemma4Block(
                        commandBuffer: cb,
                        weights: views.router.buffer,
                        weightsOffset: Int(views.router.offset),
                        scales: views.router.buffer,
                        scalesOffset: Int(views.router.scaleOffset),
                        biases: views.router.buffer,
                        biasesOffset: Int(views.router.biasOffset),
                        hidden: scratch.routerX,
                        effectiveScale: effectiveScaleBuffers[L],
                        perExpertScale: views.routerPerExpertScale.buffer,
                        perExpertScaleOffset: Int(views.routerPerExpertScale.offset),
                        outIndices: scratch.routeIDs,
                        outWeights: scratch.routeWeights,
                        queryCount: UInt32(t),
                        numExperts: UInt32(cfg.numExperts),
                        d: UInt32(D),
                        topK: UInt32(cfg.topKExperts),
                        hiddenStrideElements: UInt32(D))

                    // Clear routed partials here, in a command buffer the host
                    // completes, so routed guests on other queues never race it.
                    // Only this chunk's `t * topK` rows are read.
                    if let blit = cb.makeBlitCommandEncoder() {
                        blit.fill(buffer: scratch.routePartials,
                                  range: 0..<(t * cfg.topKExperts * D * MemoryLayout<Float16>.stride),
                                  value: 0)
                        blit.endEncoding()
                    }
                    cb.commit()
                    try waitForCompletion(cb)
                    prefillPreRouterGPUNanos += UInt64(max(0, cb.gpuEndTime - cb.gpuStartTime) * 1e9)

                    let routeCount = t * cfg.topKExperts
                    let idPtr = scratch.routeIDs.contents()
                        .bindMemory(to: UInt32.self, capacity: routeCount)
                    let weightPtr = scratch.routeWeights.contents()
                        .bindMemory(to: Float16.self, capacity: routeCount)
                    var routeIDs = [UInt32]()
                    routeIDs.reserveCapacity(routeCount)
                    var routeWeights = [Float16]()
                    routeWeights.reserveCapacity(routeCount)
                    for i in 0..<routeCount {
                        routeIDs.append(min(idPtr[i], UInt32(cfg.numExperts - 1)))
                        routeWeights.append(weightPtr[i])
                        recordExactRoute(expert: routeIDs[i], weight: routeWeights[i])
                    }
                    capture?.routes[L] = routeIDs
                    let uncappedUnion = Set(routeIDs)
                    prefillUnionExpertsBeforeCap &+= UInt64(uncappedUnion.count)
                    prefillUnionLayerTiles &+= 1

                    if let cap = prefillExpertUnionCap, uncappedUnion.count > cap {
                        var mass = [Float](repeating: 0, count: cfg.numExperts)
                        for index in routeIDs.indices {
                            mass[Int(routeIDs[index])] += abs(Float(routeWeights[index]))
                        }
                        let retained = Set(
                            mass.indices
                                .filter { mass[$0] > 0 }
                                .sorted {
                                    if mass[$0] != mass[$1] { return mass[$0] > mass[$1] }
                                    return $0 < $1
                                }
                                .prefix(cap)
                                .map(UInt32.init))
                        for token in 0..<t {
                            let start = token * cfg.topKExperts
                            let end = start + cfg.topKExperts
                            var originalMass: Float = 0
                            var retainedMass: Float = 0
                            for index in start..<end {
                                let value = Float(routeWeights[index])
                                originalMass += value
                                if retained.contains(routeIDs[index]) { retainedMass += value }
                            }
                            let scale = retainedMass != 0 ? originalMass / retainedMass : 0
                            for index in start..<end {
                                if retained.contains(routeIDs[index]) {
                                    routeWeights[index] = Float16(Float(routeWeights[index]) * scale)
                                } else {
                                    routeWeights[index] = 0
                                }
                                weightPtr[index] = routeWeights[index]
                            }
                        }
                    }
                    let pairs = PrefillRouter.makeTokenExpertPairs(indices: routeIDs,
                                                                   weights: routeWeights,
                                                                   queryCount: t,
                                                                   topK: cfg.topKExperts)
                        .filter { $0.weight != 0 }
                    prefillUnionExpertsAfterCap &+= UInt64(Set(pairs.map(\.expert)).count)
                    let schedulerConfig = Self.prefillRoutedTileSchedulerConfig
                    // A tile never needs more experts than the routed cache has
                    // slots; its scheduler drains before reusing one.
                    let routeTileExpertCount = model.routedExpertCacheSlotCount(layer: L)
                        .map { min(schedulerConfig.tileExperts, $0) } ?? schedulerConfig.tileExperts
                    let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
                        pairs,
                        queryCount: t,
                        topK: cfg.topKExperts,
                        numExperts: cfg.numExperts,
                        tileExpertCount: routeTileExpertCount,
                        expertSortKeys: model.routedExpertPhysicalOffsets(layer: L))

                    guard let sharedCB = ctx.queue.makeCommandBuffer() else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    sharedCB.label =
                        "prefill start=\(startPosition) count=\(t) layer=\(L) phase=shared_expert"
                    let sharedProj = sharedExpertProjections[L]
                    try prefillSharedExpert.encodeBlock(commandBuffer: sharedCB,
                                                        x: scratch.denseX,
                                                        y: scratch.h1,
                                                        gate: sharedProj.gate,
                                                        up: sharedProj.up,
                                                        down: sharedProj.down,
                                                        scratchGate: scratch.sharedGateScratch,
                                                        scratchUp: scratch.sharedUpScratch,
                                                        scratchAct: scratch.sharedActScratch,
                                                        queryCount: t,
                                                        d: D,
                                                        intermediate: cfg.intermediateSize,
                                                        xStrideElements: D,
                                                        yStrideElements: D,
                                                        smallBatch: verifyTokens != nil)
                    prefillRMS.encodeBF16W(commandBuffer: sharedCB,
                                           x: scratch.h1,
                                           weight: sharedProj.postF1.buffer,
                                           weightOffset: Int(sharedProj.postF1.offset),
                                           out: scratch.h1,
                                           t: UInt32(t),
                                           d: UInt32(D),
                                           eps: eps)
                    sharedCB.commit()
                    // Do not block here. Expert planning and pread are host
                    // work and can overlap the independent shared FFN. All
                    // routed/tail command buffers use the same queue, so Metal
                    // preserves the h1 dependency without a host round-trip.

                    let metadata = try prefillGroupedMoE.makeStreamedMetadataBuffers(
                        device: ctx.device,
                        routes: routes)
                    let routedOffsets = model.routedExpertOffsets(layer: L)
                    struct PendingPrefillTile {
                        let tileIndex: Int
                        let commandBuffer: MTLCommandBuffer
                        let fetch: PrefillStreamedTileFetchResult
                        let argumentBuffer: PrefillStreamedTileArgumentBuffer
                    }
                    var pendingTiles: [PendingPrefillTile] = []
                    var tileLifetime = PrefillStreamedTileSlotLifetime()
                    func drainOldestPendingTile() throws {
                        guard !pendingTiles.isEmpty else { return }
                        let drainStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                        defer { prefillTileDrainNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - drainStart }
                        let pending = pendingTiles.removeFirst()
                        try withExtendedLifetime((pending.fetch, pending.argumentBuffer)) {
                            try waitForCompletion(pending.commandBuffer)
                        }
                        prefillRoutedGPUNanos += UInt64(max(0, pending.commandBuffer.gpuEndTime - pending.commandBuffer.gpuStartTime) * 1e9)
                        if !pending.fetch.plannedAssignedSlots.isEmpty {
                            try tileLifetime.complete(tileIndex: pending.tileIndex)
                        }
                    }

                    let routedWallStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                    let routedTileScheduler = PrefillRoutedTileScheduler(config: schedulerConfig)
                    var routedOnHost = true
                    if verifyTokens == nil, let guest = supervisor.routedPrefill(layer: L) {
                        let completed = try await supervisor.supervise(layer: L) {
                            try await guest.runLayer(model: model, layer: L, hidden: scratch.routedX,
                                partials: scratch.routePartials, routes: routes, pairs: metadata.sortedPairs)
                        }
                        routedOnHost = !completed
                        if routedOnHost {
                            // The guest drained before throwing. Every routed
                            // row is rewritten below; clearing first keeps a
                            // partial guest write from surviving a host skip.
                            guard let clearCB = ctx.queue.makeCommandBuffer(),
                                  let blit = clearCB.makeBlitCommandEncoder() else {
                                throw ModelError.residentBufferWrapFailed
                            }
                            blit.fill(buffer: scratch.routePartials,
                                      range: 0..<scratch.routePartials.length, value: 0)
                            blit.endEncoding()
                            clearCB.commit()
                        }
                    }
                    if routedOnHost {
                    for (tileIndex, tile) in routes.tiles.enumerated() {
                        let expertIDs = try PrefillStreamedTileBinding.expertIDs(
                            forTile: tileIndex,
                            routes: routes)
                        var plannedFetch: RoutedExpertFetchPlan?
                        if !pendingTiles.isEmpty {
                            let pendingAssignedSlots = pendingTiles.flatMap(\.fetch.plannedAssignedSlots)
                            if !pendingAssignedSlots.isEmpty {
                                let pendingSlots = Set(pendingAssignedSlots)
                                let plan = try model.planRoutedExpertsIfPossible(
                                    layer: L,
                                    experts: expertIDs,
                                    avoidingSlots: pendingSlots)
                                let decision = routedTileScheduler.decide(
                                    PrefillRoutedTileSchedulerInput(
                                        hasPendingTile: true,
                                        pendingDepth: pendingTiles.count,
                                        pendingAssignedSlots: pendingAssignedSlots,
                                        avoidingSlotPlanAvailable: plan != nil))
                                switch decision {
                                case .prefetchNext:
                                    guard plan != nil else {
                                        throw ModelError.indexCorrupt(
                                            detail: "routed tile scheduler requested missing plan")
                                    }
                                    plannedFetch = plan
                                case .drainBeforeIssue:
                                    try drainOldestPendingTile()
                                case .issueWithoutPending:
                                    throw ModelError.indexCorrupt(
                                        detail: "routed tile scheduler ignored pending tile")
                                }
                            } else {
                                let decision = routedTileScheduler.decide(
                                    PrefillRoutedTileSchedulerInput(
                                        hasPendingTile: true,
                                        pendingDepth: pendingTiles.count,
                                        pendingAssignedSlots: [],
                                        avoidingSlotPlanAvailable: false))
                                switch decision {
                                case .drainBeforeIssue:
                                    try drainOldestPendingTile()
                                case .issueWithoutPending, .prefetchNext:
                                    throw ModelError.indexCorrupt(
                                        detail: "routed tile scheduler failed to drain empty-slot pending tile")
                                }
                            }
                        } else {
                            let decision = routedTileScheduler.decide(
                                PrefillRoutedTileSchedulerInput(
                                    hasPendingTile: false,
                                    pendingAssignedSlots: [],
                                    avoidingSlotPlanAvailable: false))
                            switch decision {
                            case .issueWithoutPending:
                                break
                            case .prefetchNext, .drainBeforeIssue:
                                throw ModelError.indexCorrupt(
                                    detail: "routed tile scheduler requested pending action without pending tile")
                            }
                        }
                        let fetchStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                        let fetch = try await PrefillStreamedTileBinding.fetchBindingForTile(
                            model: model,
                            layer: L,
                            tileIndex: tileIndex,
                            routes: routes,
                            plannedFetch: plannedFetch,
                            avoidingSlots: Set(pendingTiles.flatMap(
                                \.fetch.plannedAssignedSlots)))
                        prefillTileFetchNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - fetchStart
                        prefillTileHits &+= UInt64(fetch.plannedHits)
                        prefillTileMisses &+= UInt64(fetch.plannedMissIndices.count)
                        try fetch.binding.validateCoversPairs(routes.sortedPairs,
                                                              pairStart: Int(tile.pairStart),
                                                              pairCount: Int(tile.pairCount))
                        if !fetch.plannedAssignedSlots.isEmpty {
                            try tileLifetime.begin(tileIndex: tileIndex,
                                                   plannedSlots: fetch.plannedAssignedSlots)
                        }
                        let argumentBuffer = try prefillGroupedMoE.makeStreamedArgumentBuffer(
                            device: ctx.device,
                            binding: fetch.binding,
                            reuseSlot: tileIndex % 2)
                        let streamedParams = PrefillGroupedRoutedMoEStreamedParams(
                            pairStart: tile.pairStart,
                            pairCount: tile.pairCount,
                            d: UInt32(D),
                            routedIntermediate: UInt32(cfg.moeIntermediateSize),
                            topK: UInt32(cfg.topKExperts),
                            hiddenStrideElements: UInt32(D),
                            binding: fetch.binding,
                            offsets: routedOffsets)
                        guard let tileCB = ctx.queue.makeCommandBuffer() else {
                            throw ModelError.residentBufferWrapFailed
                        }
                        tileCB.label =
                            "prefill start=\(startPosition) count=\(t) layer=\(L) phase=routed_tile"
                        _ = prefillGroupedMoE.encodeStreamedBatched(
                            commandBuffer: tileCB,
                            hidden: scratch.routedX,
                            sortedPairs: metadata.sortedPairs,
                            routePartials: scratch.routePartials,
                            gateUpActScratch: scratch.routedGateUpActScratch,
                            downScratch: scratch.routedDownScratch,
                            argumentBuffer: argumentBuffer,
                            binding: fetch.binding,
                            params: streamedParams,
                            pairMicrobatchRows: scratch.layout.routedPairMicrobatchRows)
                        tileCB.commit()
                        pendingTiles.append(PendingPrefillTile(tileIndex: tileIndex,
                                                               commandBuffer: tileCB,
                                                               fetch: fetch,
                                                               argumentBuffer: argumentBuffer))
                        while pendingTiles.count > schedulerConfig.maxPendingDepth {
                            try drainOldestPendingTile()
                        }
                    }
                    while !pendingTiles.isEmpty {
                        try drainOldestPendingTile()
                    }
                    }
                    prefillRoutedWallNanos &+=
                        clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - routedWallStart
                    guard let tailCB = ctx.queue.makeCommandBuffer() else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    prefillMoE.encodeReduceTokenMajor(commandBuffer: tailCB,
                                                      routePartials: scratch.routePartials,
                                                      routeWeights: scratch.routeWeights,
                                                      h2: scratch.h2,
                                                      queryCount: UInt32(t),
                                                      topK: UInt32(cfg.topKExperts),
                                                      d: UInt32(D))
                    let scalarBits = views.layerScalar.buffer.contents()
                        .advanced(by: Int(views.layerScalar.offset))
                        .assumingMemoryBound(to: UInt16.self)[0]
                    prefillLayerTail.encode(commandBuffer: tailCB,
                                            h2: scratch.h2,
                                            h1: scratch.h1,
                                            hidden: scratch.hidden,
                                            postFFN2Weight: views.postFFN2.buffer,
                                            postFFN2WeightOffset: Int(views.postFFN2.offset),
                                            postFFNWeight: views.postFFN.buffer,
                                            postFFNWeightOffset: Int(views.postFFN.offset),
                                            queryCount: UInt32(t),
                                            d: UInt32(D),
                                            h2StrideElements: UInt32(D),
                                            h1StrideElements: UInt32(D),
                                            hiddenStrideElements: UInt32(D),
                                            eps: eps,
                                            layerScalar: Quantization.bf16ToFloat(scalarBits))
                    if let capture, let slot = capture.hiddenLayers.firstIndex(of: L),
                       let blit = tailCB.makeBlitCommandEncoder() {
                        let rowBytes = D * MemoryLayout<Float16>.stride
                        blit.copy(from: scratch.hidden, sourceOffset: 0, to: capture.buffer!,
                                  destinationOffset: slot * scratch.layout.chunkTokens * rowBytes,
                                  size: t * rowBytes)
                        blit.endEncoding()
                    }
                    tailCB.commit()
                    try withExtendedLifetime(metadata) {
                        try waitForCompletion(tailCB)
                    }
                    prefillSharedGPUNanos += UInt64(max(0, sharedCB.gpuEndTime - sharedCB.gpuStartTime) * 1e9)
                    try checkCommandBufferError(sharedCB)
                    if L + 1 < cfg.numLayers {
                        guard let nextCB = ctx.queue.makeCommandBuffer() else {
                            throw ModelError.residentBufferWrapFailed
                        }
                        nextCB.label =
                            "prefill start=\(startPosition) count=\(t) layer=\(L + 1) phase=qkv_attention"
                        cb = nextCB
                    }
                    continue
        }

        if writeFinalHead {
            let finalNorm = model.finalNorm
            let lm = model.lmHead
            guard let finalCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            finalCB.label = "prefill start=\(startPosition) count=\(t) phase=final_head"
            if let verifyTokens {
                fusionHead.encodeGreedyRows(
                    commandBuffer: finalCB,
                    hidden: scratch.hidden,
                    rowCount: t,
                    normWeight: finalNorm.buffer,
                    normOffset: Int(finalNorm.offset),
                    weights: lm.buffer,
                    weightsOffset: Int(lm.offset),
                    scales: lm.buffer,
                    scalesOffset: Int(lm.scaleOffset),
                    biases: lm.buffer,
                    biasesOffset: Int(lm.biasOffset),
                    outTokens: verifyTokens,
                    d: UInt32(D),
                    vocab: UInt32(cfg.vocabSize),
                    rmsEps: eps)
            } else if outputMode == .greedyIfAvailable, useFusedGreedyHead {
                fusionHead.encodeGreedyDecode(
                    commandBuffer: finalCB,
                    hidden: scratch.hidden,
                    hiddenOffset: (t - 1) * D * MemoryLayout<Float16>.stride,
                    normWeight: finalNorm.buffer,
                    normOffset: Int(finalNorm.offset),
                    weights: lm.buffer,
                    weightsOffset: Int(lm.offset),
                    scales: lm.buffer,
                    scalesOffset: Int(lm.scaleOffset),
                    biases: lm.buffer,
                    biasesOffset: Int(lm.biasOffset),
                    outToken: greedyTokenBuf,
                    outTokenOffset: 0,
                    d: UInt32(D),
                    vocab: UInt32(cfg.vocabSize),
                    rmsEps: eps)
            } else {
                prefillFinalRowHead.encodeLogits(commandBuffer: finalCB,
                                                 hiddenBlock: scratch.hidden,
                                                 row: t - 1,
                                                 rowStrideElements: D,
                                                 normWeight: finalNorm.buffer,
                                                 normWeightOffset: Int(finalNorm.offset),
                                                 weights: lm.buffer,
                                                 weightsOffset: Int(lm.offset),
                                                 scales: lm.buffer,
                                                 scalesOffset: Int(lm.scaleOffset),
                                                 biases: lm.buffer,
                                                 biasesOffset: Int(lm.biasOffset),
                                                 logits: logits,
                                                 d: UInt32(D),
                                                 vocab: UInt32(cfg.vocabSize),
                                                 rmsEps: eps)
            }
            finalCB.commit()
            try waitForCompletion(finalCB)
            if verifyTokens == nil, outputMode == .greedyIfAvailable, useFusedGreedyHead {
                lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
            }
        }

        if let capture, let buffer = capture.buffer {
            // Compact the per-layer slots to `[layers][t][D]` for the consumer.
            let base = buffer.contents().assumingMemoryBound(to: Float16.self)
            let slotElements = scratch.layout.chunkTokens * D
            for slot in capture.hiddenLayers.indices where slot > 0 {
                memmove(base + slot * t * D, base + slot * slotElements, t * D * MemoryLayout<Float16>.stride)
            }
            try capture.onChunk(PrefillCaptureChunk(
                startPosition: startPosition, tokenCount: t,
                hidden: UnsafeBufferPointer(start: base, count: capture.hiddenLayers.count * t * D),
                routes: capture.routes))
        }

        kv.advance(by: tokens.count)
        prefillChunkState.markCommitted()
    }

    private func produceToken(token: Int32,
                              position: Int,
                              into logits: MTLBuffer,
                              outputMode: PrefillOutputMode) async throws {
        let kvPosition = kv.position
        guard kvPosition == position else {
            throw PrefillError.prefillCursorMismatch(
                "produce cursor \(kvPosition) != position \(position)")
        }
        guard position < maxContext else {
            throw PrefillError.prefillCursorMismatch(
                "produce position \(position) exceeds maxContext \(maxContext)")
        }
        let D    = UInt32(cfg.hiddenSize)
        let FmoE = UInt32(cfg.moeIntermediateSize)
        let eps: Float = 1e-6
        let sqrtHidden = Float(cfg.hiddenSize).squareRoot()
        struct PendingRoutedCommand {
            let cb: MTLCommandBuffer
            let sharedCB: MTLCommandBuffer
            let phase1HitCB: MTLCommandBuffer?
            let encodeAndCommitNanos: UInt64
        }
        var pendingRoutedCommand: PendingRoutedCommand?

        func finishPendingRoutedCommand(_ pending: PendingRoutedCommand,
                                        waitIfNeeded: Bool) throws {
            if waitIfNeeded {
                waitUntilCompleted(pending.sharedCB)
                if let phase1HitCB = pending.phase1HitCB {
                    waitUntilCompleted(phase1HitCB)
                }
                waitUntilCompleted(pending.cb)
            }
            try checkCommandBufferError(pending.sharedCB)
            if let phase1HitCB = pending.phase1HitCB {
                try checkCommandBufferError(phase1HitCB)
            }
            try checkCommandBufferError(pending.cb)
            totalCb2Nanos &+= pending.encodeAndCommitNanos
        }

        func writeActiveSlots(_ slots: [UInt32], into buffer: MTLBuffer) {
            let ptr = buffer.contents().assumingMemoryBound(to: UInt32.self)
            for i in 0..<slots.count { ptr[i] = slots[i] }
        }

        // Embed lookup + sqrt(H) fused.
        let emb = model.embedding
        do {
            try runSync { cb in
                embedInt4.encode(commandBuffer: cb,
                                 table:  emb.buffer, tableOffset:  Int(emb.offset),
                                 scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                                 biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                                 out: hidden,
                                 tokenId: UInt32(bitPattern: token),
                                 d: D,
                                 outScale: sqrtHidden)
            }
        }

        for L in 0..<cfg.numLayers {
            let isFull = cfg.fullAttentionLayerMask[L] != 0
            let headDimL = isFull ? cfg.fullHeadDim : cfg.headDim
            let numKVL   = isFull ? cfg.numFullKVHeads : cfg.numKVHeads
            let qDim     = UInt32(cfg.numHeads * headDimL)
            let kvDim    = UInt32(numKVL * headDimL)
            let kSlot    = kv.kSlot(layer: L, position: position)
            let vSlot    = kv.vSlot(layer: L, position: position)
            let seqLen   = UInt32(position + 1)

            let inNorm   = try model.inputNorm(layer: L)
            let q        = try model.qProj(layer: L)
            let k        = try model.kProj(layer: L)
            // v_proj only exists on SWA layers; full layers reuse k_proj.
            let vProj    = isFull ? k : (try model.vProj(layer: L))
            let o        = try model.oProj(layer: L)
            let postAttn = try model.postAttnNorm(layer: L)
            let qNorm    = try model.qNorm(layer: L)
            let kNorm    = try model.kNorm(layer: L)
            let preFFN   = try model.preFFN(layer: L)
            let preFFN2  = try model.preFFN2(layer: L)
            let sharedProj = sharedExpertProjections[L]
            let postF2   = try model.postFFN2(layer: L)
            let postF    = try model.postFFN(layer: L)
            let routerW  = try model.router(layer: L)
            let perExpertScale = try model.routerPerExpertScale(layer: L)
            let layerScalarView = try model.layerScalar(layer: L)

            let tCb1Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            // Everything up to and including the router runs in a single CB:
            // the only reason to break is the CPU readback of router indices
            // needed to issue I/O for the routed-expert blobs.
            let gInputNorm: (MTLCommandBuffer) -> Void = { [self] cb in
                rms.encodeBF16W(commandBuffer: cb,
                                x: hidden,
                                weight: inNorm.buffer, weightOffset: Int(inNorm.offset),
                                out: normed,
                                d: D, eps: eps)
            }

            let gQKV: (MTLCommandBuffer) -> Void = { [self] cb in
                fusedQKVGEMV.encode(commandBuffer: cb,
                                    qWeights: q.buffer, qWeightsOffset: Int(q.offset),
                                    qScales: q.buffer, qScalesOffset: Int(q.scaleOffset),
                                    qBiases: q.buffer, qBiasesOffset: Int(q.biasOffset),
                                    kWeights: k.buffer, kWeightsOffset: Int(k.offset),
                                    kScales: k.buffer, kScalesOffset: Int(k.scaleOffset),
                                    kBiases: k.buffer, kBiasesOffset: Int(k.biasOffset),
                                    vWeights: vProj.buffer, vWeightsOffset: Int(vProj.offset),
                                    vScales: vProj.buffer, vScalesOffset: Int(vProj.scaleOffset),
                                    vBiases: vProj.buffer, vBiasesOffset: Int(vProj.biasOffset),
                                    x: normed,
                                    qOut: qScratch,
                                    kOut: kSlot.buffer, kOutOffset: kSlot.offset,
                                    vOut: vSlot.buffer, vOutOffset: vSlot.offset,
                                    qRows: qDim,
                                    kvRows: kvDim,
                                    n: D)
            }

            let gQKVEpilogue: (MTLCommandBuffer) -> Void = { [self] cb in
                let rotated = isFull
                    ? UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor / 2.0)
                    : UInt32(headDimL / 2)
                fusedQKVEpilogue.encode(commandBuffer: cb,
                                        q: qScratch,
                                        k: kSlot.buffer,
                                        kOffset: kSlot.offset,
                                        v: vSlot.buffer,
                                        vOffset: vSlot.offset,
                                        qWeight: qNorm.buffer,
                                        qWeightOffset: Int(qNorm.offset),
                                        kWeight: kNorm.buffer,
                                        kWeightOffset: Int(kNorm.offset),
                                        headDim: UInt32(headDimL),
                                        numQHeads: UInt32(cfg.numHeads),
                                        numKVHeads: UInt32(numKVL),
                                        position: UInt32(position),
                                        theta: isFull ? Float(cfg.fullRopeTheta) : Float(cfg.ropeTheta),
                                        rotatedPairs: rotated,
                                        eps: eps)
            }

            let gAttention: (MTLCommandBuffer) -> Void = { [self] cb in
                if isFull {
                    attention.encodeFull(commandBuffer: cb,
                                         q: qScratch,
                                         k: kSlot.buffer, kOffset: 0,
                                         v: vSlot.buffer, vOffset: 0,
                                         out: attnOut,
                                         headDim: UInt32(headDimL),
                                         numQHeads: UInt32(cfg.numHeads),
                                         numKVHeads: UInt32(numKVL),
                                         seqLen: seqLen,
                                         scale: 1.0)
                } else {
                    let ringCapacity = kv.ringCapacity(layer: L)
                    let activeRingCapacity = ringCapacity > 0 && Int(seqLen) > ringCapacity
                        ? UInt32(ringCapacity)
                        : 0
                    attention.encodeSWA(commandBuffer: cb,
                                        q: qScratch,
                                        k: kSlot.buffer, kOffset: 0,
                                        v: vSlot.buffer, vOffset: 0,
                                        out: attnOut,
                                        headDim: UInt32(headDimL),
                                        numQHeads: UInt32(cfg.numHeads),
                                        numKVHeads: UInt32(numKVL),
                                        seqLen: seqLen,
                                        window: UInt32(cfg.slidingWindow),
                                        scale: 1.0,
                                        ringCapacity: activeRingCapacity)
                }
            }
            let gOProj: (MTLCommandBuffer) -> Void = { [self] cb in
                int4.encode(commandBuffer: cb,
                            weights: o.buffer, weightsOffset: Int(o.offset),
                            scales:  o.buffer, scalesOffset:  Int(o.scaleOffset),
                            biases:  o.buffer, biasesOffset:  Int(o.biasOffset),
                            x: attnOut, y: oOut, m: D, n: qDim)
            }

            let gPostAttnSetup: (MTLCommandBuffer) -> Void = { [self] cb in
                fusedPostAttentionSetup.encode(commandBuffer: cb,
                                               hidden: hidden,
                                               attn: oOut,
                                               denseX: denseX,
                                               routedX: routedX,
                                               routerX: routerInput,
                                               postAttentionWeight: postAttn.buffer,
                                               postAttentionWeightOffset: Int(postAttn.offset),
                                               preFFNWeight: preFFN.buffer,
                                               preFFNWeightOffset: Int(preFFN.offset),
                                               preFFN2Weight: preFFN2.buffer,
                                               preFFN2WeightOffset: Int(preFFN2.offset),
                                               d: D,
                                               eps: eps)
            }

            let gRouter: (MTLCommandBuffer) -> Void = { [self] cb in
                moe.encodeRouterGemma4(commandBuffer: cb,
                    weights: routerW.buffer, weightsOffset: Int(routerW.offset),
                    scales:  routerW.buffer, scalesOffset:  Int(routerW.scaleOffset),
                    biases:  routerW.buffer, biasesOffset:  Int(routerW.biasOffset),
                    hidden: routerInput,
                    effectiveScale: effectiveScaleBuffers[L],
                    perExpertScale: perExpertScale.buffer,
                    perExpertScaleOffset: Int(perExpertScale.offset),
                    outIndices: outIndices, outWeights: outWeights,
                    numExperts: UInt32(cfg.numExperts), d: D, topK: UInt32(cfg.topKExperts))
            }

            let gSharedFFN: (MTLCommandBuffer) -> Void = { [self] cb in
                try! shared.encode(commandBuffer: cb,
                                   x: denseX,
                                   gate: sharedProj.gate,
                                   up: sharedProj.up,
                                   down: sharedProj.down,
                                   y: h1Buf,
                                   scratchGate: denseScratchGate,
                                   scratchUp: denseScratchUp,
                                   scratchAct: denseScratchAct)
            }
            let gSharedNorm: (MTLCommandBuffer) -> Void = { [self] cb in
                rms.encodeBF16W(commandBuffer: cb, x: h1Buf,
                                weight: sharedProj.postF1.buffer,
                                weightOffset: Int(sharedProj.postF1.offset),
                                out: h1Buf, d: D, eps: eps)
            }
            let cb = ctx.queue.makeCommandBuffer()!
            gInputNorm(cb)
            gQKV(cb)
            gQKVEpilogue(cb)
            gAttention(cb)
            gOProj(cb)
            gPostAttnSetup(cb)
            gRouter(cb)
            cb.commit()
            let tWait = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            waitUntilCompleted(cb)
            let waitNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tWait
            if let pending = pendingRoutedCommand {
                try finishPendingRoutedCommand(pending, waitIfNeeded: false)
                pendingRoutedCommand = nil
            }
            try checkCommandBufferError(cb)
            totalCb1Nanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb1Start - waitNanos

            // CPU readback to fetch routed-expert blobs from disk.
            let idxPtr = outIndices.contents().bindMemory(to: UInt32.self,
                                                          capacity: cfg.topKExperts)
            let fullRoutingWeights = snapshotFP16(outWeights,
                                                   count: cfg.topKExperts)
            if fullRoutingWeights.count == routedWeightMassByRank.count {
                for rank in fullRoutingWeights.indices {
                    routedWeightMassByRank[rank] += Double(fullRoutingWeights[rank])
                }
                routedWeightMassSamples &+= 1
            }
            var experts = [Int](repeating: 0, count: activeRoutedExperts)
            for i in 0..<activeRoutedExperts {
                experts[i] = min(Int(idxPtr[i]), cfg.numExperts - 1)
            }
            var currentRoutingWeights = Array(fullRoutingWeights.prefix(activeRoutedExperts))
            if activeRoutedExperts < cfg.topKExperts {
                let fullMass = fullRoutingWeights.reduce(Float(0)) { $0 + Float($1) }
                let keptMass = currentRoutingWeights.reduce(Float(0)) { $0 + Float($1) }
                if keptMass != 0 {
                    let scale = fullMass / keptMass
                    let weightPtr = outWeights.contents().assumingMemoryBound(to: Float16.self)
                    for rank in currentRoutingWeights.indices {
                        currentRoutingWeights[rank] = Float16(Float(currentRoutingWeights[rank]) * scale)
                        weightPtr[rank] = currentRoutingWeights[rank]
                    }
                }
            }

            let routedOffsets = model.routedExpertOffsets(layer: L)
            let topK = UInt32(activeRoutedExperts)
            let canPlanPhase1HitSplit =
                activeRoutedExperts <= MoE.maxStreamedExperts
            let plannedFetch = canPlanPhase1HitSplit
                ? try model.planRoutedExperts(layer: L, experts: experts)
                : nil
            if let plannedFetch {
                routedExpertCacheHits &+= UInt64(plannedFetch.hits)
                routedExpertCacheMisses &+= UInt64(plannedFetch.misses.count)
            }
            var phase1HitCB: MTLCommandBuffer?
            var phase1HitSplitArgBuf: MTLBuffer?
            var phase1HitSplitRoutedBufs: [MTLBuffer] = []
            var phase1HitSlots: [UInt32] = []
            var phase1MissSlots: [UInt32] = []

            if let plan = plannedFetch {
                let missSet = Set(plan.misses)
                phase1HitSlots = (0..<activeRoutedExperts)
                    .filter { !missSet.contains($0) }
                    .map { UInt32($0) }
                phase1MissSlots = plan.misses.map { UInt32($0) }
            }
            func encodeRoutedPhase1Full(
                _ cb: MTLCommandBuffer,
                argBuf: MTLBuffer,
                routedBufs: [MTLBuffer]
            ) {
                moe.encodeRoutedPersistentPhase1U16Load(commandBuffer: cb,
                                                        routedArgBuffer: argBuf,
                                                        routedBlobs: routedBufs,
                                                        routedOffsets: routedOffsets,
                                                        x: routedX,
                                                        acts: moeActs,
                                                        d: D,
                                                        f: FmoE,
                                                        topK: topK)
            }

            func encodeRoutedPhase1Subset(
                _ cb: MTLCommandBuffer,
                argBuf: MTLBuffer,
                routedBufs: [MTLBuffer],
                activeSlots: MTLBuffer,
                activeSlotIndices: [UInt32],
                activeCount: UInt32
            ) {
                moe.encodeRoutedPersistentPhase1SubsetU16Load(
                    commandBuffer: cb,
                    routedArgBuffer: argBuf,
                    routedBlobs: routedBufs,
                    routedOffsets: routedOffsets,
                    x: routedX,
                    acts: moeActs,
                    activeSlots: activeSlots,
                    activeSlotIndices: activeSlotIndices,
                    activeCount: activeCount,
                    d: D,
                    f: FmoE,
                    topK: topK)
            }

            if let plan = plannedFetch,
               plan.hits > 0,
               !plan.misses.isEmpty {
                let plannedBlobs = try model.routedExpertBuffers(for: plan)
                phase1HitSplitRoutedBufs = plannedBlobs.map { $0.buffer }
                phase1HitSplitArgBuf = moe.makeRoutedArgumentBuffer(
                    routedViews: plannedBlobs,
                    topK: topK)
                if let argBuf = phase1HitSplitArgBuf, plan.hits > 0, !plan.misses.isEmpty {
                    writeActiveSlots(phase1HitSlots, into: moeHitActiveSlots)
                    let cb = ctx.queue.makeCommandBuffer()!
                    encodeRoutedPhase1Subset(
                        cb,
                        argBuf: argBuf,
                        routedBufs: phase1HitSplitRoutedBufs,
                        activeSlots: moeHitActiveSlots,
                        activeSlotIndices: phase1HitSlots,
                        activeCount: UInt32(phase1HitSlots.count))
                    phase1HitCB = cb
                }
            }

            // The shared dense MLP depends only on denseX, not on the routed
            // experts. Commit it without waiting so its GPU work overlaps the
            // routed-expert pread. The routed CB follows it on the same queue,
            // so the combine sees h1Buf.
            let sharedCB = ctx.queue.makeCommandBuffer()!
            gSharedFFN(sharedCB)
            gSharedNorm(sharedCB)
            sharedCB.commit()
            phase1HitCB?.commit()

            // Routed-expert pread — overlaps the shared MLP GPU work above.
            let tIoStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let blobs: [TensorView]
            if let plannedFetch {
                blobs = try await model.fetchRoutedExperts(plan: plannedFetch)
            } else {
                blobs = try await model.fetchRoutedExperts(layer: L, experts: experts)
            }
            let layerIo = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tIoStart
            totalIoNanos &+= layerIo
            let routedBufs = blobs.map { $0.buffer }
            let tCb2Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let scalarPtr = layerScalarView.buffer.contents()
                .advanced(by: Int(layerScalarView.offset))
                .assumingMemoryBound(to: UInt16.self)
            let layerScalar = Quantization.bf16ToFloat(scalarPtr[0])

            let gTail: (MTLCommandBuffer) -> Void = { [self] cb in
                fusedTail.encode(commandBuffer: cb,
                                 h2: h2Buf,
                                 h1: h1Buf,
                                 hidden: hidden,
                                 postFFN2Weight: postF2.buffer,
                                 postFFN2WeightOffset: Int(postF2.offset),
                                 postFFNWeight: postF.buffer,
                                 postFFNWeightOffset: Int(postF.offset),
                                 d: D,
                                 eps: eps,
                                 layerScalar: layerScalar)
            }
            let routedCB = ctx.queue.makeCommandBuffer()!
            let splitArgBuf = phase1HitCB != nil && !phase1MissSlots.isEmpty
                ? phase1HitSplitArgBuf
                : nil
            let argBuf = splitArgBuf ?? moe.makeReusedRoutedArgumentBuffer(
                routedViews: blobs, topK: topK)
            if phase1HitCB != nil && !phase1MissSlots.isEmpty {
                writeActiveSlots(phase1MissSlots, into: moeMissActiveSlots)
                encodeRoutedPhase1Subset(
                    routedCB,
                    argBuf: argBuf,
                    routedBufs: routedBufs,
                    activeSlots: moeMissActiveSlots,
                    activeSlotIndices: phase1MissSlots,
                    activeCount: UInt32(phase1MissSlots.count))
            } else {
                encodeRoutedPhase1Full(routedCB,
                                       argBuf: argBuf,
                                       routedBufs: routedBufs)
            }
            moe.encodeRoutedPersistentPhase2Reduce(commandBuffer: routedCB,
                                                   routedArgBuffer: argBuf,
                                                   routedBlobs: routedBufs,
                                                   routedOffsets: routedOffsets,
                                                   acts: moeActs,
                                                   routingWeights: outWeights,
                                                   residual: zeroResidual,
                                                   y: h2Buf,
                                                   d: D,
                                                   f: FmoE,
                                                   topK: topK)
            gTail(routedCB)
            routedCB.commit()
            precondition(pendingRoutedCommand == nil,
                         "routed command-buffer pipeline drained before queuing the next layer")
            pendingRoutedCommand = PendingRoutedCommand(
                cb: routedCB,
                sharedCB: sharedCB,
                phase1HitCB: phase1HitCB,
                encodeAndCommitNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb2Start)
            continue
        }
        if let pending = pendingRoutedCommand {
            try finishPendingRoutedCommand(pending, waitIfNeeded: true)
            pendingRoutedCommand = nil
        }
        // The fused head skips the vocab buffer and leaves a greedy token in
        // greedyTokenBuf; the logits path writes the complete vector.
        let fNorm = model.finalNorm
        let lm    = model.embedding
        let gFinalNorm: (MTLCommandBuffer) -> Void = { cb in
            self.rms.encodeBF16W(commandBuffer: cb, x: self.hidden,
                                 weight: fNorm.buffer, weightOffset: Int(fNorm.offset),
                                 out: self.normed, d: D, eps: eps)
        }
        let gLmHead: (MTLCommandBuffer) -> Void = { cb in
            self.int4.encode(commandBuffer: cb,
                             weights: lm.buffer, weightsOffset: Int(lm.offset),
                             scales:  lm.buffer, scalesOffset:  Int(lm.scaleOffset),
                             biases:  lm.buffer, biasesOffset:  Int(lm.biasOffset),
                             x: self.normed, y: logits, m: UInt32(self.cfg.vocabSize), n: D)
        }
        let gFusionHead: (MTLCommandBuffer) -> Void = { cb in
            self.fusionHead.encodeGreedyDecode(
                commandBuffer: cb,
                hidden: self.hidden,
                normWeight: fNorm.buffer, normOffset: Int(fNorm.offset),
                weights: lm.buffer, weightsOffset: Int(lm.offset),
                scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                outToken: self.greedyTokenBuf,
                d: D, vocab: UInt32(self.cfg.vocabSize),
                rmsEps: eps)
        }
        let tHead = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        if useFusedGreedyHead && outputMode == .greedyIfAvailable {
            try runSync(gFusionHead)
            totalHeadFusedNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
            lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
        } else {
            try runSync { cb in
                gFinalNorm(cb)
                gLmHead(cb)
            }
            totalHeadNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
        }

        kv.advance()
    }

    private func runSync(_ body: (MTLCommandBuffer) -> Void) throws {
        let cb = ctx.queue.makeCommandBuffer()!
        body(cb)
        cb.commit()
        try waitForCompletion(cb)
    }

    private nonisolated func waitForCompletion(_ cb: MTLCommandBuffer) throws {
        waitUntilCompleted(cb)
        try checkCommandBufferError(cb)
    }

    private nonisolated func waitUntilCompleted(_ cb: MTLCommandBuffer) {
        cb.waitUntilCompleted()
    }

}
