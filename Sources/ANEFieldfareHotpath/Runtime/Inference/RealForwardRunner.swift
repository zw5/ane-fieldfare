import Foundation
public import Metal

/// The Gemma 4 forward pass (decode, chunked prefill and verify) against the
/// `.gturbo` model. Per token:
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
///   token = head(rmsnorm_bf16w(h, model.norm), embed_table^T)
public final class RealForwardRunner: ChunkedPrefillRunner, MultimodalPrefillRunner, ContextWindowReporting,
    SamplingTokenProducer, PrefixReusingProducer {
    private struct LayerSharedExpertProjections {
        let gate: SharedExpertProjection
        let up: SharedExpertProjection
        let down: SharedExpertProjection
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
    private let prefillMPPAffineInt4: MPPPrefillInt4QMM
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
    private let hidden: any MTLBuffer        // [D] FP16
    private let normed: any MTLBuffer        // [D] FP16
    private let attnOut: any MTLBuffer       // [N_HEADS * head_dim] FP16
    private let qScratch: any MTLBuffer      // [N_HEADS * head_dim] FP16
    private let oOut: any MTLBuffer          // [D] FP16
    private let h1Buf: any MTLBuffer         // [D] FP16 (dense MLP output)
    private let h2Buf: any MTLBuffer         // [D] FP16 (routed output)
    private let routedX: any MTLBuffer       // [D] FP16 (pre_feedforward_layernorm_2 output)
    private let denseX: any MTLBuffer        // [D] FP16 (pre_feedforward_layernorm output)
    private let denseScratchAct: any MTLBuffer  // [F] FP16
    private let routerInput: any MTLBuffer   // [D] FP16 (rmsnorm_no_scale(h))
    private let nextLayerScores: any MTLBuffer // [numExperts] FP32, early-read guess
    private let zeroResidual: any MTLBuffer  // [D] FP16 zeros — for routed branch base
    private let outIndices: any MTLBuffer    // [topK] UInt32
    private let outWeights: any MTLBuffer    // [topK] FP16
    private let moeActs: any MTLBuffer       // [topK * FmoE] FP16
    // GPU-routed decode: the routed slots the GPU fills, each layer's expert
    // address table (0 where unbound), the zero expert an unbound slot reads,
    // the layer input kept for a host patch, and per layer the slots a patch
    // recomputes (several patches can be in flight).
    private let routedSlots: any MTLBuffer          // [maxStreamedExperts] UInt64
    private let routedAddressTable: any MTLBuffer   // [numLayers * numExperts] UInt64
    private let routedFallback: any MTLBuffer       // one zeroed expert
    private let savedHidden: any MTLBuffer          // [D] FP16
    private let patchSlots: any MTLBuffer           // [numLayers * topK] UInt32
    private let gpuRoutedDecode: Bool
    /// Per layer, the expert GPU addresses and the mapping they belong to: a
    /// layer can be closed and reopened at new addresses (vision residency).
    private final class ExpertAddresses {
        weak var layer: MappedExpertLayer?
        let addresses: [UInt64]
        init(layer: MappedExpertLayer, addresses: [UInt64]) {
            self.layer = layer
            self.addresses = addresses
        }
    }
    private var routedExpertAddressCache: [Int: ExpertAddresses] = [:]
    /// Single-page probes per expert when choosing which experts a decode
    /// layer binds. A wrong "resident" costs a page fault at submission.
    static let bindProbes = 2
    /// Issues verify's advisory next-layer reads in order, off its thread.
    // SAFETY: a serial queue keeps advisories in issue order, so storage
    // serves the earliest-needed layer first. Each work item captures only a
    // `Sendable` layer mapping and its expert IDs.
    private static let readAheadQueue = DispatchQueue(label: "ane-fieldfare.verify-read-ahead", qos: .userInitiated)
    /// Guessed experts a GPU-routed decode layer binds.
    private static let bindTop = 24
    /// Decode folds the shared expert's post-FFN-1 norm and the next layer's
    /// input norm into the layer tail.
    private var fuseTailNorms: Bool { fusedTail.fusesNorms }
    private let greedyTokenBuf: any MTLBuffer // 4 B UInt32 fused-head output
    /// The hot-vocabulary head `HotHeadDecoder` guesses with; nil when the
    /// runtime ships no list for this vocabulary or the head writes logits.
    let hotVocabulary: HotVocabulary?
    /// When set, decode leaves its normalized final row at `row` of
    /// `headCheckRows`. With `check` false it guesses its token from
    /// `hotVocabulary`; with `check` true it runs the full head over rows
    /// 0...row instead, leaving their tokens in `checkedTokens` and its own as
    /// `lastGreedyToken`.
    var hotHead: (row: Int, check: Bool)?
    private(set) var checkedTokens: [UInt32] = []
    private let headCheckRows: any MTLBuffer
    private let headCheckTokens: any MTLBuffer
    private var prefillChunkState = PrefillChunkCommitState()
    private var prefillScratch: PrefillChunkScratchBuffers?
    /// Owns the ANE guest; the forward loop asks it per layer whether the
    /// guest runs routed prefill.
    let supervisor: DeviceSupervisor
    public var aneRoutedPrefillProfile: [String: Any]? { supervisor.routedPrefillProfile }
    public var supervisorProfile: [String: Any] { supervisor.profile }

    /// Per-layer `router.scale * D^-0.5`, folded once at init into one BF16
    /// buffer per layer.
    private let effectiveScaleBuffers: [any MTLBuffer]
    private let sharedExpertProjections: [LayerSharedExpertProjections]
    private let layers: [LayerWeights]
    /// Next-layer early read; nil when `RuntimeConfiguration.earlyExpertRead`
    /// is off. See `NextLayerExpertPrefetcher`.
    private let earlyExpertReads: NextLayerExpertPrefetcher?
    /// Early-read counts since this runner was created; nil when it is off.
    public var earlyExpertReadStats: EarlyExpertReadStats? { earlyExpertReads?.stats }
    /// The guess the early read scores with, `fitted` or `router`; `off` when off.
    public var earlyExpertReadGuess: RuntimeEarlyExpertRead { earlyExpertReads?.guess ?? .off }

    public let maxContext: Int

    /// The fused head (default) skips the 512 KB logits write and leaves its
    /// token in `lastGreedyToken`; a caller that samples from the logits
    /// buffer needs `forceLogitsHead: true` or reads a never-written buffer.
    public let usesFusedGreedyHead: Bool
    private let activeRoutedExperts: Int
    public let prefillExpertUnionCap: Int?
    /// Set to record selected layer outputs and every layer's routes during
    /// chunked prefill.
    public var prefillCapture: PrefillCapture?
    /// Set while a drafter is attached: prefill, verify and decode copy the
    /// drafter's context layers into its feature ring.
    var draftFeatureTap: DraftFeatureTap?
    /// Decode GPU time: attention through router, shared FFN, routed + tail.
    public private(set) var decodeAttentionGPUNanos: UInt64 = 0
    public private(set) var decodeSharedGPUNanos: UInt64 = 0
    public private(set) var decodeRoutedGPUNanos: UInt64 = 0
    /// Decode layers that ran the host-routed experts: layer 0 always, a
    /// GPU-routed layer that routed to an unbound expert, and every layer when
    /// GPU routing is off (no early reads).
    public private(set) var decodeLayerPatches: UInt64 = 0
    /// Signalled by each decode layer's router, ahead of its command buffer's
    /// completion.
    private let routerEvent: any MTLSharedEvent
    private var routerEventValue: UInt64 = 0
    /// Verify routes a layer's experts on the GPU, from the previous layer's
    /// guess, before the host has read its routes;
    /// `ANE_FIELDFARE_HOST_ROUTED_VERIFY` plans every layer on the host.
    private let gpuRoutedVerify = ProcessInfo.processInfo.environment["ANE_FIELDFARE_HOST_ROUTED_VERIFY"] == nil
    /// Guessed experts per verify row a GPU-routed layer binds.
    private lazy var verifyBindTop = max(1, min(cfg.numExperts,
        ProcessInfo.processInfo.environment["ANE_FIELDFARE_VERIFY_BIND_TOP"].flatMap(Int.init) ?? 48))
    /// Per layer, the GPU-routed verify pass's bound experts.
    private let verifyBoundList: any MTLBuffer
    /// Verify routes the GPU-routed pass ran, and layers that needed a host
    /// patch for the rest.
    public private(set) var boundVerifyRoutes: UInt64 = 0
    public private(set) var patchedVerifyLayers: UInt64 = 0
    /// A verify block's token IDs and greedy next tokens, for `verify`.
    private let verifyTokenIDs: any MTLBuffer
    private let verifyGreedyTokens: any MTLBuffer
    /// Per-row next-layer guesses for a verify block's early read.
    private let chunkGuessIDs: any MTLBuffer
    private let chunkGuessWeights: any MTLBuffer
    /// Experts advised ahead of verify blocks' next layers.
    public private(set) var prefillEarlyReads: UInt64 = 0
    public convenience init(model: Model, context: MetalContext, maxContext: Int,
                runtimeConfiguration: RuntimeConfiguration = RuntimeConfiguration()) throws {
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
        self.usesFusedGreedyHead = runtimeConfiguration.headPath == .fusedRows
        self.activeRoutedExperts = runtimeConfiguration.activeRoutedExperts
        self.prefillExpertUnionCap = runtimeConfiguration.prefillExpertUnionCap
        self.supervisor = try supervisor ?? DeviceSupervisor(
            configuration: runtimeConfiguration, model: model, context: context)
        self.kv = try KVCacheManager(device: context.device,
                                     config: cfg,
                                     maxContext: maxContext,
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

        func buf(_ count: Int, _ stride: Int = MemoryLayout<Float16>.size) throws -> any MTLBuffer {
            guard let b = device.makeBuffer(length: max(count, 1) * stride,
                                            options: .storageModeShared) else {
                throw MetalError.allocationFailed("runner scratch")
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
        self.denseScratchAct  = try buf(F)
        self.routerInput   = try buf(D)
        self.nextLayerScores = try buf(cfg.numExperts, MemoryLayout<Float>.size)
        self.zeroResidual  = try buf(D)
        // The routed MoE kernel seeds y[d] = residual[d]; pinning this buffer
        // to zero once at init makes the routed branch's residual contribution
        // exactly zero (it's combined with the dense MLP downstream).
        unsafe memset(self.zeroResidual.contents(), 0, self.zeroResidual.length)
        self.outIndices    = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size)
        self.outWeights    = try buf(cfg.topKExperts)
        self.moeActs       = try buf(cfg.topKExperts * cfg.moeIntermediateSize)
        guard let routerEvent = device.makeSharedEvent() else {
            throw MetalError.allocationFailed("router event")
        }
        self.routerEvent = routerEvent
        self.routedSlots = try buf(MoE.maxStreamedExperts, MemoryLayout<UInt64>.size)
        self.routedAddressTable = try buf(cfg.numLayers * cfg.numExperts, MemoryLayout<UInt64>.size)
        self.routedFallback = try buf(model.routedExpertBytes, 1)
        unsafe memset(self.routedFallback.contents(), 0, self.routedFallback.length)
        self.savedHidden = try buf(cfg.hiddenSize)
        self.patchSlots = try buf(cfg.numLayers * cfg.topKExperts, MemoryLayout<UInt32>.size)
        self.gpuRoutedDecode = moe.routedSlotsAreAddresses
        self.greedyTokenBuf = try buf(1, MemoryLayout<UInt32>.size)
        let vocab = cfg.vocabSize, hiddenSize = cfg.hiddenSize
        self.hotVocabulary = try runtimeConfiguration.headPath == .fusedRows
            && UInt32(hiddenSize) == LMHeadChainInt4.realDecodeD
            ? HotVocabulary.shipped(vocab: vocab).map {
                try HotVocabulary(url: $0, vocab: vocab, device: device)
            }
            : nil
        self.headCheckRows = try buf(LMHeadChainInt4.maxCheckRows * cfg.hiddenSize)
        self.headCheckTokens = try buf(LMHeadChainInt4.maxCheckRows, MemoryLayout<UInt32>.size)
        self.chunkGuessIDs = try buf(Self.maxVerifyTokens * cfg.topKExperts, MemoryLayout<UInt32>.size)
        self.verifyTokenIDs = try buf(Self.maxVerifyTokens, MemoryLayout<UInt32>.size)
        self.verifyGreedyTokens = try buf(Self.maxVerifyTokens, MemoryLayout<UInt32>.size)
        self.verifyBoundList = try buf(cfg.numLayers * cfg.numExperts, MemoryLayout<VerifyBoundExpert>.stride)
        self.chunkGuessWeights = try buf(Self.maxVerifyTokens * cfg.topKExperts)
        self.routedWeightMassByRank = Array(repeating: 0, count: cfg.topKExperts)

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
        var layerWeights: [LayerWeights] = []
        for L in 0..<cfg.numLayers {
            let isFull = cfg.fullAttentionLayerMask[L] != 0
            let scalar = try model.layerScalar(layer: L)
            let scalarBits = unsafe scalar.buffer.contents().advanced(by: Int(scalar.offset))
                .assumingMemoryBound(to: UInt16.self)[0]
            layerWeights.append(LayerWeights(
                isFull: isFull,
                headDim: isFull ? cfg.fullHeadDim : cfg.headDim,
                numKVHeads: isFull ? cfg.numFullKVHeads : cfg.numKVHeads,
                inputNorm: try model.inputNorm(layer: L), q: try model.qProj(layer: L),
                k: try model.kProj(layer: L),
                // Full-attention layers have no v_proj and reuse k_proj.
                v: isFull ? try model.kProj(layer: L) : try model.vProj(layer: L),
                o: try model.oProj(layer: L), postAttention: try model.postAttnNorm(layer: L),
                preFFN: try model.preFFN(layer: L), preFFN2: try model.preFFN2(layer: L),
                postFFN2: try model.postFFN2(layer: L), postFFN: try model.postFFN(layer: L),
                qNorm: try model.qNorm(layer: L), kNorm: try model.kNorm(layer: L),
                router: try model.router(layer: L),
                routerPerExpertScale: try model.routerPerExpertScale(layer: L),
                layerScalar: Quantization.bf16ToFloat(scalarBits)))
        }
        self.layers = layerWeights

        // Pre-fold 1/sqrt(D) into router.scale per layer. Each layer gets its
        // own BF16 [D] buffer — the kernel reads `effective_scale[i]` and we
        // pay for the multiply once per generation, not per token.
        var perLayer: [any MTLBuffer] = []
        perLayer.reserveCapacity(cfg.numLayers)
        let invSqrtD = Float(1.0) / Float(D).squareRoot()
        for L in 0..<cfg.numLayers {
            let scaleView = try model.routerScale(layer: L)
            guard let buf = device.makeBuffer(length: D * MemoryLayout<UInt16>.size,
                                              options: .storageModeShared) else {
                throw MetalError.allocationFailed("router effective scale")
            }
            let src = unsafe scaleView.buffer.contents()
                .advanced(by: Int(scaleView.offset))
                .assumingMemoryBound(to: UInt16.self)
            let dst = unsafe buf.contents().assumingMemoryBound(to: UInt16.self)
            for i in 0..<D {
                let v = unsafe Quantization.bf16ToFloat(src[i]) * invSqrtD
                unsafe dst[i] = Quantization.bf16Bits(v)
            }
            buf.label = "effective_scale.L\(L)"
            perLayer.append(buf)
        }
        self.effectiveScaleBuffers = perLayer
        self.earlyExpertReads = runtimeConfiguration.earlyExpertRead == .off
            ? nil
            : try NextLayerExpertPrefetcher(model: model,
                                            guess: runtimeConfiguration.earlyExpertRead,
                                            device: context.device)
    }

    /// Tokens currently in the KV cache.
    public var kvPosition: Int { kv.position }

    /// The runner's device, and a command buffer on its queue, for work
    /// that must run in order with the runner's own (a drafter's).
    var device: any MTLDevice { ctx.device }
    func makeCommandBuffer() throws -> any MTLCommandBuffer { try ctx.makeCommandBuffer() }

    public func reset() {
        kv.reset()
        prefillChunkState.markCommitted()
    }

    /// Maximum rows one `verify` call accepts.
    public static let maxVerifyTokens = LMHeadChainInt4.maxGreedyRows

    /// Runs `tokens` at the KV cursor as one block and returns the target's
    /// greedy next token after each row. The cursor advances by
    /// `tokens.count`; the caller rewinds rows it rejects.
    public func verify(tokens: [Int32], startPosition: Int) async throws -> [Int32] {
        guard !tokens.isEmpty, tokens.count <= Self.maxVerifyTokens else {
            throw PrefillError.chunkedUnsupported(
                "verify takes 1...\(Self.maxVerifyTokens) tokens, got \(tokens.count)")
        }
        let ids = unsafe verifyTokenIDs.contents().bindMemory(to: UInt32.self, capacity: tokens.count)
        for (i, token) in tokens.enumerated() { unsafe ids[i] = UInt32(bitPattern: token) }
        try await verifyBlock(rows: tokens.count, startPosition: startPosition,
                              tokens: verifyTokenIDs, greedy: verifyGreedyTokens)
        let values = unsafe verifyGreedyTokens.contents().bindMemory(to: UInt32.self, capacity: tokens.count)
        return (0..<tokens.count).map { unsafe Int32(bitPattern: values[$0]) }
    }

    /// One layer's resident weights, resolved once per runner.
    private struct LayerWeights {
        let isFull: Bool
        let headDim: Int
        let numKVHeads: Int
        let inputNorm, q, k, v, o, postAttention, preFFN, preFFN2, postFFN2, postFFN: TensorView
        let qNorm, kNorm, router, routerPerExpertScale: TensorView
        let layerScalar: Float
    }

    /// Runs a verify block of `rows` tokens at the KV cursor, reading their
    /// IDs from `tokens` on the GPU (so a drafter's output feeds it without a
    /// host copy), and leaves each row's greedy next token in `greedy` as
    /// `UInt32`. `behindHead`, when given, runs once the final head is
    /// committed, to queue dependent GPU work; the command buffer it returns,
    /// if any, is the one waited on instead of the head, so the caller's
    /// work and the verify complete under one host wait.
    ///
    /// The layers are pipelined: while the GPU runs a layer's attention, the
    /// host encodes that layer's tail and the next layer's attention and
    /// shared FFN, so once the routes are back the only work between the
    /// router and the next layer is planning and committing the routed tiles.
    func verifyBlock(rows t: Int, startPosition: Int, tokens: any MTLBuffer, greedy: any MTLBuffer,
                     behindHead: (() throws -> (any MTLCommandBuffer)?)? = nil) async throws {
        try prefillChunkState.requireClean(operation: "verify")
        guard t > 0, t <= Self.maxVerifyTokens else {
            throw PrefillError.chunkedUnsupported("verify takes 1...\(Self.maxVerifyTokens) rows, got \(t)")
        }
        guard kv.position == startPosition, startPosition + t <= maxContext else {
            throw PrefillError.chunkedUnsupported(
                "verify at \(startPosition) of \(t) rows; cursor \(kv.position), maxContext \(maxContext)")
        }
        try requireRingCapacity(rows: t)
        let scratch = try prefillScratch ?? ensurePrefillScratch(config: .production(chunkTokens: 32))
        guard t <= scratch.layout.chunkTokens else {
            throw PrefillError.chunkedUnsupported(
                "verify of \(t) rows exceeds the \(scratch.layout.chunkTokens)-row scratch")
        }
        let D = cfg.hiddenSize, topK = cfg.topKExperts, numExperts = cfg.numExperts
        let eps: Float = 1e-6
        let routeCount = t * topK
        prefillChunkState.markDirty(startPosition: startPosition, tokenCount: t)
        let traceBlock = GPUTrace.begin()
        defer {
            GPUTrace.end("verify \(t)", since: traceBlock)
            GPUTrace.collect()
        }

        func project(_ cb: any MTLCommandBuffer, _ w: TensorView, x: any MTLBuffer, y: any MTLBuffer,
                     rows: Int, columns: Int) throws {
            try int4.encodeRows(commandBuffer: cb,
                                weights: w.buffer, weightsOffset: Int(w.offset),
                                scales: w.buffer, scalesOffset: Int(w.scaleOffset),
                                biases: w.buffer, biasesOffset: Int(w.biasOffset),
                                x: x, y: y, m: UInt32(rows), n: UInt32(columns), rows: t,
                                xStrideElements: columns, yStrideElements: rows)
        }

        /// Layer `L` through its router, which signals `routerEvent`; the
        /// router also scores each row's guess at layer `L + 1`'s experts.
        func encodeAttention(_ L: Int) throws -> (cb: any MTLCommandBuffer, event: UInt64) {
            GPUTrace.setScope("verify L\(L)")
            model.beginOpeningRoutedExpertLayer(layer: L)
            let w = layers[L]
            let qDim = cfg.numHeads * w.headDim, kvDim = w.numKVHeads * w.headDim
            let cb = try ctx.makeCommandBuffer()
            cb.label = "verify rows=\(t) layer=\(L) phase=attention"
            if L == 0 {
                let emb = model.embedding
                prefillEmbed.encode(commandBuffer: cb,
                                    table: emb.buffer, tableOffset: Int(emb.offset),
                                    scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                                    biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                                    tokens: tokens, out: scratch.hidden,
                                    t: UInt32(t), d: UInt32(D), outScale: Float(D).squareRoot())
            }
            prefillRMS.encodeBF16W(commandBuffer: cb, x: scratch.hidden,
                                   weight: w.inputNorm.buffer, weightOffset: Int(w.inputNorm.offset),
                                   out: scratch.normed, t: UInt32(t), d: UInt32(D), eps: eps)
            try project(cb, w.q, x: scratch.normed, y: scratch.q, rows: qDim, columns: D)
            try project(cb, w.k, x: scratch.normed, y: scratch.kStage, rows: kvDim, columns: D)
            try project(cb, w.v, x: scratch.normed, y: scratch.vStage, rows: kvDim, columns: D)
            let theta = w.isFull ? Float(cfg.fullRopeTheta) : Float(cfg.ropeTheta)
            let rotatedPairs = ropePairs(w)
            // Decode's fused Q/K/V epilogue, one grid row per token.
            let fused = fusedQKVEpilogue.encodeRows(
                commandBuffer: cb,
                q: scratch.q, k: scratch.kStage, v: scratch.vStage,
                qWeight: w.qNorm.buffer, qWeightOffset: Int(w.qNorm.offset),
                kWeight: w.kNorm.buffer, kWeightOffset: Int(w.kNorm.offset),
                headDim: UInt32(w.headDim), numQHeads: UInt32(cfg.numHeads),
                numKVHeads: UInt32(w.numKVHeads), position: UInt32(startPosition), rows: t,
                qStrideElements: UInt32(qDim), kvStrideElements: UInt32(kvDim),
                theta: theta, rotatedPairs: rotatedPairs, eps: eps)
            if !fused {
                prefillQKVEpilogue.encode(commandBuffer: cb, q: scratch.q, k: scratch.kStage, v: scratch.vStage,
                                          qWeight: w.qNorm.buffer, qWeightOffset: Int(w.qNorm.offset),
                                          kWeight: w.kNorm.buffer, kWeightOffset: Int(w.kNorm.offset),
                                          startPosition: UInt32(startPosition), queryCount: UInt32(t),
                                          headDim: UInt32(w.headDim), numQHeads: UInt32(cfg.numHeads),
                                          numKVHeads: UInt32(w.numKVHeads),
                                          qTokenStrideElements: UInt32(qDim),
                                          kvTokenStrideElements: UInt32(kvDim),
                                          theta: theta, rotatedPairs: rotatedPairs, eps: eps)
            }
            try encodeKVCacheWrite(commandBuffer: cb, layer: L, startPosition: startPosition, tokenCount: t,
                                   keySource: scratch.kStage, valueSource: scratch.vStage,
                                   bytesPerToken: kvDim * MemoryLayout<Float16>.stride)
            try encodePrefillAttention(cb, layer: L, startPosition: startPosition, rows: t,
                                       q: scratch.q, out: scratch.attentionOutput, bidirectionalBlock: nil)
            try project(cb, w.o, x: scratch.attentionOutput, y: scratch.h1, rows: D, columns: qDim)
            encodePrefillPostAttention(cb, layer: L, rows: t, scratch: scratch)
            try encodeVerifyRouter(cb, layer: L, rows: t, scratch: scratch)
            routerEventValue &+= 1
            cb.encodeSignalEvent(routerEvent, value: routerEventValue)
            return (cb, routerEventValue)
        }

        /// The shared FFN, which needs only the attention's output and runs
        /// while the host plans the routed experts.
        func encodeShared(_ L: Int) throws -> any MTLCommandBuffer {
            let cb = try ctx.makeCommandBuffer()
            cb.label = "verify rows=\(t) layer=\(L) phase=shared_expert"
            try encodePrefillShared(cb, layer: L, rows: t, scratch: scratch, smallBatch: true)
            return cb
        }

        /// The routed experts' reduce, the layer tail and the drafter's
        /// feature copy; committed once the layer's routed tiles are.
        func encodeTail(_ L: Int) throws -> any MTLCommandBuffer {
            let cb = try ctx.makeCommandBuffer()
            cb.label = "verify rows=\(t) layer=\(L) phase=tail"
            encodePrefillTail(cb, layer: L, rows: t, scratch: scratch)
            try draftFeatureTap?.encodeCopy(commandBuffer: cb, layer: L, source: scratch.hidden,
                                            position: startPosition, rows: t)
            return cb
        }

        var submitted: [(any MTLCommandBuffer, ReferenceWritableKeyPath<RealForwardRunner, UInt64>?)] = []

        /// Binds the resident experts among `candidates` (clearing the rest)
        /// and queues layer `layer`'s GPU-routed experts behind its shared
        /// FFN, before its routes are known; the routes it does not cover are
        /// patched once they are.
        func bindRouted(layer: Int, candidates: inout [Bool]) throws {
            let traceBind = GPUTrace.begin()
            defer { GPUTrace.end("bind_next", since: traceBind) }
            GPUTrace.setScope("verify L\(layer)")
            let mapped = try model.expertLayer(layer)
            let addresses = try routedExpertAddresses(layer: layer, mapped: mapped)
            let listOffset = layer * numExperts * MemoryLayout<VerifyBoundExpert>.stride
            let list = unsafe (verifyBoundList.contents() + listOffset)
                .bindMemory(to: VerifyBoundExpert.self, capacity: numExperts)
            var resident: [any MTLBuffer] = []
            for e in 0..<numExperts where candidates[e] {
                if mapped.isResident(e, probes: Self.bindProbes) {
                    unsafe list[resident.count] = VerifyBoundExpert(address: addresses[e], expert: UInt32(e))
                    resident.append(mapped.buffers[e])
                } else {
                    candidates[e] = false
                }
            }
            guard !resident.isEmpty else { return }
            let cb = try ctx.makeCommandBuffer()
            cb.label = "verify rows=\(t) layer=\(layer) phase=routed_gpu"
            try prefillGroupedMoE.encodeVerifyBound(
                commandBuffer: cb, hidden: scratch.routedX, routeIDs: scratch.routeIDs,
                bound: verifyBoundList, boundOffset: listOffset,
                boundCount: resident.count, resident: resident, routePartials: scratch.routePartials,
                rows: t,
                params: PrefillGroupedRoutedMoEStreamedParams(
                    d: UInt32(D), routedIntermediate: UInt32(cfg.moeIntermediateSize), topK: UInt32(topK),
                    hiddenStrideElements: UInt32(D), offsets: model.routedExpertOffsets(layer: layer)))
            cb.commit()
            submitted.append((cb, \.prefillRoutedGPUNanos))
        }

        var attention = try encodeAttention(0)
        var shared = try encodeShared(0)
        attention.cb.commit()
        shared.commit()
        /// Experts of the next layer found resident by this layer's early read.
        var guessedResident = Set<Int>()
        var guesses = [UInt32](repeating: 0, count: routeCount)
        let guessPtr = unsafe chunkGuessIDs.contents().bindMemory(to: UInt32.self, capacity: routeCount)
        let guessing = earlyExpertReads != nil
        // The GPU routes a layer's experts itself when they are bound: by the
        // previous layer's guess, or for layer 0 every resident expert. A
        // lossy union cap keeps every layer on the host.
        let gpuRouted = guessing && gpuRoutedVerify && prefillExpertUnionCap == nil
        let guessLogits = unsafe moe.guessRowsLogits.contents().bindMemory(to: Float.self, capacity: t * numExperts)
        /// The experts this layer's GPU-routed pass binds; empty on the host path.
        var bound = [Bool](repeating: gpuRouted, count: numExperts)
        var nextBound = [Bool](repeating: false, count: numExperts)
        var scores = [Float](repeating: 0, count: numExperts)
        // Layer 0 has no previous layer to guess from: it binds every
        // resident expert while the GPU runs its attention.
        if gpuRouted { try bindRouted(layer: 0, candidates: &bound) }
        for L in 0..<cfg.numLayers {
            let last = L + 1 == cfg.numLayers
            // Encoded while the GPU runs this layer's attention.
            let traceEncode = GPUTrace.begin()
            let tail = try encodeTail(L)
            let nextAttention = last ? nil : try encodeAttention(L + 1)
            let nextShared = last ? nil : try encodeShared(L + 1)
            GPUTrace.end("encode_ahead", since: traceEncode)
            GPUTrace.setScope("verify L\(L)")
            try awaitRouter(attention.event, signaledBy: attention.cb)
            submitted.append((attention.cb, \.prefillPreRouterGPUNanos))
            submitted.append((shared, \.prefillSharedGPUNanos))

            // Everything the router left for the host, before the next
            // layer's router can overwrite it.
            let tracePlan = GPUTrace.begin()
            let (routeIDs, routeWeights) = readRoutes(scratch, rows: t)
            if guessing, !last {
                for i in 0..<routeCount { unsafe guesses[i] = guessPtr[i] }
            }
            if gpuRouted, !last {
                // The next layer's bound experts: the union of each row's
                // guessed best `verifyBindTop`.
                for e in 0..<numExperts { nextBound[e] = false }
                for r in 0..<t {
                    for e in 0..<numExperts { unsafe scores[e] = guessLogits[r * numExperts + e] }
                    for e in topIndices(scores, count: verifyBindTop) { nextBound[e] = true }
                }
            }
            let allPairs = PrefillRouter.makeTokenExpertPairs(indices: routeIDs, weights: routeWeights,
                                                              queryCount: t, topK: topK)
            prefillUnionExpertsAfterCap &+= UInt64(Set(allPairs.filter { $0.weight != 0 }.map(\.expert)).count)
            // The host runs the routes the GPU-routed pass did not bind. A
            // zero-weight route is run too, unless a cap dropped it, so every
            // partial the reduce reads is written.
            let capped = prefillExpertUnionCap != nil
            let pairs = allPairs.filter { !bound[Int($0.expert)] && (!capped || $0.weight != 0) }
            boundVerifyRoutes &+= UInt64(allPairs.count - allPairs.count { !bound[Int($0.expert)] })
            if capped, pairs.count < routeCount {
                // A dropped route leaves its partial unwritten; the tiles
                // write every other one.
                let clear = try ctx.makeCommandBuffer()
                guard let blit = clear.makeTracedBlitEncoder() else { throw MetalError.encoderUnavailable }
                blit.fill(buffer: scratch.routePartials,
                          range: 0..<(routeCount * D * MemoryLayout<Float16>.stride), value: 0)
                blit.endEncoding()
                clear.commit()
            }
            if pairs.isEmpty { GPUTrace.end("route_plan", since: tracePlan) }
            if !pairs.isEmpty {
                // Experts the previous layer's early read found resident are
                // not probed again.
                let plan = try planRoutedTiles(layer: L, pairs: pairs, rows: t,
                                               knownResident: guessedResident, probes: Self.bindProbes)
                GPUTrace.end("route_plan", since: tracePlan)
                let routedWallStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                patchedVerifyLayers &+= 1
                let routedOffsets = model.routedExpertOffsets(layer: L)
                let tileCBs = try await encodeRoutedTiles(layer: L, plan: plan) { tileCB, tile, binding, arguments in
                    tileCB.label = "verify rows=\(t) layer=\(L) phase=routed_tile"
                    let groups = plan.routes.groups[Int(tile.groupStart)..<Int(tile.groupStart + tile.groupCount)]
                    try prefillGroupedMoE.encodeVerifyRows(
                        commandBuffer: tileCB, hidden: scratch.routedX, sortedPairs: plan.metadata.sortedPairs,
                        routePartials: scratch.routePartials, argumentBuffer: arguments, binding: binding,
                        params: PrefillGroupedRoutedMoEStreamedParams(
                            pairStart: tile.pairStart, pairCount: tile.pairCount, d: UInt32(D),
                            routedIntermediate: UInt32(cfg.moeIntermediateSize), topK: UInt32(topK),
                            hiddenStrideElements: UInt32(D), binding: binding, offsets: routedOffsets),
                        segments: groups.map { SIMD2($0.pairStart, $0.pairCount) },
                        totalPairs: plan.routes.sortedPairs.count)
                }
                prefillRoutedWallNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - routedWallStart
                submitted += tileCBs.map { ($0, \.prefillRoutedGPUNanos) }
            }
            tail.commit()
            submitted.append((tail, nil))
            guard let nextAttention, let nextShared else { break }
            nextAttention.cb.commit()
            nextShared.commit()
            attention = nextAttention
            shared = nextShared
            swap(&bound, &nextBound)
            if gpuRouted { try bindRouted(layer: L + 1, candidates: &bound) }

            // This layer's reads are done, so storage would idle until the
            // next layer routes: advise each row's guessed top-8 next-layer
            // experts that are not in memory. A GPU-routed layer just probed
            // them all while binding.
            if guessing {
                let traceEarly = GPUTrace.begin()
                precondition(guesses.allSatisfy { $0 < numExperts }, "router guessed an expert past \(numExperts)")
                let likely = Set(guesses.map(Int.init))
                let absentNext = gpuRouted
                    ? likely.filter { !bound[$0] }.sorted()
                    : try model.nonResidentRoutedExperts(layer: L + 1, experts: likely.sorted(),
                                                         probes: Self.bindProbes)
                guessedResident = likely.subtracting(absentNext)
                // Advisory reads go out off the critical path.
                let nextLayer = try model.expertLayer(L + 1)
                Self.readAheadQueue.async { nextLayer.prefetch(absentNext) }
                prefillEarlyReads &+= UInt64(absentNext.count)
                GPUTrace.end("early_read_plan", since: traceEarly)
            }
        }

        GPUTrace.setScope("verify head")
        let headCB = try ctx.makeCommandBuffer()
        headCB.label = "verify rows=\(t) phase=head"
        try encodeGreedyRowsHead(headCB, hidden: scratch.hidden, hiddenOffset: 0, rows: t,
                                 outTokens: greedy,
                                 // Row r's token sits at startPosition + 1 + r.
                                 sampling: tokenSampling.map { .init(rule: $0, keyBase: startPosition + 1) })
        headCB.commit()
        if let behind = try behindHead?() {
            try waitForCompletion(behind)
            submitted.append((behind, nil))
        }
        try waitForCompletion(headCB)
        for (cb, counter) in submitted {
            try checkCommandBufferError(cb)
            if let counter {
                self[keyPath: counter] &+= UInt64(max(0, cb.gpuEndTime - cb.gpuStartTime) * 1e9)
            }
        }
        kv.advance(by: t)
        prefillChunkState.markCommitted()
    }

    /// Verify's router: decode's router kernels, one grid row per token, with
    /// each row's next-layer guess (for the early read) in the same dispatch.
    private func encodeVerifyRouter(_ cb: any MTLCommandBuffer, layer L: Int, rows t: Int,
                                    scratch: PrefillChunkScratchBuffers) throws {
        let D = UInt32(cfg.hiddenSize)
        let w = layers[L]
        let next = nextLayerGuess(after: L, outLogits: chunkGuessWeights)
        guard MoE.routesRows(numExperts: UInt32(cfg.numExperts), d: D, rows: t) else {
            throw PrefillError.chunkedUnsupported(
                "verify routing needs the row router: \(cfg.numExperts) experts, D \(D), \(t) rows")
        }
        let nextScale = next == nil ? nil : layers[L + 1].routerPerExpertScale
        moe.encodeRouterRowsGemma4(
            commandBuffer: cb,
            weights: w.router.buffer, weightsOffset: Int(w.router.offset),
            scales: w.router.buffer, scalesOffset: Int(w.router.scaleOffset),
            biases: w.router.buffer, biasesOffset: Int(w.router.biasOffset),
            hidden: scratch.routerX, hiddenStrideElements: D, rows: t,
            effectiveScale: effectiveScaleBuffers[L],
            perExpertScale: w.routerPerExpertScale.buffer,
            perExpertScaleOffset: Int(w.routerPerExpertScale.offset),
            outIndices: scratch.routeIDs, outWeights: scratch.routeWeights,
            numExperts: UInt32(cfg.numExperts), d: D,
            next: next,
            nextPerExpertScale: nextScale?.buffer, nextPerExpertScaleOffset: Int(nextScale?.offset ?? 0),
            nextIndices: chunkGuessIDs, nextWeights: chunkGuessWeights)
    }

    /// The early read's scoring of layer `L + 1`'s experts, which layer `L`'s
    /// router writes to `outLogits`; nil when the early read is off or `L` is
    /// the last layer.
    private func nextLayerGuess(after L: Int, outLogits: any MTLBuffer) -> MoE.RouterLogitsTarget? {
        guard let early = earlyExpertReads, L + 1 < cfg.numLayers else { return nil }
        if let guess = early.guessWeights {
            let m = guess.matrix(targetLayer: L + 1)
            return MoE.RouterLogitsTarget(
                weights: guess.buffer, weightsOffset: m.weightsOffset,
                scales: guess.buffer, scalesOffset: m.scalesOffset,
                biases: guess.buffer, biasesOffset: m.biasesOffset,
                effectiveScale: guess.unitScale, outLogits: outLogits)
        }
        let router = layers[L + 1].router
        return MoE.RouterLogitsTarget(
            weights: router.buffer, weightsOffset: Int(router.offset),
            scales: router.buffer, scalesOffset: Int(router.scaleOffset),
            biases: router.buffer, biasesOffset: Int(router.biasOffset),
            effectiveScale: effectiveScaleBuffers[L + 1], outLogits: outLogits)
    }

    /// The routes the router left in `scratch` for `rows` rows, with the
    /// union cap (if any) applied to the weights in place.
    private func readRoutes(_ scratch: PrefillChunkScratchBuffers, rows t: Int) -> (ids: [UInt32], weights: [Float16]) {
        let count = t * cfg.topKExperts
        let ids = unsafe Array(UnsafeBufferPointer(
            start: scratch.routeIDs.contents().bindMemory(to: UInt32.self, capacity: count), count: count))
        let weightPtr = unsafe scratch.routeWeights.contents().bindMemory(to: Float16.self, capacity: count)
        var weights = unsafe Array(UnsafeBufferPointer(start: weightPtr, count: count))
        precondition(ids.allSatisfy { $0 < cfg.numExperts }, "router wrote an expert past \(cfg.numExperts)")
        prefillUnionExpertsBeforeCap &+= UInt64(Set(ids).count)
        prefillUnionLayerTiles &+= 1
        if let cap = prefillExpertUnionCap {
            unsafe Self.capExpertUnion(ids: ids, weights: &weights, rows: t, topK: cfg.topKExperts,
                                       numExperts: cfg.numExperts, cap: cap, write: weightPtr)
        }
        return (ids, weights)
    }

    /// Keeps the `cap` experts with the most routed weight in a block and
    /// rescales each row's kept weights to its original total, writing the
    /// new weights back to `write` for the reduce.
    private static func capExpertUnion(ids: [UInt32], weights: inout [Float16], rows: Int, topK: Int,
                                       numExperts: Int, cap: Int,
                                       write: UnsafeMutablePointer<Float16>) {
        guard Set(ids).count > cap else { return }
        var mass = [Float](repeating: 0, count: numExperts)
        for index in ids.indices { mass[Int(ids[index])] += abs(Float(weights[index])) }
        let retained = Set(mass.indices.filter { mass[$0] > 0 }
            .sorted { mass[$0] != mass[$1] ? mass[$0] > mass[$1] : $0 < $1 }
            .prefix(cap).map(UInt32.init))
        for row in 0..<rows {
            let range = (row * topK)..<((row + 1) * topK)
            let original = range.reduce(Float(0)) { $0 + Float(weights[$1]) }
            let kept = range.reduce(Float(0)) { retained.contains(ids[$1]) ? $0 + Float(weights[$1]) : $0 }
            let scale = kept != 0 ? original / kept : 0
            for index in range {
                weights[index] = retained.contains(ids[index]) ? Float16(Float(weights[index]) * scale) : 0
                unsafe write[index] = weights[index]
            }
        }
    }

    /// Whether `rewind(to:)` can return to `position`: it must be cached, and
    /// the sliding-window rings must still hold its window.
    public func canRewind(to position: Int) -> Bool {
        position > 0 && position <= kv.position && kv.highWaterPosition - position <= kv.maxRewindTokens
    }

    public func canReuse(prefix position: Int) -> Bool { canRewind(to: position) }

    public func rewindForReuse(to position: Int) throws {
        hotHead = nil
        try rewind(to: position)
    }

    /// Abandons the newest KV tokens, e.g. rejected speculative drafts. Rows
    /// past the new cursor are never read and are overwritten on the next write.
    public func rewind(to position: Int) throws {
        guard canRewind(to: position) else {
            throw PrefillError.prefillCursorMismatch(
                "cannot rewind KV from \(kv.position) to \(position); "
                    + "high water \(kv.highWaterPosition), "
                    + "ring slack \(kv.maxRewindTokens)")
        }
        kv.rewind(to: position)
        prefillChunkState.markCommitted()
    }

    /// Host time reading absent routed experts in decode.
    public private(set) var decodeExpertReadNanos: UInt64 = 0
    public private(set) var prefillPreRouterGPUNanos: UInt64 = 0
    public private(set) var prefillSharedGPUNanos: UInt64 = 0
    public private(set) var prefillRoutedGPUNanos: UInt64 = 0
    public private(set) var totalCb1Nanos: UInt64 = 0
    public private(set) var totalCb2Nanos: UInt64 = 0
    public private(set) var totalHeadNanos: UInt64 = 0
    public private(set) var totalHeadFusedNanos: UInt64 = 0
    public private(set) var lastGreedyToken: UInt32 = 0
    /// When set, the fused heads (decode, prefill's last row, verify) draw
    /// under this rule instead of taking the argmax.
    public var tokenSampling: TokenSampling?
    public private(set) var routedExpertCacheHits: UInt64 = 0
    public private(set) var routedExpertCacheMisses: UInt64 = 0
    public private(set) var routedWeightMassByRank: [Double]
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

    public func produce(token: Int32, position: Int, into logits: any MTLBuffer) async throws {
        try prefillChunkState.requireClean(operation: "produce")
        try await produceToken(token: token, position: position, into: logits, outputMode: .greedyIfAvailable)
    }

    /// One decode step that writes the full logits whichever head the runner
    /// was built with; `LogitsHeadProducer` samples through it.
    func produceLogits(token: Int32, position: Int, into logits: any MTLBuffer) async throws {
        try prefillChunkState.requireClean(operation: "produce")
        try await produceToken(token: token, position: position, into: logits, outputMode: .logits)
    }

    public func prefillChunked(tokens: ArraySlice<Int32>,
                               startPosition: Int,
                               outputMode: PrefillOutputMode,
                               config: PrefillRuntimeConfig,
                               into logits: any MTLBuffer,
                               onProgress: (Int) -> Void) async throws -> PrefillResult {
        try prefillChunkState.requireClean(operation: "prefillChunked")
        guard config.mode == .chunked else {
            throw PrefillError.chunkedUnsupported(
                "prefillChunked requires PrefillRuntimeConfig.mode == .chunked")
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
        for (spanIndex, span) in spans.enumerated() {
            // A chunk in flight finishes; cancellation is observed between chunks.
            try Task.checkCancellation()
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
        return prefillResult(newPosition: startPosition + tokens.count, outputMode: outputMode)
    }

    public func prefillMultimodal(input: MultimodalPrefillInput,
                                  startPosition: Int,
                                  outputMode: PrefillOutputMode,
                                  config: PrefillRuntimeConfig,
                                  into logits: any MTLBuffer,
                                  onProgress: (Int) -> Void) async throws -> PrefillResult {
        try prefillChunkState.requireClean(operation: "prefillMultimodal")
        guard config.servesImagePrompt else {
            throw PrefillError.chunkedUnsupported(
                "multimodal prefill requires the complete chunked prefill path")
        }
        let tokens = input.embeddingTokenIDs
        guard startPosition + tokens.count <= maxContext else {
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

        // Text chunks are cut at the scratch layout's clamp.
        let textChunkTokens = min(config.chunkTokens, PrefillRuntimeConfig.maxChunkTokens)
        let work = PrefillChunkPlanner.multimodalWork(
            tokenCount: tokens.count,
            imageRanges: input.imageSpans.map(\.tokenRange),
            chunkTokens: config.chunkTokens)
        let imageBuffers = input.imageSpans.map(\.features.buffer)

        // One scratch layout for the whole turn, sized for its largest item.
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
        return prefillResult(newPosition: startPosition + tokens.count, outputMode: outputMode)
    }

    private func prefillResult(newPosition: Int, outputMode: PrefillOutputMode) -> PrefillResult {
        if outputMode == .greedyIfAvailable, usesFusedGreedyHead {
            return PrefillResult(newPosition: newPosition, seed: .greedyToken(lastGreedyToken))
        }
        return PrefillResult(newPosition: newPosition, seed: .logitsWritten)
    }

    /// Throws unless every sliding-window ring holds the window plus `rows`
    /// new rows.
    private func requireRingCapacity(rows: Int) throws {
        guard let ring = (0..<cfg.numLayers).map(kv.ringCapacity(layer:)).first(where: { $0 > 0 }) else {
            return
        }
        let required = min(maxContext, cfg.slidingWindow + rows)
        guard required <= ring else {
            throw PrefillError.chunkedUnsupported(
                "FP16 KV ring capacity \(ring) cannot hold required capacity \(required) for maxContext \(maxContext), slidingWindow \(cfg.slidingWindow), and \(rows) rows")
        }
    }

    /// Copies `tokenCount` rows of K and V at `startPosition` into layer
    /// `layer`'s cache, splitting at the ring's wrap.
    private func encodeKVCacheWrite(commandBuffer: any MTLCommandBuffer, layer: Int, startPosition: Int,
                                    tokenCount: Int, keySource: any MTLBuffer, valueSource: any MTLBuffer,
                                    bytesPerToken: Int) throws {
        guard tokenCount > 0 else { return }
        let capacity = kv.capacity(layer: layer)
        let firstSpan = min(tokenCount, capacity - startPosition % capacity)
        guard let blit = commandBuffer.makeTracedBlitEncoder() else { throw MetalError.encoderUnavailable }
        for (offset, count) in [(0, firstSpan), (firstSpan, tokenCount - firstSpan)] where count > 0 {
            let keys = kv.kRange(layer: layer, start: startPosition + offset, count: count)
            let values = kv.vRange(layer: layer, start: startPosition + offset, count: count)
            blit.copy(from: keySource, sourceOffset: offset * bytesPerToken, to: keys.buffer,
                      destinationOffset: keys.offset, size: count * bytesPerToken)
            blit.copy(from: valueSource, sourceOffset: offset * bytesPerToken, to: values.buffer,
                      destinationOffset: values.offset, size: count * bytesPerToken)
        }
        blit.endEncoding()
    }

    /// RoPE pairs per head: partial rotation on full layers, all of it on
    /// sliding-window layers.
    private func ropePairs(_ w: LayerWeights) -> UInt32 {
        w.isFull
            ? UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor / 2.0)
            : UInt32(w.headDim / 2)
    }

    /// Causal attention for `rows` query rows at `startPosition` over layer
    /// `L`'s cache; image rows in `bidirectionalBlock` also see each other.
    private func encodePrefillAttention(_ cb: any MTLCommandBuffer, layer L: Int, startPosition: Int, rows t: Int,
                                        q: any MTLBuffer, out: any MTLBuffer,
                                        bidirectionalBlock: Range<Int>?) throws {
        let w = layers[L]
        let qDim = cfg.numHeads * w.headDim, kvDim = w.numKVHeads * w.headDim
        let validCount = startPosition + t
        let ringCapacity = kv.ringCapacity(layer: L)
        try prefillAttention.encodeCausal(
            commandBuffer: cb, q: q,
            k: kv.keyBuffer(layer: L), v: kv.valueBuffer(layer: L), out: out,
            params: PrefillAttentionParams(
                startPosition: UInt32(startPosition), queryCount: UInt32(t),
                headDim: UInt32(w.headDim), numQHeads: UInt32(cfg.numHeads),
                numKVHeads: UInt32(w.numKVHeads), kvValidCount: UInt32(validCount),
                slidingWindow: w.isFull ? UInt32(validCount) : UInt32(cfg.slidingWindow),
                kvTokenStrideElements: UInt32(kvDim), qTokenStrideElements: UInt32(qDim),
                oTokenStrideElements: UInt32(qDim), scale: 1.0,
                bidirectionalBlockStart: UInt32(bidirectionalBlock?.lowerBound ?? 0),
                bidirectionalBlockEnd: UInt32(bidirectionalBlock?.upperBound ?? 0)),
            kvRingCapacity: ringCapacity > 0 && validCount > ringCapacity ? UInt32(ringCapacity) : 0,
            layerKind: w.isFull ? .full : .slidingWindow)
    }

    /// The residual add after attention (in `scratch.h1`) and the three
    /// FFN-input norms, for `rows` rows.
    private func encodePrefillPostAttention(_ cb: any MTLCommandBuffer, layer L: Int, rows t: Int,
                                            scratch: PrefillChunkScratchBuffers) {
        let w = layers[L]
        let D = UInt32(cfg.hiddenSize)
        prefillPostAttention.encode(commandBuffer: cb, hidden: scratch.hidden, attn: scratch.h1,
                                    denseX: scratch.denseX, routedX: scratch.routedX, routerX: scratch.routerX,
                                    postAttentionWeight: w.postAttention.buffer,
                                    postAttentionWeightOffset: Int(w.postAttention.offset),
                                    preFFNWeight: w.preFFN.buffer, preFFNWeightOffset: Int(w.preFFN.offset),
                                    preFFN2Weight: w.preFFN2.buffer, preFFN2WeightOffset: Int(w.preFFN2.offset),
                                    queryCount: UInt32(t), d: D,
                                    hiddenStrideElements: D, attnStrideElements: D,
                                    denseStrideElements: D, routedStrideElements: D,
                                    routerStrideElements: D, eps: 1e-6)
    }

    /// The shared FFN and its post-FFN-1 norm, `scratch.denseX` to `scratch.h1`.
    private func encodePrefillShared(_ cb: any MTLCommandBuffer, layer L: Int, rows t: Int,
                                     scratch: PrefillChunkScratchBuffers, smallBatch: Bool) throws {
        let D = cfg.hiddenSize
        let proj = sharedExpertProjections[L]
        try prefillSharedExpert.encodeBlock(commandBuffer: cb, x: scratch.denseX, y: scratch.h1,
                                            gate: proj.gate, up: proj.up, down: proj.down,
                                            scratchAct: scratch.sharedActScratch,
                                            queryCount: t, d: D, intermediate: cfg.intermediateSize,
                                            xStrideElements: D, yStrideElements: D, smallBatch: smallBatch)
        prefillRMS.encodeBF16W(commandBuffer: cb, x: scratch.h1,
                               weight: proj.postF1.buffer, weightOffset: Int(proj.postF1.offset),
                               out: scratch.h1, t: UInt32(t), d: UInt32(D), eps: 1e-6)
    }

    /// The routed experts' weighted reduce and the layer tail, updating
    /// `scratch.hidden`.
    private func encodePrefillTail(_ cb: any MTLCommandBuffer, layer L: Int, rows t: Int,
                                   scratch: PrefillChunkScratchBuffers) {
        let w = layers[L]
        let D = UInt32(cfg.hiddenSize)
        prefillMoE.encodeReduceTokenMajor(commandBuffer: cb, routePartials: scratch.routePartials,
                                          routeWeights: scratch.routeWeights, h2: scratch.h2,
                                          queryCount: UInt32(t), topK: UInt32(cfg.topKExperts), d: D)
        prefillLayerTail.encode(commandBuffer: cb, h2: scratch.h2, h1: scratch.h1, hidden: scratch.hidden,
                                postFFN2Weight: w.postFFN2.buffer, postFFN2WeightOffset: Int(w.postFFN2.offset),
                                postFFNWeight: w.postFFN.buffer, postFFNWeightOffset: Int(w.postFFN.offset),
                                queryCount: UInt32(t), d: D, h2StrideElements: D,
                                h1StrideElements: D, hiddenStrideElements: D,
                                eps: 1e-6, layerScalar: w.layerScalar)
    }

    /// The fused head over `rows` final hidden rows from `hiddenOffset`,
    /// leaving each row's token in `outTokens`.
    private func encodeGreedyRowsHead(_ cb: any MTLCommandBuffer, hidden: any MTLBuffer, hiddenOffset: Int,
                                      rows: Int, outTokens: any MTLBuffer,
                                      sampling: LMHeadChainInt4.RowSampling?) throws {
        let finalNorm = model.finalNorm, lm = model.lmHead
        try fusionHead.encodeGreedyRows(commandBuffer: cb, hidden: hidden, hiddenOffset: hiddenOffset,
                                        rowCount: rows,
                                        normWeight: finalNorm.buffer, normOffset: Int(finalNorm.offset),
                                        weights: lm.buffer, weightsOffset: Int(lm.offset),
                                        scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                                        biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                                        outTokens: outTokens, d: UInt32(cfg.hiddenSize),
                                        vocab: UInt32(cfg.vocabSize), rmsEps: 1e-6,
                                        softcap: Float(cfg.finalLogitSoftcap), sampling: sampling)
    }

    /// One layer's routed pairs grouped into tiles, resident experts' tiles
    /// first, and the experts that must be read before their tiles run.
    private struct RoutedTilePlan {
        let routes: PrefillMoEGroupedRoutes
        /// Kept alive until the tiles that read it complete.
        let metadata: PrefillGroupedRoutedMoEStreamedMetadataBuffers
        let absent: Set<Int>
    }

    /// Plans layer `L`'s routed tiles. Every expert has a fixed buffer, so
    /// the whole union binds at once; tiles of resident experts go first and
    /// run while the absent experts read, in tile order. Experts in
    /// `knownResident` are not probed.
    private func planRoutedTiles(layer L: Int, pairs: [PrefillTokenExpertPair], rows t: Int,
                                 knownResident: Set<Int> = [],
                                 probes: Int = MappedExpertLayer.residencyProbes) throws -> RoutedTilePlan {
        let unionExperts = Array(Set(pairs.map { Int($0.expert) })).sorted()
        let absent = Set(try model.nonResidentRoutedExperts(
            layer: L, experts: unionExperts.filter { !knownResident.contains($0) }, probes: probes))
        prefillTileHits &+= UInt64(unionExperts.count - absent.count)
        prefillTileMisses &+= UInt64(absent.count)
        var expertSortKeys = model.routedExpertPhysicalOffsets(layer: L)
        for e in expertSortKeys.indices {
            expertSortKeys[e] = absent.contains(e) ? expertSortKeys[e] + 1 : 0
        }
        let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs, queryCount: t, topK: cfg.topKExperts, numExperts: cfg.numExperts,
            tileExpertCount: 16, expertSortKeys: expertSortKeys)
        let metadata = try prefillGroupedMoE.makeStreamedMetadataBuffers(device: ctx.device, routes: routes)
        return RoutedTilePlan(routes: routes, metadata: metadata, absent: absent)
    }

    /// Commits `plan`'s tiles in order, each encoded by `encodeTile` once its
    /// absent experts have been read. One reader reads each tile's absent
    /// experts in tile order and reports each tile as it lands, so a tile
    /// waits only for its own experts and storage never idles between tiles.
    /// If a tile throws, reads not yet started are dropped and leaving the
    /// group waits for the one in flight.
    private func encodeRoutedTiles(
        layer L: Int, plan: RoutedTilePlan,
        encodeTile: (_ cb: any MTLCommandBuffer, _ tile: PrefillMoETile, _ binding: PrefillStreamedTileBinding,
                     _ arguments: PrefillStreamedTileArgumentBuffer) throws -> Void
    ) async throws -> [any MTLCommandBuffer] {
        let routes = plan.routes
        let expertsByTile = try routes.tiles.indices.map { tileIndex in
            try PrefillStreamedTileBinding.expertIDs(forTile: tileIndex, routes: routes)
        }
        let missingByTile = expertsByTile.map { $0.filter(plan.absent.contains) }
        let mapped = try model.expertLayer(L)
        var tileCBs: [any MTLCommandBuffer] = []
        try await withThrowingTaskGroup(of: Void.self) { reader in
            let (landed, landing) = AsyncThrowingStream.makeStream(of: Int.self)
            reader.addTask {
                do {
                    for (tileIndex, missing) in missingByTile.enumerated() where !missing.isEmpty {
                        try Task.checkCancellation()
                        let traceRead = GPUTrace.begin()
                        try await Model.read(missing, of: mapped)
                        GPUTrace.end("expert_read", since: traceRead, track: .io)
                        landing.yield(tileIndex)
                    }
                    landing.finish()
                } catch {
                    landing.finish(throwing: error)
                    throw error
                }
            }
            var arrivals = landed.makeAsyncIterator()
            var landedThrough = -1
            let argumentSlots = PrefillGroupedRoutedMoE.streamedArgumentBufferCount
            for (tileIndex, tile) in routes.tiles.enumerated() {
                if !missingByTile[tileIndex].isEmpty {
                    let fetchStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                    while landedThrough < tileIndex {
                        guard let landedTile = try await arrivals.next() else {
                            throw ModelError.indexCorrupt(detail: "routed tile \(tileIndex) was never read")
                        }
                        landedThrough = landedTile
                    }
                    prefillTileFetchNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - fetchStart
                    GPUTrace.end("tile_fetch_wait", since: fetchStart)
                }
                let traceTile = GPUTrace.begin()
                defer { GPUTrace.end("tile_encode", since: traceTile) }
                let expertIDs = expertsByTile[tileIndex]
                let binding = try PrefillStreamedTileBinding(
                    expertIDs: expertIDs, views: try model.routedExpertViews(layer: L, experts: expertIDs))
                if tileIndex >= argumentSlots {
                    // The argument buffer this tile reuses must be free.
                    let drainStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                    try waitForCompletion(tileCBs[tileIndex - argumentSlots])
                    prefillTileDrainNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - drainStart
                }
                let argumentBuffer = try prefillGroupedMoE.makeStreamedArgumentBuffer(
                    device: ctx.device, binding: binding, reuseSlot: tileIndex % argumentSlots)
                let tileCB = try ctx.makeCommandBuffer()
                try encodeTile(tileCB, tile, binding, argumentBuffer)
                tileCB.commit()
                tileCBs.append(tileCB)
            }
            try await reader.waitForAll()
        }
        return tileCBs
    }

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
                                     logits: any MTLBuffer,
                                     scratch: PrefillChunkScratchBuffers,
                                     config: PrefillRuntimeConfig,
                                     writeFinalHead: Bool,
                                     embeddingOverride: any MTLBuffer? = nil,
                                     bidirectionalBlock: Range<Int>? = nil) async throws {
        guard !tokens.isEmpty else { return }
        guard tokens.count <= scratch.layout.chunkTokens else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill token count \(tokens.count) exceeds scratch chunk size \(scratch.layout.chunkTokens)")
        }
        try requireRingCapacity(rows: config.chunkTokens)

        let tokenIDs = tokens.map { UInt32(bitPattern: $0) }
        guard let tokenBuffer = unsafe ctx.device.makeBuffer(bytes: tokenIDs,
                                                             length: tokenIDs.count * MemoryLayout<UInt32>.stride,
                                                             options: .storageModeShared) else {
            throw MetalError.allocationFailed("prefill token IDs")
        }
        let D = cfg.hiddenSize
        let eps: Float = 1e-6
        let t = tokens.count

        /// Long chunks take the tensor-core QMM when it is available, else
        /// the tiled QMM for K/V/O and repeated GEMV for Q; short chunks
        /// always run repeated GEMV.
        func encodeInt4Projection(_ cb: any MTLCommandBuffer, _ w: TensorView, x: any MTLBuffer, y: any MTLBuffer,
                                  rows: Int, columns: Int, isQ: Bool) {
            if t >= 32,
               prefillMPPAffineInt4.encode(commandBuffer: cb,
                                           weights: w.buffer, weightsOffset: Int(w.offset),
                                           scales: w.buffer, scalesOffset: Int(w.scaleOffset),
                                           biases: w.buffer, biasesOffset: Int(w.biasOffset),
                                           x: x, y: y, m: t, n: rows, k: columns) == .affineThreadgroupF16 {
                return
            }
            if t >= 32, !isQ {
                prefillQMM.encode(commandBuffer: cb,
                                  weights: w.buffer, weightsOffset: Int(w.offset),
                                  scales: w.buffer, scalesOffset: Int(w.scaleOffset),
                                  biases: w.buffer, biasesOffset: Int(w.biasOffset),
                                  x: x, y: y, t: t, n: rows, k: columns)
                return
            }
            for row in 0..<t {
                int4.encode(commandBuffer: cb,
                            weights: w.buffer, weightsOffset: Int(w.offset),
                            scales: w.buffer, scalesOffset: Int(w.scaleOffset),
                            biases: w.buffer, biasesOffset: Int(w.biasOffset),
                            x: x, xOffset: row * columns * MemoryLayout<Float16>.stride,
                            y: y, yOffset: row * rows * MemoryLayout<Float16>.stride,
                            m: UInt32(rows), n: UInt32(columns))
            }
        }

        prefillChunkState.markDirty(startPosition: startPosition, tokenCount: tokens.count)

        var cb = try ctx.makeCommandBuffer()
        cb.label = "prefill start=\(startPosition) count=\(t) phase=embed"
        if let embeddingOverride {
            // The planner emits image spans as standalone chunks, so the
            // override covers the whole chunk and replaces the INT4 gather.
            let bytes = t * D * MemoryLayout<Float16>.stride
            guard embeddingOverride.length >= bytes else {
                throw VisionRuntimeError.invalidInput(
                    "projected image feature buffer is too small")
            }
            guard let blit = cb.makeTracedBlitEncoder() else { throw MetalError.encoderUnavailable }
            blit.copy(from: embeddingOverride, sourceOffset: 0, to: scratch.hidden, destinationOffset: 0,
                      size: bytes)
            blit.endEncoding()
        } else {
            let emb = model.embedding
            prefillEmbed.encode(commandBuffer: cb,
                                table: emb.buffer, tableOffset: Int(emb.offset),
                                scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                                biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                                tokens: tokenBuffer, out: scratch.hidden,
                                t: UInt32(t), d: UInt32(D), outScale: Float(D).squareRoot())
        }

        let capture = prefillCapture
        try capture?.prepare(device: ctx.device, chunkTokens: scratch.layout.chunkTokens,
                             hiddenSize: D, numLayers: cfg.numLayers)

        struct PendingLayerTail {
            let attention: any MTLCommandBuffer
            let tail: any MTLCommandBuffer
            let shared: any MTLCommandBuffer
            let tiles: [any MTLCommandBuffer]
            /// Kept alive until the tiles that read it complete.
            let plan: RoutedTilePlan
        }
        var pendingTail: PendingLayerTail?
        func settle(_ pending: PendingLayerTail) throws {
            try waitForCompletion(pending.tail)
            try checkCommandBufferError(pending.attention)
            prefillPreRouterGPUNanos += UInt64(max(0, pending.attention.gpuEndTime - pending.attention.gpuStartTime) * 1e9)
            for tile in pending.tiles {
                try checkCommandBufferError(tile)
                prefillRoutedGPUNanos &+= UInt64(max(0, tile.gpuEndTime - tile.gpuStartTime) * 1e9)
            }
            try checkCommandBufferError(pending.shared)
            prefillSharedGPUNanos &+= UInt64(max(0, pending.shared.gpuEndTime - pending.shared.gpuStartTime) * 1e9)
        }

        let traceChunk = GPUTrace.begin()
        defer {
            GPUTrace.end("prefill \(t)", since: traceChunk)
            GPUTrace.collect()
        }
        for L in 0..<cfg.numLayers {
            GPUTrace.setScope("prefill L\(L)")
            let traceAttention = GPUTrace.begin()
            model.beginOpeningRoutedExpertLayer(layer: L)
            let w = layers[L]
            let qDim = cfg.numHeads * w.headDim
            let kvDim = w.numKVHeads * w.headDim

            prefillRMS.encodeBF16W(commandBuffer: cb, x: scratch.hidden,
                                   weight: w.inputNorm.buffer, weightOffset: Int(w.inputNorm.offset),
                                   out: scratch.normed, t: UInt32(t), d: UInt32(D), eps: eps)
            encodeInt4Projection(cb, w.q, x: scratch.normed, y: scratch.q, rows: qDim, columns: D, isQ: true)
            encodeInt4Projection(cb, w.k, x: scratch.normed, y: scratch.kStage, rows: kvDim, columns: D, isQ: false)
            encodeInt4Projection(cb, w.v, x: scratch.normed, y: scratch.vStage, rows: kvDim, columns: D, isQ: false)
            prefillQKVEpilogue.encode(commandBuffer: cb, q: scratch.q, k: scratch.kStage, v: scratch.vStage,
                                      qWeight: w.qNorm.buffer, qWeightOffset: Int(w.qNorm.offset),
                                      kWeight: w.kNorm.buffer, kWeightOffset: Int(w.kNorm.offset),
                                      startPosition: UInt32(startPosition), queryCount: UInt32(t),
                                      headDim: UInt32(w.headDim), numQHeads: UInt32(cfg.numHeads),
                                      numKVHeads: UInt32(w.numKVHeads),
                                      qTokenStrideElements: UInt32(qDim),
                                      kvTokenStrideElements: UInt32(kvDim),
                                      theta: w.isFull ? Float(cfg.fullRopeTheta) : Float(cfg.ropeTheta),
                                      rotatedPairs: ropePairs(w), eps: eps)
            try encodeKVCacheWrite(commandBuffer: cb, layer: L, startPosition: startPosition, tokenCount: t,
                                   keySource: scratch.kStage, valueSource: scratch.vStage,
                                   bytesPerToken: kvDim * MemoryLayout<Float16>.stride)
            try encodePrefillAttention(cb, layer: L, startPosition: startPosition, rows: t,
                                       q: scratch.q, out: scratch.attentionOutput,
                                       bidirectionalBlock: bidirectionalBlock)
            encodeInt4Projection(cb, w.o, x: scratch.attentionOutput, y: scratch.h1, rows: D, columns: qDim,
                                 isQ: false)
            encodePrefillPostAttention(cb, layer: L, rows: t, scratch: scratch)
            prefillRouter.encodeGemma4Block(
                commandBuffer: cb,
                weights: w.router.buffer, weightsOffset: Int(w.router.offset),
                scales: w.router.buffer, scalesOffset: Int(w.router.scaleOffset),
                biases: w.router.buffer, biasesOffset: Int(w.router.biasOffset),
                hidden: scratch.routerX,
                effectiveScale: effectiveScaleBuffers[L],
                perExpertScale: w.routerPerExpertScale.buffer,
                perExpertScaleOffset: Int(w.routerPerExpertScale.offset),
                outIndices: scratch.routeIDs, outWeights: scratch.routeWeights,
                queryCount: UInt32(t), numExperts: UInt32(cfg.numExperts), d: UInt32(D),
                topK: UInt32(cfg.topKExperts), hiddenStrideElements: UInt32(D))
            // Clear routed partials here, in a command buffer the host
            // completes, so routed guests on other queues never race it.
            // Only this chunk's `t * topK` rows are read.
            guard let clear = cb.makeTracedBlitEncoder() else { throw MetalError.encoderUnavailable }
            clear.fill(buffer: scratch.routePartials,
                       range: 0..<(t * cfg.topKExperts * D * MemoryLayout<Float16>.stride),
                       value: 0)
            clear.endEncoding()
            routerEventValue &+= 1
            cb.encodeSignalEvent(routerEvent, value: routerEventValue)
            cb.commit()
            GPUTrace.end("attention_encode", since: traceAttention)

            // The shared FFN needs only this command buffer's output, so it is
            // queued behind it and runs while the host plans the routed experts.
            let traceShared = GPUTrace.begin()
            let sharedCB = try ctx.makeCommandBuffer()
            sharedCB.label = "prefill start=\(startPosition) count=\(t) layer=\(L) phase=shared_expert"
            try encodePrefillShared(sharedCB, layer: L, rows: t, scratch: scratch, smallBatch: false)
            sharedCB.commit()
            GPUTrace.end("shared_encode", since: traceShared)
            try waitForCompletion(cb)
            // This layer's attention read the previous layer's output, so the
            // previous layer's tail, shared FFN and tiles are done.
            if let pending = pendingTail { try settle(pending); pendingTail = nil }

            let tracePlan = GPUTrace.begin()
            let (routeIDs, routeWeights) = readRoutes(scratch, rows: t)
            capture?.routes[L] = routeIDs
            let pairs = PrefillRouter.makeTokenExpertPairs(indices: routeIDs, weights: routeWeights,
                                                           queryCount: t, topK: cfg.topKExperts)
                .filter { $0.weight != 0 }
            prefillUnionExpertsAfterCap &+= UInt64(Set(pairs.map(\.expert)).count)
            let plan = try planRoutedTiles(layer: L, pairs: pairs, rows: t)
            let routedWallStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            var routedOnHost = true
            if let guest = supervisor.routedPrefill(layer: L) {
                let completed = try await supervisor.supervise(layer: L) {
                    try await guest.runLayer(model: model, layer: L, hidden: scratch.routedX,
                                             partials: scratch.routePartials, routes: plan.routes,
                                             pairs: plan.metadata.sortedPairs)
                }
                routedOnHost = !completed
                if routedOnHost {
                    // The guest drained before throwing. Every routed row is
                    // rewritten below; clearing first keeps a partial guest
                    // write from surviving a host skip.
                    let clearCB = try ctx.makeCommandBuffer()
                    guard let blit = clearCB.makeTracedBlitEncoder() else {
                        throw MetalError.encoderUnavailable
                    }
                    blit.fill(buffer: scratch.routePartials, range: 0..<scratch.routePartials.length, value: 0)
                    blit.endEncoding()
                    clearCB.commit()
                }
            }
            GPUTrace.end("route_plan", since: tracePlan)
            var tileCBs: [any MTLCommandBuffer] = []
            if routedOnHost {
                let routedOffsets = model.routedExpertOffsets(layer: L)
                tileCBs = try await encodeRoutedTiles(layer: L, plan: plan) { tileCB, tile, binding, arguments in
                    tileCB.label = "prefill start=\(startPosition) count=\(t) layer=\(L) phase=routed_tile"
                    _ = prefillGroupedMoE.encodeStreamedBatched(
                        commandBuffer: tileCB,
                        hidden: scratch.routedX,
                        sortedPairs: plan.metadata.sortedPairs,
                        routePartials: scratch.routePartials,
                        gateUpActScratch: scratch.routedGateUpActScratch,
                        downScratch: scratch.routedDownScratch,
                        argumentBuffer: arguments,
                        binding: binding,
                        params: PrefillGroupedRoutedMoEStreamedParams(
                            pairStart: tile.pairStart,
                            pairCount: tile.pairCount,
                            d: UInt32(D),
                            routedIntermediate: UInt32(cfg.moeIntermediateSize),
                            topK: UInt32(cfg.topKExperts),
                            hiddenStrideElements: UInt32(D),
                            binding: binding,
                            offsets: routedOffsets),
                        pairMicrobatchRows: scratch.layout.routedPairMicrobatchRows)
                }
            }
            prefillRoutedWallNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - routedWallStart
            let traceTail = GPUTrace.begin()
            let tailCB = try ctx.makeCommandBuffer()
            encodePrefillTail(tailCB, layer: L, rows: t, scratch: scratch)
            if let capture, let captureBuffer = capture.buffer,
               let slot = capture.hiddenLayers.firstIndex(of: L) {
                guard let blit = tailCB.makeTracedBlitEncoder() else { throw MetalError.encoderUnavailable }
                let rowBytes = D * MemoryLayout<Float16>.stride
                blit.copy(from: scratch.hidden, sourceOffset: 0, to: captureBuffer,
                          destinationOffset: slot * scratch.layout.chunkTokens * rowBytes,
                          size: t * rowBytes)
                blit.endEncoding()
            }
            try draftFeatureTap?.encodeCopy(commandBuffer: tailCB, layer: L, source: scratch.hidden,
                                            position: startPosition, rows: t)
            tailCB.commit()
            GPUTrace.end("tail_encode", since: traceTail)
            // Not waited here: the next layer's attention waits for it, as
            // decode does, which saves a host round trip per layer.
            pendingTail = PendingLayerTail(attention: cb, tail: tailCB, shared: sharedCB, tiles: tileCBs,
                                           plan: plan)
            if L + 1 < cfg.numLayers {
                cb = try ctx.makeCommandBuffer()
                cb.label = "prefill start=\(startPosition) count=\(t) layer=\(L + 1) phase=qkv_attention"
            }
        }
        if let pending = pendingTail { try settle(pending) }

        if writeFinalHead {
            let finalNorm = model.finalNorm
            let lm = model.lmHead
            GPUTrace.setScope("prefill head")
            let finalCB = try ctx.makeCommandBuffer()
            finalCB.label = "prefill start=\(startPosition) count=\(t) phase=final_head"
            let lastRowOffset = (t - 1) * D * MemoryLayout<Float16>.stride
            if outputMode == .greedyIfAvailable, usesFusedGreedyHead, let sampling = tokenSampling {
                // The prompt's next token sits at position startPosition + t.
                try encodeGreedyRowsHead(finalCB, hidden: scratch.hidden, hiddenOffset: lastRowOffset, rows: 1,
                                         outTokens: greedyTokenBuf,
                                         sampling: .init(rule: sampling, keyBase: startPosition + t))
            } else if outputMode == .greedyIfAvailable, usesFusedGreedyHead {
                fusionHead.encodeGreedyDecode(
                    commandBuffer: finalCB, hidden: scratch.hidden, hiddenOffset: lastRowOffset,
                    normWeight: finalNorm.buffer, normOffset: Int(finalNorm.offset),
                    weights: lm.buffer, weightsOffset: Int(lm.offset),
                    scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                    biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                    outToken: greedyTokenBuf, outTokenOffset: 0,
                    d: UInt32(D), vocab: UInt32(cfg.vocabSize), rmsEps: eps)
            } else {
                prefillFinalRowHead.encodeLogits(commandBuffer: finalCB, hiddenBlock: scratch.hidden,
                                                 row: t - 1, rowStrideElements: D,
                                                 normWeight: finalNorm.buffer,
                                                 normWeightOffset: Int(finalNorm.offset),
                                                 weights: lm.buffer, weightsOffset: Int(lm.offset),
                                                 scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                                                 biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                                                 logits: logits, d: UInt32(D), vocab: UInt32(cfg.vocabSize),
                                                 rmsEps: eps)
            }
            finalCB.commit()
            try waitForCompletion(finalCB)
            if outputMode == .greedyIfAvailable, usesFusedGreedyHead {
                lastGreedyToken = unsafe greedyTokenBuf.contents().load(as: UInt32.self)
            }
        }

        if let capture, let buffer = capture.buffer {
            // Compact the per-layer slots to `[layers][t][D]` for the consumer.
            let base = unsafe buffer.contents().assumingMemoryBound(to: Float16.self)
            let slotElements = scratch.layout.chunkTokens * D
            for slot in capture.hiddenLayers.indices where slot > 0 {
                unsafe memmove(base + slot * t * D, base + slot * slotElements, t * D * MemoryLayout<Float16>.stride)
            }
            try capture.onChunk(PrefillCaptureChunk(
                startPosition: startPosition, tokenCount: t, routes: capture.routes,
                storage: buffer, hiddenCount: capture.hiddenLayers.count * t * D))
        }

        kv.advance(by: tokens.count)
        prefillChunkState.markCommitted()
    }

    private func produceToken(token: Int32,
                              position: Int,
                              into logits: any MTLBuffer,
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
        let numExperts = cfg.numExperts
        let topK = UInt32(activeRoutedExperts)
        // The GPU routes a layer itself when it can look up every routed
        // expert's address; otherwise every layer takes the host patch.
        let gpuRouted = gpuRoutedDecode && activeRoutedExperts == cfg.topKExperts

        /// One decode layer, encoded while the GPU runs the layer before it
        /// and committed once that layer's routes are known to be bound, so
        /// the GPU never waits for the host between layers.
        struct LayerWork {
            let layer: Int
            /// Through the router; signals `event` when the routes are ready.
            let attention: any MTLCommandBuffer
            let event: UInt64
            let shared: any MTLCommandBuffer
            /// The routed experts, looked up on the GPU, and the layer tail;
            /// nil on the host path.
            var routed: (any MTLCommandBuffer)?
            /// Experts `routed` binds; a route outside it is patched.
            var bound: [Bool] = []
        }
        var committed: [(any MTLCommandBuffer, ReferenceWritableKeyPath<RealForwardRunner, UInt64>)] = []

        func encodeTail(_ cb: any MTLCommandBuffer, layer L: Int) throws {
            let w = layers[L]
            if fuseTailNorms {
                // Also writes the next layer's normalized input to `normed`.
                fusedTail.encodeWithNorms(commandBuffer: cb, h2: h2Buf, h1: h1Buf, hidden: hidden,
                                          postFFN1Weight: sharedExpertProjections[L].postF1,
                                          postFFN2Weight: w.postFFN2,
                                          postFFNWeight: w.postFFN,
                                          next: L + 1 < cfg.numLayers
                                              ? (inputNorm: layers[L + 1].inputNorm, normed: normed) : nil,
                                          d: D, eps: eps, layerScalar: w.layerScalar)
            } else {
                fusedTail.encode(commandBuffer: cb, h2: h2Buf, h1: h1Buf, hidden: hidden,
                                 postFFN2Weight: w.postFFN2.buffer,
                                 postFFN2WeightOffset: Int(w.postFFN2.offset),
                                 postFFNWeight: w.postFFN.buffer,
                                 postFFNWeightOffset: Int(w.postFFN.offset),
                                 d: D, eps: eps, layerScalar: w.layerScalar)
            }
            try draftFeatureTap?.encodeCopy(commandBuffer: cb, layer: L, source: hidden,
                                            position: position, rows: 1)
        }

        func encodeLayer(_ L: Int) throws -> LayerWork {
            GPUTrace.setScope("decode L\(L)")
            let tEncode = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            defer { totalCb1Nanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tEncode }
            let w = layers[L]
            let qDim = UInt32(cfg.numHeads * w.headDim)
            let kvDim = UInt32(w.numKVHeads * w.headDim)
            let kSlot = kv.kSlot(layer: L, position: position)
            let vSlot = kv.vSlot(layer: L, position: position)
            let seqLen = UInt32(position + 1)
            let sharedProj = sharedExpertProjections[L]

            let cb = try ctx.makeCommandBuffer()
            if L == 0 {
                // Embed lookup + sqrt(H) fused.
                let emb = model.embedding
                embedInt4.encode(commandBuffer: cb,
                                 table:  emb.buffer, tableOffset:  Int(emb.offset),
                                 scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                                 biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                                 out: hidden,
                                 tokenId: UInt32(bitPattern: token),
                                 d: D,
                                 outScale: Float(cfg.hiddenSize).squareRoot())
            }
            // Past layer 0 the previous layer's tail wrote `normed`.
            if L == 0 || !fuseTailNorms {
                rms.encodeBF16W(commandBuffer: cb, x: hidden,
                                weight: w.inputNorm.buffer, weightOffset: Int(w.inputNorm.offset),
                                out: normed, d: D, eps: eps)
            }
            fusedQKVGEMV.encode(commandBuffer: cb,
                                qWeights: w.q.buffer, qWeightsOffset: Int(w.q.offset),
                                qScales: w.q.buffer, qScalesOffset: Int(w.q.scaleOffset),
                                qBiases: w.q.buffer, qBiasesOffset: Int(w.q.biasOffset),
                                kWeights: w.k.buffer, kWeightsOffset: Int(w.k.offset),
                                kScales: w.k.buffer, kScalesOffset: Int(w.k.scaleOffset),
                                kBiases: w.k.buffer, kBiasesOffset: Int(w.k.biasOffset),
                                vWeights: w.v.buffer, vWeightsOffset: Int(w.v.offset),
                                vScales: w.v.buffer, vScalesOffset: Int(w.v.scaleOffset),
                                vBiases: w.v.buffer, vBiasesOffset: Int(w.v.biasOffset),
                                x: normed,
                                qOut: qScratch,
                                kOut: kSlot.buffer, kOutOffset: kSlot.offset,
                                vOut: vSlot.buffer, vOutOffset: vSlot.offset,
                                qRows: qDim,
                                kvRows: kvDim,
                                n: D)
            fusedQKVEpilogue.encode(commandBuffer: cb,
                                    q: qScratch,
                                    k: kSlot.buffer,
                                    kOffset: kSlot.offset,
                                    v: vSlot.buffer,
                                    vOffset: vSlot.offset,
                                    qWeight: w.qNorm.buffer,
                                    qWeightOffset: Int(w.qNorm.offset),
                                    kWeight: w.kNorm.buffer,
                                    kWeightOffset: Int(w.kNorm.offset),
                                    headDim: UInt32(w.headDim),
                                    numQHeads: UInt32(cfg.numHeads),
                                    numKVHeads: UInt32(w.numKVHeads),
                                    position: UInt32(position),
                                    theta: w.isFull ? Float(cfg.fullRopeTheta) : Float(cfg.ropeTheta),
                                    rotatedPairs: ropePairs(w),
                                    eps: eps)
            if w.isFull {
                attention.encodeFull(commandBuffer: cb,
                                     q: qScratch,
                                     k: kSlot.buffer, kOffset: 0,
                                     v: vSlot.buffer, vOffset: 0,
                                     out: attnOut,
                                     headDim: UInt32(w.headDim),
                                     numQHeads: UInt32(cfg.numHeads),
                                     numKVHeads: UInt32(w.numKVHeads),
                                     seqLen: seqLen,
                                     scale: 1.0)
            } else {
                let ringCapacity = kv.ringCapacity(layer: L)
                attention.encodeSWA(commandBuffer: cb,
                                    q: qScratch,
                                    k: kSlot.buffer, kOffset: 0,
                                    v: vSlot.buffer, vOffset: 0,
                                    out: attnOut,
                                    headDim: UInt32(w.headDim),
                                    numQHeads: UInt32(cfg.numHeads),
                                    numKVHeads: UInt32(w.numKVHeads),
                                    seqLen: seqLen,
                                    window: UInt32(cfg.slidingWindow),
                                    scale: 1.0,
                                    ringCapacity: ringCapacity > 0 && Int(seqLen) > ringCapacity
                                        ? UInt32(ringCapacity) : 0)
            }
            int4.encode(commandBuffer: cb,
                        weights: w.o.buffer, weightsOffset: Int(w.o.offset),
                        scales:  w.o.buffer, scalesOffset:  Int(w.o.scaleOffset),
                        biases:  w.o.buffer, biasesOffset:  Int(w.o.biasOffset),
                        x: attnOut, y: oOut, m: D, n: qDim)
            fusedPostAttentionSetup.encode(commandBuffer: cb,
                                           hidden: hidden,
                                           attn: oOut,
                                           denseX: denseX,
                                           routedX: routedX,
                                           routerX: routerInput,
                                           postAttentionWeight: w.postAttention.buffer,
                                           postAttentionWeightOffset: Int(w.postAttention.offset),
                                           preFFNWeight: w.preFFN.buffer,
                                           preFFNWeightOffset: Int(w.preFFN.offset),
                                           preFFN2Weight: w.preFFN2.buffer,
                                           preFFN2WeightOffset: Int(w.preFFN2.offset),
                                           d: D,
                                           eps: eps)
            // The router also scores layer L+1's experts for the early read.
            moe.encodeRouterGemma4(commandBuffer: cb,
                weights: w.router.buffer, weightsOffset: Int(w.router.offset),
                scales:  w.router.buffer, scalesOffset:  Int(w.router.scaleOffset),
                biases:  w.router.buffer, biasesOffset:  Int(w.router.biasOffset),
                hidden: routerInput,
                effectiveScale: effectiveScaleBuffers[L],
                perExpertScale: w.routerPerExpertScale.buffer,
                perExpertScaleOffset: Int(w.routerPerExpertScale.offset),
                outIndices: outIndices, outWeights: outWeights,
                numExperts: UInt32(numExperts), d: D, topK: UInt32(cfg.topKExperts),
                next: nextLayerGuess(after: L, outLogits: nextLayerScores))
            routerEventValue &+= 1
            cb.encodeSignalEvent(routerEvent, value: routerEventValue)

            // The shared dense MLP depends only on denseX; it runs while the
            // host checks the routes.
            let sharedCB = try ctx.makeCommandBuffer()
            try shared.encode(commandBuffer: sharedCB,
                              x: denseX,
                              gate: sharedProj.gate,
                              up: sharedProj.up,
                              down: sharedProj.down,
                              y: h1Buf,
                              scratchAct: denseScratchAct)
            // A norm-fusing tail normalizes the raw shared output itself.
            if !fuseTailNorms {
                rms.encodeBF16W(commandBuffer: sharedCB, x: h1Buf,
                                weight: sharedProj.postF1.buffer,
                                weightOffset: Int(sharedProj.postF1.offset),
                                out: h1Buf, d: D, eps: eps)
            }

            return LayerWork(layer: L, attention: cb, event: routerEventValue, shared: sharedCB)
        }

        /// Adds the GPU-routed experts and tail to `work`, binding the
        /// resident experts among the guess's best `bindTop`.
        func encodeRouted(_ work: LayerWork, guess: [Float]) throws -> LayerWork {
            let tEncode = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            defer { totalCb1Nanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tEncode }
            let L = work.layer
            GPUTrace.setScope("decode L\(L)")
            let mapped = try model.expertLayer(L)
            let addresses = try routedExpertAddresses(layer: L, mapped: mapped)
            let table = unsafe routedAddressTable.contents().assumingMemoryBound(to: UInt64.self)
                + L * numExperts
            var bound = [Bool](repeating: false, count: numExperts)
            var resident: [any MTLBuffer] = [routedFallback]
            for e in topIndices(guess, count: Self.bindTop) where mapped.isResident(e, probes: Self.bindProbes) {
                bound[e] = true
                resident.append(mapped.buffers[e])
            }
            for e in 0..<numExperts { unsafe table[e] = bound[e] ? addresses[e] : 0 }
            let routedCB = try ctx.makeCommandBuffer()
            moe.encodeGatherRoutedSlots(commandBuffer: routedCB,
                                        indices: outIndices,
                                        table: routedAddressTable,
                                        tableOffset: L * numExperts * MemoryLayout<UInt64>.stride,
                                        slots: routedSlots,
                                        hidden: hidden, saved: savedHidden,
                                        topK: topK, numExperts: UInt32(numExperts), d: D,
                                        fallback: routedFallback.gpuAddress)
            moe.encodeRoutedPersistentGPUSlots(commandBuffer: routedCB,
                                               slots: routedSlots,
                                               resident: resident,
                                               routedOffsets: model.routedExpertOffsets(layer: L),
                                               x: routedX,
                                               acts: moeActs,
                                               routingWeights: outWeights,
                                               residual: zeroResidual,
                                               y: h2Buf,
                                               d: D, f: FmoE, topK: topK)
            try encodeTail(routedCB, layer: L)
            var routedWork = work
            routedWork.routed = routedCB
            routedWork.bound = bound
            return routedWork
        }

        func commit(_ work: LayerWork) {
            work.attention.commit()
            work.shared.commit()
            committed.append((work.attention, \.decodeAttentionGPUNanos))
            committed.append((work.shared, \.decodeSharedGPUNanos))
        }

        func commitRouted(_ work: LayerWork) {
            guard let routed = work.routed else { return }
            routed.commit()
            committed.append((routed, \.decodeRoutedGPUNanos))
        }

        /// Runs the layer's routed experts from the host's reading of the
        /// routes, after the GPU-routed pass (if any) found one unbound.
        func patch(_ work: LayerWork, experts: [Int], absent: [Int]) async throws {
            let L = work.layer
            GPUTrace.setScope("decode L\(L) patch")
            let tracePatch = GPUTrace.begin()
            defer { GPUTrace.end("patch_encode", since: tracePatch) }
            if !absent.isEmpty {
                let tIoStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                try await model.readRoutedExperts(layer: L, experts: absent)
                decodeExpertReadNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tIoStart
                GPUTrace.end("expert_read", since: tIoStart, track: .io)
            }
            let tEncode = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            if activeRoutedExperts < cfg.topKExperts {
                let weights = unsafe outWeights.contents().assumingMemoryBound(to: Float16.self)
                let full = (0..<cfg.topKExperts).reduce(Float(0)) { unsafe $0 + Float(weights[$1]) }
                let kept = (0..<activeRoutedExperts).reduce(Float(0)) { unsafe $0 + Float(weights[$1]) }
                if kept != 0 {
                    for rank in 0..<activeRoutedExperts {
                        unsafe weights[rank] = unsafe Float16(Float(weights[rank]) * full / kept)
                    }
                }
            }
            let views = try model.routedExpertViews(layer: L, experts: experts)
            let blobs = views.map(\.buffer)
            let offsets = model.routedExpertOffsets(layer: L)
            let argBuf = moe.makeReusedRoutedArgumentBuffer(routedViews: views, topK: topK)
            let cb = try ctx.makeCommandBuffer()
            if work.routed != nil {
                // The GPU-routed pass already ran the tail; restart from the
                // layer input it saved.
                guard let blit = cb.makeTracedBlitEncoder() else { throw MetalError.encoderUnavailable }
                blit.copy(from: savedHidden, sourceOffset: 0, to: hidden, destinationOffset: 0,
                          size: cfg.hiddenSize * MemoryLayout<Float16>.stride)
                blit.endEncoding()
                // It also left correct activations in the bound slots;
                // recompute only the unbound ones, then the reduce.
                let slots = (0..<experts.count).filter { !work.bound[experts[$0]] }.map(UInt32.init)
                let slotOffset = L * cfg.topKExperts * MemoryLayout<UInt32>.stride
                let slotPtr = unsafe (patchSlots.contents() + slotOffset).assumingMemoryBound(to: UInt32.self)
                for (i, slot) in slots.enumerated() { unsafe slotPtr[i] = slot }
                moe.encodeRoutedPersistentPhase1SubsetU16Load(commandBuffer: cb,
                                                              routedArgBuffer: argBuf,
                                                              routedBlobs: blobs,
                                                              routedOffsets: offsets,
                                                              x: routedX,
                                                              acts: moeActs,
                                                              activeSlots: patchSlots,
                                                              activeSlotsOffset: slotOffset,
                                                              activeSlotIndices: slots,
                                                              activeCount: UInt32(slots.count),
                                                              d: D, f: FmoE, topK: topK)
            } else {
                moe.encodeRoutedPersistentPhase1U16Load(commandBuffer: cb,
                                                        routedArgBuffer: argBuf,
                                                        routedBlobs: blobs,
                                                        routedOffsets: offsets,
                                                        x: routedX,
                                                        acts: moeActs,
                                                        d: D, f: FmoE, topK: topK)
            }
            moe.encodeRoutedPersistentPhase2Reduce(commandBuffer: cb,
                                                   routedArgBuffer: argBuf,
                                                   routedBlobs: blobs,
                                                   routedOffsets: offsets,
                                                   acts: moeActs,
                                                   routingWeights: outWeights,
                                                   residual: zeroResidual,
                                                   y: h2Buf,
                                                   d: D, f: FmoE, topK: topK)
            try encodeTail(cb, layer: L)
            cb.commit()
            committed.append((cb, \.decodeRoutedGPUNanos))
            totalCb2Nanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tEncode
        }

        let traceToken = GPUTrace.begin()
        earlyExpertReads?.beginForward()
        var scores = [Float](repeating: 0, count: numExperts)
        var work = try encodeLayer(0)
        commit(work)
        for L in 0..<cfg.numLayers {
            let next = L + 1 < cfg.numLayers ? try encodeLayer(L + 1) : nil
            try awaitRouter(work.event, signaledBy: work.attention)
            // Read everything this layer left for the host before the next
            // layer's buffers can overwrite it.
            let idxPtr = unsafe outIndices.contents().assumingMemoryBound(to: UInt32.self)
            let weightPtr = unsafe outWeights.contents().assumingMemoryBound(to: Float16.self)
            for rank in 0..<cfg.topKExperts {
                unsafe routedWeightMassByRank[rank] += Double(weightPtr[rank])
            }
            routedWeightMassSamples &+= 1
            let experts = (0..<activeRoutedExperts).map { unsafe Int(idxPtr[$0]) }
            precondition(experts.allSatisfy { $0 < numExperts }, "router wrote an expert past \(numExperts)")
            if earlyExpertReads != nil, next != nil {
                let p = unsafe nextLayerScores.contents().assumingMemoryBound(to: Float.self)
                for e in 0..<numExperts { scores[e] = unsafe p[e] }
            }
            let unbound = work.routed == nil ? experts : experts.filter { !work.bound[$0] }
            let absent = unbound.isEmpty ? [] : try model.nonResidentRoutedExperts(layer: L, experts: unbound)
            routedExpertCacheHits &+= UInt64(experts.count - absent.count)
            routedExpertCacheMisses &+= UInt64(absent.count)
            earlyExpertReads?.observe(layer: L, experts: experts, absent: absent.count)
            if !unbound.isEmpty {
                decodeLayerPatches &+= 1
                try await patch(work, experts: experts, absent: absent)
            }
            guard var next else { break }
            commit(next)
            // The GPU has this layer's experts and the next layer's attention
            // queued; bind the next layer's guessed experts behind them.
            if gpuRouted, earlyExpertReads != nil {
                next = try encodeRouted(next, guess: scores)
                commitRouted(next)
            }
            // This layer's reads are done, so storage is idle until the next
            // layer's: advise the next layer's likeliest absent experts now.
            if let early = earlyExpertReads {
                let layer = try model.expertLayer(L + 1)
                let chosen = NextLayerExpertPrefetcher.choose(
                    scores: scores, limit: early.readsPerLayer) { !layer.isResident($0) }
                if !chosen.isEmpty {
                    let traceIssue = GPUTrace.begin()
                    try early.issue(layer: L + 1, experts: chosen, model: model)
                    GPUTrace.end("early_read_issue", since: traceIssue)
                }
            }
            work = next
        }

        // The fused head skips the vocab buffer and leaves a greedy token in
        // greedyTokenBuf; the logits path writes the complete vector.
        let fNorm = model.finalNorm
        let lm = model.lmHead
        GPUTrace.setScope("decode head")
        let tHead = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        if usesFusedGreedyHead && outputMode == .greedyIfAvailable, let sampling = tokenSampling {
            // The full head's logits and a draw for the token at position + 1.
            try runSync { cb in
                try self.encodeGreedyRowsHead(cb, hidden: self.hidden, hiddenOffset: 0, rows: 1,
                                              outTokens: self.greedyTokenBuf,
                                              sampling: .init(rule: sampling, keyBase: position + 1))
            }
            totalHeadFusedNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
            lastGreedyToken = unsafe greedyTokenBuf.contents().load(as: UInt32.self)
        } else if usesFusedGreedyHead && outputMode == .greedyIfAvailable, let (row, check) = hotHead,
           let hot = hotVocabulary {
            try runSync { cb in
                if check {
                    self.fusionHead.encodeNormRow(
                        commandBuffer: cb, hidden: self.hidden,
                        normWeight: fNorm.buffer, normOffset: Int(fNorm.offset),
                        normedRows: self.headCheckRows, row: row, d: D, rmsEps: eps)
                    self.fusionHead.encodeGreedyCheck(
                        commandBuffer: cb, normedRows: self.headCheckRows, rowCount: row + 1,
                        weights: lm.buffer, weightsOffset: Int(lm.offset),
                        scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                        biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                        outTokens: self.headCheckTokens, d: D, vocab: UInt32(self.cfg.vocabSize))
                } else {
                    self.fusionHead.encodeHotGreedyDecode(
                        commandBuffer: cb, hidden: self.hidden,
                        normWeight: fNorm.buffer, normOffset: Int(fNorm.offset), table: hot,
                        weights: lm.buffer, weightsOffset: Int(lm.offset),
                        scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                        biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                        normedRows: self.headCheckRows, row: row,
                        outToken: self.greedyTokenBuf, d: D, rmsEps: eps)
                }
            }
            totalHeadFusedNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
            if check {
                let out = unsafe headCheckTokens.contents().bindMemory(to: UInt32.self, capacity: row + 1)
                checkedTokens = (0...row).map { unsafe out[$0] }
                lastGreedyToken = checkedTokens[row]
            } else {
                lastGreedyToken = unsafe greedyTokenBuf.contents().load(as: UInt32.self)
            }
        } else if usesFusedGreedyHead && outputMode == .greedyIfAvailable {
            try runSync { cb in
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
            totalHeadFusedNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
            lastGreedyToken = unsafe greedyTokenBuf.contents().load(as: UInt32.self)
        } else {
            try runSync { cb in
                self.rms.encodeBF16W(commandBuffer: cb, x: self.hidden,
                                     weight: fNorm.buffer, weightOffset: Int(fNorm.offset),
                                     out: self.normed, d: D, eps: eps)
                self.int4.encode(commandBuffer: cb,
                                 weights: lm.buffer, weightsOffset: Int(lm.offset),
                                 scales:  lm.buffer, scalesOffset:  Int(lm.scaleOffset),
                                 biases:  lm.buffer, biasesOffset:  Int(lm.biasOffset),
                                 x: self.normed, y: logits, m: UInt32(self.cfg.vocabSize), n: D)
            }
            totalHeadNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
        }
        // Everything before the head has finished on the queue.
        for (cb, counter) in committed {
            waitUntilCompleted(cb)
            try checkCommandBufferError(cb)
            self[keyPath: counter] &+= UInt64(max(0, cb.gpuEndTime - cb.gpuStartTime) * 1e9)
        }
        GPUTrace.end("decode_token", since: traceToken)
        GPUTrace.collect()
        kv.advance()
    }

    /// Indices of the `count` largest values, largest first.
    private func topIndices(_ values: [Float], count: Int) -> [Int] {
        Array(values.indices.sorted { values[$0] > values[$1] }.prefix(count))
    }

    /// Experts' GPU addresses for layer `layer`, cached while `mapped` is the
    /// layer's open mapping. A reopened layer maps at new addresses.
    private func routedExpertAddresses(layer: Int, mapped: MappedExpertLayer) throws -> [UInt64] {
        if let cached = routedExpertAddressCache[layer], cached.layer === mapped { return cached.addresses }
        let addresses = try model.routedExpertViews(layer: layer, experts: Array(0..<cfg.numExperts))
            .map { $0.buffer.gpuAddress + $0.offset }
        routedExpertAddressCache[layer] = ExpertAddresses(layer: mapped, addresses: addresses)
        return addresses
    }

    /// How long `awaitRouter` spins before it blocks.
    private static let routerSpinNanos: UInt64 = 5_000_000

    /// Waits for `routerEvent` to reach `value`, which `cb` signals. Spins
    /// first, since a sleeping wait's thread wake outlasts most of the wait;
    /// past `routerSpinNanos` it blocks on `cb`, so a stalled GPU cannot pin
    /// a core.
    private func awaitRouter(_ value: UInt64, signaledBy cb: any MTLCommandBuffer) throws {
        let traceStart = GPUTrace.begin()
        defer { GPUTrace.end("router_wait", since: traceStart) }
        let deadline = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) + Self.routerSpinNanos
        while routerEvent.signaledValue < value {
            if cb.status == .error { try checkCommandBufferError(cb) }
            if clock_gettime_nsec_np(CLOCK_UPTIME_RAW) > deadline {
                try waitForCompletion(cb)
                return
            }
        }
    }

    private func runSync(_ body: (any MTLCommandBuffer) throws -> Void) throws {
        let cb = try ctx.makeCommandBuffer()
        try body(cb)
        cb.commit()
        try waitForCompletion(cb)
    }

    private nonisolated func waitForCompletion(_ cb: any MTLCommandBuffer) throws {
        waitUntilCompleted(cb)
        try checkCommandBufferError(cb)
    }

    private nonisolated func waitUntilCompleted(_ cb: any MTLCommandBuffer) {
        let traceStart = GPUTrace.begin()
        defer { GPUTrace.end("gpu_wait", since: traceStart) }
        cb.waitUntilCompleted()
    }

}
