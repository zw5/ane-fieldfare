import Foundation
public import Metal

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

public final class RealForwardRunner: ChunkedPrefillRunner, MultimodalPrefillRunner, ContextWindowReporting,
    FusedGreedyProducer {
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
    private let hidden: any MTLBuffer        // [D] FP16
    private let normed: any MTLBuffer        // [D] FP16
    private let attnOut: any MTLBuffer       // [N_HEADS * head_dim] FP16
    private let qScratch: any MTLBuffer      // [N_HEADS * head_dim] FP16
    private let oOut: any MTLBuffer          // [D] FP16
    private let h1Buf: any MTLBuffer         // [D] FP16 (dense MLP output)
    private let h2Buf: any MTLBuffer         // [D] FP16 (routed output)
    private let routedX: any MTLBuffer       // [D] FP16 (pre_feedforward_layernorm_2 output)
    private let denseX: any MTLBuffer        // [D] FP16 (pre_feedforward_layernorm output)
    private let denseScratchGate: any MTLBuffer // [F=2112] FP16
    private let denseScratchUp: any MTLBuffer   // [F=2112] FP16
    private let denseScratchAct: any MTLBuffer  // [F=2112] FP16
    private let routerInput: any MTLBuffer   // [D] FP16 (rmsnorm_no_scale(h))
    private let nextLayerScores: any MTLBuffer // [numExperts] FP32, early-read guess
    private let zeroResidual: any MTLBuffer  // [D] FP16 zeros — for routed branch base
    private let outIndices: any MTLBuffer    // [topK] UInt32
    private let outWeights: any MTLBuffer    // [topK] FP16
    // Persistent MoE scratch, allocated once; about 56 KiB at production shape.
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
    private let profileKernels = ProcessInfo.processInfo.environment["ANE_FIELDFARE_KERNEL_PROFILE"] != nil
    private var kernelProfile: [String: Double] = [:]
    private var kernelProfileTokens = 0
    /// Single-page probes per expert when choosing which experts a decode
    /// layer binds. A wrong "resident" costs a page fault at submission.
    static let bindProbes = 2
    /// Issues verify's advisory next-layer reads in order, off its thread.
    // SAFETY: a serial queue keeps advisories in issue order, so storage
    // serves the earliest-needed layer first. Each work item captures only a
    // `Sendable` layer mapping and its expert IDs.
    private static let readAheadQueue = DispatchQueue(label: "ane-fieldfare.verify-read-ahead", qos: .userInitiated)
    /// Guessed experts a GPU-routed decode layer binds; the guess's top 24
    /// hold all 8 routes in about 9 of 10 layers.
    private let bindTop = ProcessInfo.processInfo.environment["ANE_FIELDFARE_BIND_TOP"].flatMap(Int.init) ?? 24
    /// Decode folds the shared expert's post-FFN-1 norm and the next layer's
    /// input norm into the layer tail (bitwise the same);
    /// `ANE_FIELDFARE_FUSED_TAIL_NORMS=0` runs them as separate dispatches.
    private lazy var fuseTailNorms = fusedTail.fusesNorms
        && ProcessInfo.processInfo.environment["ANE_FIELDFARE_FUSED_TAIL_NORMS"] != "0"
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
    public var aneRoutedPrefillProfile: [String: Any] { supervisor.routedPrefillProfile }
    public var supervisorProfile: [String: Any] { supervisor.profile }

    /// Per-layer `router.scale * D^-0.5` pre-folded into one BF16 buffer
    /// allocation per layer. ~168 KB total at 30 layers × 2816 BF16 — bounded
    /// host work done once at init.
    private let effectiveScaleBuffers: [any MTLBuffer]
    private let sharedExpertProjections: [LayerSharedExpertProjections]
    /// Next-layer early read; nil when `RuntimeConfiguration.earlyExpertRead`
    /// is off. See `NextLayerExpertPrefetcher`.
    private let earlyExpertReads: NextLayerExpertPrefetcher?
    /// Early-read counts since this runner was created; nil when it is off.
    public var earlyExpertReadStats: EarlyExpertReadStats? { earlyExpertReads?.stats }
    /// The guess the early read scores with, `fitted` or `router`; `off` when off.
    public var earlyExpertReadGuess: RuntimeEarlyExpertRead { earlyExpertReads?.guess ?? .off }

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
    /// Set while a drafter is attached: prefill, verify and decode copy the
    /// drafter's context layers into its feature ring.
    var draftFeatureTap: DraftFeatureTap?
    /// Decode GPU time: attention through router, shared FFN, routed + tail.
    public private(set) var decodeAttentionGPUNanos: UInt64 = 0
    public private(set) var decodeSharedGPUNanos: UInt64 = 0
    public private(set) var decodeRoutedGPUNanos: UInt64 = 0
    /// Decode layers that ran the host-routed experts: layer 0 always, a
    /// GPU-routed layer that routed to an unbound expert, and every layer when
    /// GPU routing is off (no early reads, `ANE_FIELDFARE_HOST_ROUTED_DECODE`).
    public private(set) var decodeLayerPatches: UInt64 = 0
    /// Signalled by each decode layer's router, ahead of its command buffer's
    /// completion.
    private let routerEvent: any MTLSharedEvent
    private var routerEventValue: UInt64 = 0
    /// Verify routes a layer's experts on the GPU, from the previous layer's
    /// guess, before the host has read its routes;
    /// `ANE_FIELDFARE_HOST_ROUTED_VERIFY` plans every layer on the host.
    var gpuRoutedVerify = ProcessInfo.processInfo.environment["ANE_FIELDFARE_HOST_ROUTED_VERIFY"] == nil
    /// Guessed experts per verify row a GPU-routed layer binds.
    private let verifyBindTop = ProcessInfo.processInfo.environment["ANE_FIELDFARE_VERIFY_BIND_TOP"]
        .flatMap(Int.init) ?? 48
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

        func buf(_ count: Int, _ stride: Int = MemoryLayout<Float16>.size) throws -> any MTLBuffer {
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
            throw MetalError.commandBufferUnavailable
        }
        self.routerEvent = routerEvent
        self.routedSlots = try buf(MoE.maxStreamedExperts, MemoryLayout<UInt64>.size)
        self.routedAddressTable = try buf(cfg.numLayers * cfg.numExperts, MemoryLayout<UInt64>.size)
        self.routedFallback = try buf(model.routedExpertBytes, 1)
        unsafe memset(self.routedFallback.contents(), 0, self.routedFallback.length)
        self.savedHidden = try buf(cfg.hiddenSize)
        self.patchSlots = try buf(cfg.numLayers * cfg.topKExperts, MemoryLayout<UInt32>.size)
        self.gpuRoutedDecode = moe.routedSlotsAreAddresses
            && ProcessInfo.processInfo.environment["ANE_FIELDFARE_HOST_ROUTED_DECODE"] == nil
        guard let tok = device.makeBuffer(length: MemoryLayout<UInt32>.size,
                                          options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        self.greedyTokenBuf = tok
        let vocab = cfg.vocabSize, hiddenSize = cfg.hiddenSize
        self.hotVocabulary = try runtimeConfiguration.headPath == .fusedRows && hiddenSize == 2816
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
        var perLayer: [any MTLBuffer] = []
        perLayer.reserveCapacity(cfg.numLayers)
        let invSqrtD = Float(1.0) / Float(D).squareRoot()
        let dInts = D
        for L in 0..<cfg.numLayers {
            let scaleView = try model.routerScale(layer: L)
            guard let buf = device.makeBuffer(length: dInts * MemoryLayout<UInt16>.size,
                                              options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            let src = unsafe scaleView.buffer.contents()
                .advanced(by: Int(scaleView.offset))
                .assumingMemoryBound(to: UInt16.self)
            let dst = unsafe buf.contents().assumingMemoryBound(to: UInt16.self)
            for i in 0..<dInts {
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
        resetTransientState()
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

    /// The weights one verify layer reads, resolved once per runner.
    private struct VerifyLayer {
        let isFull: Bool
        let headDim: Int
        let numKVHeads: Int
        let inputNorm, q, k, v, o, postAttention, preFFN, preFFN2, postFFN2, postFFN: TensorView
        let qNorm, kNorm, router, routerPerExpertScale: TensorView
        let layerScalar: Float
    }
    private var verifyLayerCache: [VerifyLayer]?

    private func verifyLayers() throws -> [VerifyLayer] {
        if let verifyLayerCache { return verifyLayerCache }
        let layers = try (0..<cfg.numLayers).map { L in
            let isFull = cfg.fullAttentionLayerMask[L] != 0
            let scalar = try model.layerScalar(layer: L)
            let scalarBits = unsafe scalar.buffer.contents().advanced(by: Int(scalar.offset))
                .assumingMemoryBound(to: UInt16.self)[0]
            return VerifyLayer(
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
                layerScalar: Quantization.bf16ToFloat(scalarBits))
        }
        verifyLayerCache = layers
        return layers
    }

    /// Runs a verify block of `rows` tokens at the KV cursor, reading their
    /// IDs from `tokens` on the GPU (so a drafter's output feeds it without a
    /// host copy), and leaves each row's greedy next token in `greedy` as
    /// `UInt32`. `behindHead`, when given, runs once the final head is
    /// committed and before it is waited on, to queue dependent GPU work.
    ///
    /// The layers are pipelined: while the GPU runs a layer's attention, the
    /// host encodes that layer's tail and the next layer's attention and
    /// shared FFN, so once the routes are back the only work between the
    /// router and the next layer is planning and committing the routed tiles.
    func verifyBlock(rows t: Int, startPosition: Int, tokens: any MTLBuffer, greedy: any MTLBuffer,
                     behindHead: (() throws -> Void)? = nil) async throws {
        try prefillChunkState.requireClean(operation: "verify")
        guard t > 0, t <= Self.maxVerifyTokens else {
            throw PrefillError.chunkedUnsupported("verify takes 1...\(Self.maxVerifyTokens) rows, got \(t)")
        }
        guard kv.position == startPosition, startPosition + t <= maxContext else {
            throw PrefillError.chunkedUnsupported(
                "verify at \(startPosition) of \(t) rows; cursor \(kv.position), maxContext \(maxContext)")
        }
        if kv.fp16RingEnabled, let ringLayer = (0..<cfg.numLayers).first(where: { kv.ringCapacity(layer: $0) > 0 }),
           min(maxContext, cfg.slidingWindow + t) > kv.ringCapacity(layer: ringLayer) {
            throw PrefillError.chunkedUnsupported(
                "FP16 KV ring capacity \(kv.ringCapacity(layer: ringLayer)) cannot hold a \(t)-row verify")
        }
        let scratch = try prefillScratch ?? ensurePrefillScratch(config: .production(chunkTokens: 32))
        let layers = try verifyLayers()
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
            let rotatedPairs = w.isFull
                ? UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor / 2.0)
                : UInt32(w.headDim / 2)
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
            let ringCapacity = kv.ringCapacity(layer: L)
            try prefillAttention.encodeCausal(
                commandBuffer: cb, q: scratch.q,
                k: kv.keyBuffer(layer: L, validTokenCount: startPosition + t),
                v: kv.valueBuffer(layer: L, validTokenCount: startPosition + t),
                out: scratch.attentionOutput,
                params: PrefillAttentionParams(
                    startPosition: UInt32(startPosition), queryCount: UInt32(t),
                    headDim: UInt32(w.headDim), numQHeads: UInt32(cfg.numHeads),
                    numKVHeads: UInt32(w.numKVHeads), kvValidCount: UInt32(startPosition + t),
                    slidingWindow: w.isFull ? UInt32(startPosition + t) : UInt32(cfg.slidingWindow),
                    kvTokenStrideElements: UInt32(kvDim), qTokenStrideElements: UInt32(qDim),
                    oTokenStrideElements: UInt32(qDim), scale: 1.0,
                    bidirectionalBlockStart: 0, bidirectionalBlockEnd: 0),
                kvRingCapacity: ringCapacity > 0 && startPosition + t > ringCapacity ? UInt32(ringCapacity) : 0,
                layerKind: w.isFull ? .full : .slidingWindow, path: prefillAttentionPath)
            try project(cb, w.o, x: scratch.attentionOutput, y: scratch.h1, rows: D, columns: qDim)
            prefillPostAttention.encode(commandBuffer: cb, hidden: scratch.hidden, attn: scratch.h1,
                                        denseX: scratch.denseX, routedX: scratch.routedX, routerX: scratch.routerX,
                                        postAttentionWeight: w.postAttention.buffer,
                                        postAttentionWeightOffset: Int(w.postAttention.offset),
                                        preFFNWeight: w.preFFN.buffer, preFFNWeightOffset: Int(w.preFFN.offset),
                                        preFFN2Weight: w.preFFN2.buffer, preFFN2WeightOffset: Int(w.preFFN2.offset),
                                        queryCount: UInt32(t), d: UInt32(D),
                                        hiddenStrideElements: UInt32(D), attnStrideElements: UInt32(D),
                                        denseStrideElements: UInt32(D), routedStrideElements: UInt32(D),
                                        routerStrideElements: UInt32(D), eps: eps)
            try encodeVerifyRouter(cb, layer: L, rows: t, scratch: scratch, layers: layers)
            routerEventValue &+= 1
            cb.encodeSignalEvent(routerEvent, value: routerEventValue)
            return (cb, routerEventValue)
        }

        /// The shared FFN, which needs only the attention's output and runs
        /// while the host plans the routed experts.
        func encodeShared(_ L: Int) throws -> any MTLCommandBuffer {
            let cb = try ctx.makeCommandBuffer()
            cb.label = "verify rows=\(t) layer=\(L) phase=shared_expert"
            let proj = sharedExpertProjections[L]
            try prefillSharedExpert.encodeBlock(commandBuffer: cb, x: scratch.denseX, y: scratch.h1,
                                                gate: proj.gate, up: proj.up, down: proj.down,
                                                scratchGate: scratch.sharedGateScratch,
                                                scratchUp: scratch.sharedUpScratch,
                                                scratchAct: scratch.sharedActScratch,
                                                queryCount: t, d: D, intermediate: cfg.intermediateSize,
                                                xStrideElements: D, yStrideElements: D, smallBatch: true)
            prefillRMS.encodeBF16W(commandBuffer: cb, x: scratch.h1,
                                   weight: proj.postF1.buffer, weightOffset: Int(proj.postF1.offset),
                                   out: scratch.h1, t: UInt32(t), d: UInt32(D), eps: eps)
            return cb
        }

        /// The routed experts' reduce, the layer tail and the drafter's
        /// feature copy; committed once the layer's routed tiles are.
        func encodeTail(_ L: Int) throws -> any MTLCommandBuffer {
            let w = layers[L]
            let cb = try ctx.makeCommandBuffer()
            cb.label = "verify rows=\(t) layer=\(L) phase=tail"
            prefillMoE.encodeReduceTokenMajor(commandBuffer: cb, routePartials: scratch.routePartials,
                                              routeWeights: scratch.routeWeights, h2: scratch.h2,
                                              queryCount: UInt32(t), topK: UInt32(topK), d: UInt32(D))
            prefillLayerTail.encode(commandBuffer: cb, h2: scratch.h2, h1: scratch.h1, hidden: scratch.hidden,
                                    postFFN2Weight: w.postFFN2.buffer, postFFN2WeightOffset: Int(w.postFFN2.offset),
                                    postFFNWeight: w.postFFN.buffer, postFFNWeightOffset: Int(w.postFFN.offset),
                                    queryCount: UInt32(t), d: UInt32(D), h2StrideElements: UInt32(D),
                                    h1StrideElements: UInt32(D), hiddenStrideElements: UInt32(D),
                                    eps: eps, layerScalar: w.layerScalar)
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
        var routeIDs = [UInt32](repeating: 0, count: routeCount)
        var routeWeights = [Float16](repeating: 0, count: routeCount)
        var guesses = [UInt32](repeating: 0, count: routeCount)
        let routeIDPtr = unsafe scratch.routeIDs.contents().bindMemory(to: UInt32.self, capacity: routeCount)
        let routeWeightPtr = unsafe scratch.routeWeights.contents().bindMemory(to: Float16.self, capacity: routeCount)
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
            for i in 0..<routeCount {
                unsafe routeIDs[i] = min(routeIDPtr[i], UInt32(numExperts - 1))
                unsafe routeWeights[i] = routeWeightPtr[i]
            }
            if guessing, !last {
                for i in 0..<routeCount { unsafe guesses[i] = guessPtr[i] }
            }
            prefillUnionExpertsBeforeCap &+= UInt64(Set(routeIDs).count)
            prefillUnionLayerTiles &+= 1
            if let cap = prefillExpertUnionCap {
                unsafe Self.capExpertUnion(ids: routeIDs, weights: &routeWeights, rows: t, topK: topK,
                                           numExperts: numExperts, cap: cap, write: routeWeightPtr)
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
            boundVerifyRoutes &+= UInt64(allPairs.count - allPairs.filter { !bound[Int($0.expert)] }.count)
            if capped, pairs.count < routeCount {
                // A dropped route leaves its partial unwritten; the tiles
                // write every other one.
                let clear = try ctx.makeCommandBuffer()
                guard let blit = clear.makeTracedBlitEncoder() else { throw ModelError.residentBufferWrapFailed }
                blit.fill(buffer: scratch.routePartials,
                          range: 0..<(routeCount * D * MemoryLayout<Float16>.stride), value: 0)
                blit.endEncoding()
                clear.commit()
            }
            if pairs.isEmpty { GPUTrace.end("route_plan", since: tracePlan) }
            if !pairs.isEmpty {
                // Every expert has a fixed buffer, so the whole union binds at
                // once. Tiles of resident experts go first and run while the
                // absent experts read, in tile order. Experts the previous
                // layer's early read found resident are not probed again.
                let unionExperts = Array(Set(pairs.map { Int($0.expert) })).sorted()
                let absent = Set(try model.nonResidentRoutedExperts(
                    layer: L, experts: unionExperts.filter { !guessedResident.contains($0) },
                    probes: Self.bindProbes))
                prefillTileHits &+= UInt64(unionExperts.count - absent.count)
                prefillTileMisses &+= UInt64(absent.count)
                var expertSortKeys = model.routedExpertPhysicalOffsets(layer: L)
                for e in expertSortKeys.indices {
                    expertSortKeys[e] = absent.contains(e) ? expertSortKeys[e] + 1 : 0
                }
                let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
                    pairs, queryCount: t, topK: topK, numExperts: numExperts,
                    tileExpertCount: 16, expertSortKeys: expertSortKeys)
                let metadata = try prefillGroupedMoE.makeStreamedMetadataBuffers(device: ctx.device, routes: routes)
                let routedOffsets = model.routedExpertOffsets(layer: L)
                let missingByTile = try routes.tiles.indices.map { tileIndex in
                    try PrefillStreamedTileBinding.expertIDs(forTile: tileIndex, routes: routes).filter(absent.contains)
                }
                GPUTrace.end("route_plan", since: tracePlan)

                let routedWallStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                let mapped = try model.expertLayer(L)
                var tileCBs: [any MTLCommandBuffer] = []
                patchedVerifyLayers &+= 1
                // One reader reads each tile's absent experts in tile order and
                // reports each tile as it lands, so a tile waits only for its own
                // experts and storage never idles between tiles. If a tile
                // throws, reads not yet started are dropped and leaving the group
                // waits for the one in flight.
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
                                    throw ModelError.indexCorrupt(detail: "verify tile \(tileIndex) was never read")
                                }
                                landedThrough = landedTile
                            }
                            prefillTileFetchNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - fetchStart
                            GPUTrace.end("tile_fetch_wait", since: fetchStart)
                        }
                        let traceTile = GPUTrace.begin()
                        let expertIDs = try PrefillStreamedTileBinding.expertIDs(forTile: tileIndex, routes: routes)
                        let binding = try PrefillStreamedTileBinding(
                            expertIDs: expertIDs, views: try model.routedExpertViews(layer: L, experts: expertIDs))
                        try binding.validateCoversPairs(routes.sortedPairs, pairStart: Int(tile.pairStart),
                                                        pairCount: Int(tile.pairCount))
                        if tileIndex >= argumentSlots {
                            // The argument buffer this tile reuses must be free.
                            let drainStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                            try waitForCompletion(tileCBs[tileIndex - argumentSlots])
                            prefillTileDrainNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - drainStart
                        }
                        let argumentBuffer = try prefillGroupedMoE.makeStreamedArgumentBuffer(
                            device: ctx.device, binding: binding, reuseSlot: tileIndex % argumentSlots)
                        let tileCB = try ctx.makeCommandBuffer()
                        tileCB.label = "verify rows=\(t) layer=\(L) phase=routed_tile"
                        let groups = routes.groups[Int(tile.groupStart)..<Int(tile.groupStart + tile.groupCount)]
                        try prefillGroupedMoE.encodeVerifyRows(
                            commandBuffer: tileCB, hidden: scratch.routedX, sortedPairs: metadata.sortedPairs,
                            routePartials: scratch.routePartials, argumentBuffer: argumentBuffer, binding: binding,
                            params: PrefillGroupedRoutedMoEStreamedParams(
                                pairStart: tile.pairStart, pairCount: tile.pairCount, d: UInt32(D),
                                routedIntermediate: UInt32(cfg.moeIntermediateSize), topK: UInt32(topK),
                                hiddenStrideElements: UInt32(D), binding: binding, offsets: routedOffsets),
                            segments: groups.map { SIMD2($0.pairStart, $0.pairCount) },
                            totalPairs: routes.sortedPairs.count)
                        tileCB.commit()
                        tileCBs.append(tileCB)
                        GPUTrace.end("tile_encode", since: traceTile)
                    }
                    try await reader.waitForAll()
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
                let likely = Set(guesses.map { min(Int($0), numExperts - 1) })
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

        let finalNorm = model.finalNorm
        let lm = model.lmHead
        GPUTrace.setScope("verify head")
        let headCB = try ctx.makeCommandBuffer()
        headCB.label = "verify rows=\(t) phase=head"
        try fusionHead.encodeGreedyRows(commandBuffer: headCB, hidden: scratch.hidden, rowCount: t,
                                        normWeight: finalNorm.buffer, normOffset: Int(finalNorm.offset),
                                        weights: lm.buffer, weightsOffset: Int(lm.offset),
                                        scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                                        biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                                        outTokens: greedy, d: UInt32(D), vocab: UInt32(cfg.vocabSize),
                                        rmsEps: eps)
        headCB.commit()
        try behindHead?()
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
                                    scratch: PrefillChunkScratchBuffers, layers: [VerifyLayer]) throws {
        let D = UInt32(cfg.hiddenSize)
        let w = layers[L]
        var next: MoE.RouterLogitsTarget?
        if let early = earlyExpertReads, L + 1 < cfg.numLayers {
            if let guess = early.guessWeights {
                let m = guess.matrix(targetLayer: L + 1)
                next = MoE.RouterLogitsTarget(
                    weights: guess.buffer, weightsOffset: m.weightsOffset,
                    scales: guess.buffer, scalesOffset: m.scalesOffset,
                    biases: guess.buffer, biasesOffset: m.biasesOffset,
                    effectiveScale: guess.unitScale, outLogits: chunkGuessWeights)
            } else {
                let router = layers[L + 1].router
                next = MoE.RouterLogitsTarget(
                    weights: router.buffer, weightsOffset: Int(router.offset),
                    scales: router.buffer, scalesOffset: Int(router.scaleOffset),
                    biases: router.buffer, biasesOffset: Int(router.biasOffset),
                    effectiveScale: effectiveScaleBuffers[L + 1], outLogits: chunkGuessWeights)
            }
        }
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

    /// The last verify block's final hidden rows (before the head's norm),
    /// copied out for tests.
    func lastVerifyHidden(rows: Int) throws -> [Float16] {
        guard let scratch = prefillScratch else { return [] }
        let count = rows * cfg.hiddenSize
        guard let out = ctx.device.makeBuffer(length: count * MemoryLayout<Float16>.stride,
                                              options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        try runSync { cb in
            guard let blit = cb.makeBlitCommandEncoder() else { return }
            blit.copy(from: scratch.hidden, sourceOffset: 0, to: out, destinationOffset: 0,
                      size: count * MemoryLayout<Float16>.stride)
            blit.endEncoding()
        }
        let values = unsafe out.contents().bindMemory(to: Float16.self, capacity: count)
        return unsafe Array(UnsafeBufferPointer(start: values, count: count))
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
    private func snapshotFP16(_ buffer: any MTLBuffer, count: Int) -> [Float16] {
        let pointer = unsafe buffer.contents().bindMemory(to: Float16.self, capacity: count)
        return unsafe Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    public func produce(token: Int32, position: Int, into logits: any MTLBuffer) async throws {
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
                               into logits: any MTLBuffer,
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
                                  into logits: any MTLBuffer,
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

    /// Copies `tokenCount` rows of K and V at `startPosition` into layer
    /// `layer`'s cache, splitting at the ring's wrap.
    private func encodeKVCacheWrite(commandBuffer: any MTLCommandBuffer, layer: Int, startPosition: Int,
                                    tokenCount: Int, keySource: any MTLBuffer, valueSource: any MTLBuffer,
                                    bytesPerToken: Int) throws {
        guard tokenCount > 0 else { return }
        let capacity = kv.capacity(layer: layer)
        let firstSpan = min(tokenCount, capacity - startPosition % capacity)
        guard let blit = commandBuffer.makeTracedBlitEncoder() else { throw ModelError.residentBufferWrapFailed }
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
                                     logits: any MTLBuffer,
                                     scratch: PrefillChunkScratchBuffers,
                                     config: PrefillRuntimeConfig,
                                     writeFinalHead: Bool,
                                     embeddingOverride: any MTLBuffer? = nil,
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
        guard let tokenBuffer = unsafe ctx.device.makeBuffer(bytes: tokenIDs,
                                                      length: tokenIDs.count * MemoryLayout<UInt32>.stride,
                                                      options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        let D = cfg.hiddenSize
        let eps: Float = 1e-6
        let sqrtHidden = Float(D).squareRoot()
        let t = tokens.count
        let emb = model.embedding

        func encodeInt4Projection(commandBuffer: any MTLCommandBuffer,
                                  family: PrefillProjectionFamily,
                                  weights: TensorView,
                                  x: any MTLBuffer,
                                  y: any MTLBuffer,
                                  rows: Int,
                                  columns: Int,
                                  tokenCount: Int,
                                  xStrideElements: Int,
                                  yStrideElements: Int) throws {
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
                  let blit = cb.makeTracedBlitEncoder() else {
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

        struct PendingLayerTail {
            let attention: any MTLCommandBuffer
            let tail: any MTLCommandBuffer
            let shared: any MTLCommandBuffer
            let tiles: [any MTLCommandBuffer]
            /// Kept alive until the tiles that read it complete.
            let metadata: Any
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
            try encodeInt4Projection(commandBuffer: cb,
                                 family: .q,
                                 weights: views.q,
                                 x: scratch.normed,
                                 y: scratch.q,
                                 rows: qDim,
                                 columns: D,
                                 tokenCount: t,
                                 xStrideElements: D,
                                 yStrideElements: qDim)
            try encodeInt4Projection(commandBuffer: cb,
                                 family: .kv,
                                 weights: views.k,
                                 x: scratch.normed,
                                 y: scratch.kStage,
                                 rows: kvDim,
                                 columns: D,
                                 tokenCount: t,
                                 xStrideElements: D,
                                 yStrideElements: kvDim)

            try encodeInt4Projection(commandBuffer: cb,
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
            do {
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
            }

            let bytes = t * kvDim * MemoryLayout<Float16>.stride
            try encodeKVCacheWrite(commandBuffer: cb,
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
            try prefillAttention.encodeCausal(commandBuffer: cb,
                                          q: scratch.q,
                                          k: keyBuffer,
                                          v: valueBuffer,
                                          out: scratch.attentionOutput,
                                          params: params,
                                          kvRingCapacity: activeRingCapacity,
                                          layerKind: isFull ? .full : .slidingWindow,
                                          path: prefillAttentionPath)
            try encodeInt4Projection(commandBuffer: cb,
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
                weights: views.router.buffer, weightsOffset: Int(views.router.offset),
                scales: views.router.buffer, scalesOffset: Int(views.router.scaleOffset),
                biases: views.router.buffer, biasesOffset: Int(views.router.biasOffset),
                hidden: scratch.routerX,
                effectiveScale: effectiveScaleBuffers[L],
                perExpertScale: views.routerPerExpertScale.buffer,
                perExpertScaleOffset: Int(views.routerPerExpertScale.offset),
                outIndices: scratch.routeIDs, outWeights: scratch.routeWeights,
                queryCount: UInt32(t), numExperts: UInt32(cfg.numExperts), d: UInt32(D),
                topK: UInt32(cfg.topKExperts), hiddenStrideElements: UInt32(D))
                    // Clear routed partials here, in a command buffer the host
                    // completes, so routed guests on other queues never race it.
                    // Only this chunk's `t * topK` rows are read.
                    if let blit = cb.makeTracedBlitEncoder() {
                        blit.fill(buffer: scratch.routePartials,
                                  range: 0..<(t * cfg.topKExperts * D * MemoryLayout<Float16>.stride),
                                  value: 0)
                        blit.endEncoding()
                    }
                    // The shared FFN needs only this command buffer's output,
                    // so it is queued behind it and runs while the host plans
                    // the routed experts.
                    routerEventValue &+= 1
                    cb.encodeSignalEvent(routerEvent, value: routerEventValue)
                    cb.commit()
                    GPUTrace.end("attention_encode", since: traceAttention)
                    let traceShared = GPUTrace.begin()
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
                                                        yStrideElements: D)
                    prefillRMS.encodeBF16W(commandBuffer: sharedCB,
                                           x: scratch.h1,
                                           weight: sharedProj.postF1.buffer,
                                           weightOffset: Int(sharedProj.postF1.offset),
                                           out: scratch.h1,
                                           t: UInt32(t),
                                           d: UInt32(D),
                                           eps: eps)
                    sharedCB.commit()
                    GPUTrace.end("shared_encode", since: traceShared)
                    try waitForCompletion(cb)
                    // This layer's attention read the previous layer's output,
                    // so the previous layer's tail, shared FFN and tiles are done.
                    if let pending = pendingTail { try settle(pending); pendingTail = nil }

                    let tracePlan = GPUTrace.begin()
                    let routeCount = t * cfg.topKExperts
                    let idPtr = unsafe scratch.routeIDs.contents()
                        .bindMemory(to: UInt32.self, capacity: routeCount)
                    let weightPtr = unsafe scratch.routeWeights.contents()
                        .bindMemory(to: Float16.self, capacity: routeCount)
                    var routeIDs = [UInt32]()
                    routeIDs.reserveCapacity(routeCount)
                    var routeWeights = [Float16]()
                    routeWeights.reserveCapacity(routeCount)
                    for i in 0..<routeCount {
                        unsafe routeIDs.append(min(idPtr[i], UInt32(cfg.numExperts - 1)))
                        unsafe routeWeights.append(weightPtr[i])
                    }
                    capture?.routes[L] = routeIDs
                    let uncappedUnion = Set(routeIDs)
                    prefillUnionExpertsBeforeCap &+= UInt64(uncappedUnion.count)
                    prefillUnionLayerTiles &+= 1

                    if let cap = prefillExpertUnionCap {
                        unsafe Self.capExpertUnion(ids: routeIDs, weights: &routeWeights, rows: t,
                                                   topK: cfg.topKExperts, numExperts: cfg.numExperts,
                                                   cap: cap, write: weightPtr)
                    }
                    let pairs = PrefillRouter.makeTokenExpertPairs(indices: routeIDs,
                                                                   weights: routeWeights,
                                                                   queryCount: t,
                                                                   topK: cfg.topKExperts)
                        .filter { $0.weight != 0 }
                    prefillUnionExpertsAfterCap &+= UInt64(Set(pairs.map(\.expert)).count)
                    // Every expert has a fixed buffer, so the whole union binds
                    // at once. Tiles of resident experts go first and run while
                    // the absent experts read, in tile order.
                    let unionExperts = Array(Set(pairs.map { Int($0.expert) })).sorted()
                    let absent = Set(try model.nonResidentRoutedExperts(layer: L, experts: unionExperts))
                    prefillTileHits &+= UInt64(unionExperts.count - absent.count)
                    prefillTileMisses &+= UInt64(absent.count)
                    var expertSortKeys = model.routedExpertPhysicalOffsets(layer: L)
                    for e in expertSortKeys.indices {
                        expertSortKeys[e] = absent.contains(e) ? expertSortKeys[e] + 1 : 0
                    }
                    let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
                        pairs,
                        queryCount: t,
                        topK: cfg.topKExperts,
                        numExperts: cfg.numExperts,
                        tileExpertCount: 16,
                        expertSortKeys: expertSortKeys)

                    let metadata = try prefillGroupedMoE.makeStreamedMetadataBuffers(
                        device: ctx.device,
                        routes: routes)
                    let routedOffsets = model.routedExpertOffsets(layer: L)
                    let routedWallStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                    var routedOnHost = true
                    if let guest = supervisor.routedPrefill(layer: L) {
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
                                  let blit = clearCB.makeTracedBlitEncoder() else {
                                throw ModelError.residentBufferWrapFailed
                            }
                            blit.fill(buffer: scratch.routePartials,
                                      range: 0..<scratch.routePartials.length, value: 0)
                            blit.endEncoding()
                            clearCB.commit()
                        }
                    }
                    GPUTrace.end("route_plan", since: tracePlan)
                    var tileCBs: [any MTLCommandBuffer] = []
                    if routedOnHost {
                        // One reader reads each tile's absent experts in tile
                        // order and reports each tile as it lands, so a tile
                        // waits only for its own experts and storage never
                        // idles between tiles. The reader is a child task: if
                        // a tile throws, reads not yet started are dropped and
                        // leaving this scope waits for the one in flight.
                        let missingByTile = try routes.tiles.indices.map { tileIndex in
                            try PrefillStreamedTileBinding.expertIDs(forTile: tileIndex, routes: routes)
                                .filter(absent.contains)
                        }
                        let mapped = try model.expertLayer(L)
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
                            let expertIDs = try PrefillStreamedTileBinding.expertIDs(forTile: tileIndex, routes: routes)
                            if !missingByTile[tileIndex].isEmpty {
                                let fetchStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                                while landedThrough < tileIndex {
                                    guard let landedTile = try await arrivals.next() else {
                                        throw ModelError.indexCorrupt(
                                            detail: "verify tile \(tileIndex) was never read")
                                    }
                                    landedThrough = landedTile
                                }
                                prefillTileFetchNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - fetchStart
                                GPUTrace.end("tile_fetch_wait", since: fetchStart)
                            }
                            let traceTile = GPUTrace.begin()
                            defer { GPUTrace.end("tile_encode", since: traceTile) }
                            let binding = try PrefillStreamedTileBinding(
                                expertIDs: expertIDs,
                                views: try model.routedExpertViews(layer: L, experts: expertIDs))
                            try binding.validateCoversPairs(routes.sortedPairs,
                                                            pairStart: Int(tile.pairStart),
                                                            pairCount: Int(tile.pairCount))
                            if tileIndex >= argumentSlots {
                                // The argument buffer this tile reuses must be free.
                                let drainStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                                try waitForCompletion(tileCBs[tileIndex - argumentSlots])
                                prefillTileDrainNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - drainStart
                            }
                            let argumentBuffer = try prefillGroupedMoE.makeStreamedArgumentBuffer(
                                device: ctx.device, binding: binding, reuseSlot: tileIndex % argumentSlots)
                            guard let tileCB = ctx.queue.makeCommandBuffer() else {
                                throw ModelError.residentBufferWrapFailed
                            }
                            tileCB.label = "prefill start=\(startPosition) count=\(t) layer=\(L) phase=routed_tile"
                            _ = prefillGroupedMoE.encodeStreamedBatched(
                                commandBuffer: tileCB,
                                hidden: scratch.routedX,
                                sortedPairs: metadata.sortedPairs,
                                routePartials: scratch.routePartials,
                                gateUpActScratch: scratch.routedGateUpActScratch,
                                downScratch: scratch.routedDownScratch,
                                argumentBuffer: argumentBuffer,
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
                            tileCB.commit()
                            tileCBs.append(tileCB)
                        }
                        try await reader.waitForAll()
                        }
                    }
                    prefillRoutedWallNanos &+=
                        clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - routedWallStart
                    // Long prefill chunks advise no next-layer reads: there the
                    // advice competes with the layer's own reads (measured: no
                    // gain on Metal, 16% slower on the ANE path).
                    let traceTail = GPUTrace.begin()
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
                    let scalarBits = unsafe views.layerScalar.buffer.contents()
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
                    if let capture, let captureBuffer = capture.buffer,
                       let slot = capture.hiddenLayers.firstIndex(of: L),
                       let blit = tailCB.makeTracedBlitEncoder() {
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
                    // Not waited here: the next layer's attention waits for it,
                    // as decode does, which saves a host round trip per layer.
                    pendingTail = PendingLayerTail(attention: cb, tail: tailCB, shared: sharedCB, tiles: tileCBs,
                                                   metadata: metadata)
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
        if let pending = pendingTail { try settle(pending) }

        if writeFinalHead {
            let finalNorm = model.finalNorm
            let lm = model.lmHead
            GPUTrace.setScope("prefill head")
            guard let finalCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            finalCB.label = "prefill start=\(startPosition) count=\(t) phase=final_head"
            if outputMode == .greedyIfAvailable, useFusedGreedyHead {
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
            if outputMode == .greedyIfAvailable, useFusedGreedyHead {
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
        let sqrtHidden = Float(cfg.hiddenSize).squareRoot()
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
            let routed: any MTLCommandBuffer?
            /// Experts `routed` binds; a route outside it is patched.
            let bound: [Bool]
            let layerScalar: Float
            let postFFN1: TensorView
            let postFFN2: TensorView
            let postFFN: TensorView
            /// The next layer's input norm, written by this layer's tail when
            /// the tail fuses norms; nil for the last layer.
            let nextInputNorm: TensorView?
            /// Profiling only: kernels split out ahead of `attention` and
            /// `shared`.
            var pieces: [(String, any MTLCommandBuffer)] = []
            var sharedPieces: [(String, any MTLCommandBuffer)] = []
        }
        var profiled: [(String, any MTLCommandBuffer)] = []
        var committed: [(any MTLCommandBuffer, ReferenceWritableKeyPath<RealForwardRunner, UInt64>)] = []

        func encodeTail(_ cb: any MTLCommandBuffer, _ work: LayerWork) throws {
            if fuseTailNorms {
                fusedTail.encodeWithNorms(commandBuffer: cb, h2: h2Buf, h1: h1Buf, hidden: hidden,
                                          postFFN1Weight: work.postFFN1,
                                          postFFN2Weight: work.postFFN2,
                                          postFFNWeight: work.postFFN,
                                          next: work.nextInputNorm.map { ($0, normed) },
                                          d: D, eps: eps, layerScalar: work.layerScalar)
            } else {
            fusedTail.encode(commandBuffer: cb,
                             h2: h2Buf,
                             h1: h1Buf,
                             hidden: hidden,
                             postFFN2Weight: work.postFFN2.buffer,
                             postFFN2WeightOffset: Int(work.postFFN2.offset),
                             postFFNWeight: work.postFFN.buffer,
                             postFFNWeightOffset: Int(work.postFFN.offset),
                             d: D,
                             eps: eps,
                             layerScalar: work.layerScalar)
            }
            try draftFeatureTap?.encodeCopy(commandBuffer: cb, layer: work.layer, source: hidden,
                                            position: position, rows: 1)
        }

        func encodeLayer(_ L: Int) throws -> LayerWork {
            GPUTrace.setScope("decode L\(L)")
            let tEncode = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            defer { totalCb1Nanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tEncode }
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
            let routerW  = try model.router(layer: L)
            let perExpertScale = try model.routerPerExpertScale(layer: L)
            let scalarView = try model.layerScalar(layer: L)
            let scalarPtr = unsafe scalarView.buffer.contents()
                .advanced(by: Int(scalarView.offset))
                .assumingMemoryBound(to: UInt16.self)

            var cb = try ctx.makeCommandBuffer()
            var pieces: [(String, any MTLCommandBuffer)] = []
            // Profiling: each kernel in its own command buffer, queued
            // back to back, so its GPU time is its own.
            func cut(_ name: String) throws {
                guard profileKernels else { return }
                pieces.append((name, cb))
                cb = try ctx.makeCommandBuffer()
            }
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
                                 outScale: sqrtHidden)
            }
            try cut("embed")
            // Past layer 0 the previous layer's tail wrote `normed`.
            if L == 0 || !fuseTailNorms {
                rms.encodeBF16W(commandBuffer: cb, x: hidden,
                                weight: inNorm.buffer, weightOffset: Int(inNorm.offset),
                                out: normed, d: D, eps: eps)
            }
            try cut("input_norm")
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
            try cut("qkv_gemv")
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
                                    rotatedPairs: isFull
                                        ? UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor / 2.0)
                                        : UInt32(headDimL / 2),
                                    eps: eps)
            try cut("qkv_epilogue")
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
                                    ringCapacity: ringCapacity > 0 && Int(seqLen) > ringCapacity
                                        ? UInt32(ringCapacity) : 0)
            }
            try cut(isFull ? "attn_full" : "attn_swa")
            int4.encode(commandBuffer: cb,
                        weights: o.buffer, weightsOffset: Int(o.offset),
                        scales:  o.buffer, scalesOffset:  Int(o.scaleOffset),
                        biases:  o.buffer, biasesOffset:  Int(o.biasOffset),
                        x: attnOut, y: oOut, m: D, n: qDim)
            try cut("o_proj")
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
            try cut("post_attn")
            // Early read: score layer L+1's experts over this layer's router
            // input, in the router's dispatch.
            var next: MoE.RouterLogitsTarget?
            if earlyExpertReads != nil, L + 1 < cfg.numLayers {
                if let guess = earlyExpertReads?.guessWeights {
                    let m = guess.matrix(targetLayer: L + 1)
                    next = MoE.RouterLogitsTarget(
                        weights: guess.buffer, weightsOffset: m.weightsOffset,
                        scales: guess.buffer, scalesOffset: m.scalesOffset,
                        biases: guess.buffer, biasesOffset: m.biasesOffset,
                        effectiveScale: guess.unitScale, outLogits: nextLayerScores)
                } else {
                    let w = try model.router(layer: L + 1)
                    next = MoE.RouterLogitsTarget(
                        weights: w.buffer, weightsOffset: Int(w.offset),
                        scales: w.buffer, scalesOffset: Int(w.scaleOffset),
                        biases: w.buffer, biasesOffset: Int(w.biasOffset),
                        effectiveScale: effectiveScaleBuffers[L + 1], outLogits: nextLayerScores)
                }
            }
            moe.encodeRouterGemma4(commandBuffer: cb,
                weights: routerW.buffer, weightsOffset: Int(routerW.offset),
                scales:  routerW.buffer, scalesOffset:  Int(routerW.scaleOffset),
                biases:  routerW.buffer, biasesOffset:  Int(routerW.biasOffset),
                hidden: routerInput,
                effectiveScale: effectiveScaleBuffers[L],
                perExpertScale: perExpertScale.buffer,
                perExpertScaleOffset: Int(perExpertScale.offset),
                outIndices: outIndices, outWeights: outWeights,
                numExperts: UInt32(numExperts), d: D, topK: UInt32(cfg.topKExperts),
                next: next)
            try cut("router")
            routerEventValue &+= 1
            cb.encodeSignalEvent(routerEvent, value: routerEventValue)

            // The shared dense MLP depends only on denseX; it runs while the
            // host checks the routes.
            var sharedCB = try ctx.makeCommandBuffer()
            var sharedPieces: [(String, any MTLCommandBuffer)] = []
            func cutShared(_ name: String) throws {
                guard profileKernels else { return }
                sharedPieces.append((name, sharedCB))
                sharedCB = try ctx.makeCommandBuffer()
            }
            try shared.encode(commandBuffer: sharedCB,
                              x: denseX,
                              gate: sharedProj.gate,
                              up: sharedProj.up,
                              down: sharedProj.down,
                              y: h1Buf,
                              scratchGate: denseScratchGate,
                              scratchUp: denseScratchUp,
                              scratchAct: denseScratchAct)
            try cutShared("shared_ffn")
            // A norm-fusing tail normalizes the raw shared output itself.
            if !fuseTailNorms {
                rms.encodeBF16W(commandBuffer: sharedCB, x: h1Buf,
                                weight: sharedProj.postF1.buffer,
                                weightOffset: Int(sharedProj.postF1.offset),
                                out: h1Buf, d: D, eps: eps)
            }

            return unsafe LayerWork(layer: L, attention: cb, event: routerEventValue, shared: sharedCB,
                             routed: nil, bound: [],
                             layerScalar: Quantization.bf16ToFloat(scalarPtr[0]),
                             postFFN1: sharedProj.postF1,
                             postFFN2: try model.postFFN2(layer: L),
                             postFFN: try model.postFFN(layer: L),
                             nextInputNorm: L + 1 < cfg.numLayers ? try model.inputNorm(layer: L + 1) : nil,
                             pieces: pieces, sharedPieces: sharedPieces)
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
            for e in topIndices(guess, count: bindTop) where mapped.isResident(e, probes: Self.bindProbes) {
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
            let routedWork = LayerWork(layer: L, attention: work.attention, event: work.event,
                                       shared: work.shared, routed: routedCB, bound: bound,
                                       layerScalar: work.layerScalar,
                                       postFFN1: work.postFFN1, postFFN2: work.postFFN2,
                                       postFFN: work.postFFN, nextInputNorm: work.nextInputNorm,
                                       pieces: work.pieces, sharedPieces: work.sharedPieces)
            try encodeTail(routedCB, routedWork)
            return routedWork
        }

        func commit(_ work: LayerWork) {
            for (name, cb) in work.pieces { cb.commit(); profiled.append((name, cb)) }
            work.attention.commit()
            for (name, cb) in work.sharedPieces { cb.commit(); profiled.append((name, cb)) }
            work.shared.commit()
            if profileKernels {
                profiled.append(("shared_norm", work.shared))
            }
            committed.append((work.attention, \.decodeAttentionGPUNanos))
            committed.append((work.shared, \.decodeSharedGPUNanos))
        }

        func commitRouted(_ work: LayerWork) {
            guard let routed = work.routed else { return }
            routed.commit()
            if profileKernels { profiled.append(("routed_gpu+tail", routed)) }
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
                totalIoNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tIoStart
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
            if work.routed != nil, let blit = cb.makeTracedBlitEncoder() {
                // The GPU-routed pass already ran the tail; restart from the
                // layer input it saved.
                blit.copy(from: savedHidden, sourceOffset: 0, to: hidden, destinationOffset: 0,
                          size: cfg.hiddenSize * MemoryLayout<Float16>.stride)
                blit.endEncoding()
            }
            if work.routed != nil {
                // The GPU-routed pass left correct activations in the bound
                // slots; recompute only the unbound ones, then the reduce.
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
            try encodeTail(cb, work)
            cb.commit()
            if profileKernels { profiled.append(("patch_routed+tail", cb)) }
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
            if routedWeightMassByRank.count == cfg.topKExperts {
                for rank in 0..<cfg.topKExperts {
                    unsafe routedWeightMassByRank[rank] += Double(weightPtr[rank])
                }
                routedWeightMassSamples &+= 1
            }
            let experts = (0..<activeRoutedExperts).map { unsafe min(Int(idxPtr[$0]), numExperts - 1) }
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
        let lm    = model.embedding
        GPUTrace.setScope("decode head")
        let tHead = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let headBefore = totalHeadFusedNanos
        if useFusedGreedyHead && outputMode == .greedyIfAvailable, let (row, check) = hotHead,
           let hot = hotVocabulary {
            let lm = model.lmHead
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
        } else if useFusedGreedyHead && outputMode == .greedyIfAvailable {
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
        if profileKernels {
            for (name, cb) in profiled {
                waitUntilCompleted(cb)
                kernelProfile[name, default: 0] += cb.gpuEndTime - cb.gpuStartTime
            }
            kernelProfile["head", default: 0] += Double(totalHeadFusedNanos - headBefore) / 1e9
            kernelProfileTokens += 1
            if kernelProfileTokens % 64 == 0 {
                let total = kernelProfile.values.reduce(0, +)
                var lines = "KPROF per token (us), \(kernelProfileTokens) tokens\n"
                for (name, t) in kernelProfile.sorted(by: { $0.value > $1.value }) {
                    unsafe lines += String(format: "  %-20@ %8.0f  %4.1f%%\n", name as NSString,
                                    t / Double(kernelProfileTokens) * 1e6, 100 * t / total)
                }
                FileHandle.standardError.write(Data(lines.utf8))
            }
        }
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
    /// first: a sleeping wait costs a ~40 µs thread wake per layer, longer
    /// than most of the wait. Past `routerSpinNanos` (a layer's attention
    /// takes well under 1 ms) it blocks on `cb`, so a stalled GPU cannot pin a
    /// core indefinitely.
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

    private func runSync(_ body: (any MTLCommandBuffer) -> Void) throws {
        let cb = try ctx.makeCommandBuffer()
        body(cb)
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
