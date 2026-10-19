public import Foundation
import Metal

/// The DFlash block-diffusion drafter on Metal, built around the target's
/// GPU and unified memory:
///
/// - Its weights are the installed INT4 file (`DFlashWeights`), read in place
///   from the page cache.
/// - Its context features come from `featureTap`, a ring the target's layer
///   tails write on the GPU, so they never pass through the CPU.
/// - Its context length is GPU state (`DFlashState`): `encodeAccept` advances
///   it from a verify's output on the GPU, so the next draft can be queued
///   behind the verify that decides its anchor and position.
/// - Its block (anchor, then drafts, as `UInt32`) is the buffer the next
///   verify reads its tokens from.
///
/// It shares the target's tied INT4 embedding for its input and output head
/// and drafts a block of tokens after an anchor in one forward pass.
public final class DFlashDrafter {
    public let config: DFlashDrafterConfig
    public let maxContext: Int
    public var vocabSize: Int { model.config.vocabSize }
    /// Context rows in the K/V cache, as of the last `settle` or
    /// host-driven update.
    public private(set) var contextLength = 0

    /// Where the target writes the context features.
    private(set) var featureTap: DraftFeatureTap
    /// `[maxRows]` UInt32: the block's anchor, then its drafts.
    let block: any MTLBuffer
    /// One `DFlashState`.
    let state: any MTLBuffer
    /// The vocabulary the draft head scores, 1/16 of the target's, plus the
    /// prompt's and accepted tokens: a draft outside it is a rejected draft,
    /// which verify catches. nil scores all.
    let hotVocabulary: HotVocabulary?
    /// Measured on the quick suite, 8k to 56k listed tokens left acceptance
    /// unchanged within noise.
    static let hotVocabularyCount = 16 * 1024
    /// Rows one projection dispatch takes; longer context updates loop.
    static let maxRows = DequantInt4GEMV.maxBatchRows
    /// The ring's size when no prefill needs more.
    static let baseFeatureRows = 256

    private let ctx: MetalContext
    private let model: Model
    private let weights: DFlashWeights
    private let int4: DequantInt4GEMV
    private let head: LMHeadChainInt4
    private let embedPSO, normF32PSO, normBF16PSO, addNormPSO, addHeadRowsPSO, qkNormRopePSO, storeKVPSO,
                attentionPSO, siluPSO, gatherPSO, finishPSO, acceptPSO: any MTLComputePipelineState
    private let kCache: [any MTLBuffer]
    private let vCache: [any MTLBuffer]
    /// Slots per layer's K/V cache: a sliding layer keeps only its window
    /// (plus two blocks) as a ring, the full layer every position.
    private let cacheCapacity: [Int]
    // Draft scratch, `maxRows` rows each.
    private let residual, normed, qkv, attention, delta, gateUp, act, headRows, picks: any MTLBuffer
    // Context-update scratch.
    private let feature, contextHidden, contextKV: any MTLBuffer

    /// Installs the drafter's INT4 weights next to its checkpoint on first
    /// use (a few seconds, off the concurrency pool), then maps them.
    public init(directory: URL, model: Model, context: MetalContext, maxContext: Int) async throws {
        let (config, layout) = try await BlockingIO.run {
            let config = try DFlashDrafterConfig.load(directory.appendingPathComponent("config.json"))
            return (config, try DFlashWeights.install(directory: directory, config: config))
        }
        guard config.hiddenSize == model.config.hiddenSize,
              config.contextLayers.allSatisfy({ $0 >= 0 && $0 < model.config.numLayers }),
              config.blockSize <= min(LMHeadChainInt4.maxGreedyRows, Self.maxRows) else {
            throw DFlashError.invalidCheckpoint("drafter does not match the target")
        }
        self.config = config
        self.ctx = context
        self.model = model
        self.maxContext = maxContext
        self.weights = try DFlashWeights.open(directory: directory, layout: layout, config: config,
                                              device: context.device)
        self.int4 = try DequantInt4GEMV(context: context)
        self.head = try LMHeadChainInt4(context: context, maxD: config.hiddenSize, maxVocab: model.config.vocabSize)
        self.hotVocabulary = try ProcessInfo.processInfo.environment["ANE_FIELDFARE_DRAFT_HOT_HEAD"] == "0"
            ? nil
            : HotVocabulary.shipped(vocab: model.config.vocabSize).map {
                try HotVocabulary(url: $0, vocab: model.config.vocabSize, device: context.device,
                                  shipped: Self.hotVocabularyCount)
            }
        embedPSO = try context.pipeline("dflash_embed_block")
        normF32PSO = try context.pipeline("dflash_rmsnorm_rows_f32")
        normBF16PSO = try context.pipeline("dflash_rmsnorm_rows_bf16")
        addNormPSO = try context.pipeline("dflash_residual_add_norm")
        addHeadRowsPSO = try context.pipeline("dflash_residual_add_head_rows")
        qkNormRopePSO = try context.pipeline("dflash_block_qk_norm_rope")
        storeKVPSO = try context.pipeline("dflash_store_context_kv")
        attentionPSO = try context.pipeline("dflash_block_attention")
        siluPSO = try context.pipeline("dflash_silu_mul")
        gatherPSO = try context.pipeline("dflash_gather_features")
        finishPSO = try context.pipeline("dflash_finish_block")
        acceptPSO = try context.pipeline("dflash_accept")

        let device = context.device
        func buffer(_ bytes: Int, _ label: String, shared: Bool = false) throws -> any MTLBuffer {
            guard let made = device.makeBuffer(length: max(bytes, 16),
                                               options: shared ? .storageModeShared : .storageModePrivate) else {
                throw DFlashError.invalidState("allocation failed: \(label)")
            }
            made.label = "dflash.\(label)"
            return made
        }
        let D = config.hiddenSize, F = config.intermediateSize, rows = Self.maxRows, bf = 2
        cacheCapacity = config.slidingLayers.map { sliding in
            sliding ? min(maxContext, config.slidingWindow + 2 * Self.maxRows) : maxContext
        }
        kCache = try cacheCapacity.enumerated().map { try buffer($1 * config.kvWidth * bf, "k\($0)") }
        vCache = try cacheCapacity.enumerated().map { try buffer($1 * config.kvWidth * bf, "v\($0)") }
        residual = try buffer(rows * D * 4, "residual")
        normed = try buffer(rows * D * bf, "normed")
        qkv = try buffer(rows * config.qkvWidth * bf, "qkv")
        attention = try buffer(rows * config.qWidth * bf, "attention")
        delta = try buffer(rows * D * bf, "delta")
        gateUp = try buffer(rows * 2 * F * bf, "gateUp")
        act = try buffer(rows * F * bf, "act")
        headRows = try buffer(rows * D * 2, "headRows")
        picks = try buffer(rows * 4, "picks")
        block = try buffer(rows * 4, "block", shared: true)
        state = try buffer(MemoryLayout<UInt32>.stride * 4, "state", shared: true)
        feature = try buffer(rows * config.contextLayers.count * D * bf, "feature")
        contextHidden = try buffer(rows * D * bf, "contextHidden")
        contextKV = try buffer(rows * 2 * config.kvWidth * bf, "contextKV")
        featureTap = try Self.makeTap(rows: Self.baseFeatureRows, config: config, device: device)
        unsafe memset(state.contents(), 0, state.length)
    }

    private static func makeTap(rows: Int, config: DFlashDrafterConfig, device: any MTLDevice) throws -> DraftFeatureTap {
        let bytes = config.contextLayers.count * rows * config.hiddenSize * MemoryLayout<Float16>.stride
        guard let ring = device.makeBuffer(length: bytes, options: .storageModePrivate) else {
            throw DFlashError.invalidState("feature ring allocation failed")
        }
        ring.label = "dflash.features"
        return DraftFeatureTap(buffer: ring, layers: config.contextLayers, capacity: rows,
                               hiddenSize: config.hiddenSize)
    }

    /// Resizes the feature ring to hold `rows` positions (at least the base
    /// size). Its contents are lost, so call it only once every written row
    /// is in the context. Returns whether the tap changed.
    @discardableResult
    func resizeFeatureRing(rows: Int) throws -> Bool {
        let wanted = max(rows, Self.baseFeatureRows)
        guard wanted != featureTap.capacity else { return false }
        featureTap = try Self.makeTap(rows: wanted, config: config, device: ctx.device)
        return true
    }

    /// Forget context past `length`, e.g. after a rewind. Host-driven: no
    /// drafter work may be in flight.
    public func truncateContext(to length: Int) throws {
        guard length >= 0, length <= contextLength else {
            throw DFlashError.invalidState("cannot truncate context \(contextLength) to \(length)")
        }
        setContext(length)
    }

    public func reset() { setContext(0) }

    private func setContext(_ length: Int) {
        contextLength = length
        unsafe state.contents().storeBytes(of: UInt32(length), as: UInt32.self)
    }

    /// Reads the GPU's context length and last acceptance back, once the
    /// work that set them has completed.
    func settle() -> (context: Int, accepted: Int) {
        let words = unsafe state.contents().bindMemory(to: UInt32.self, capacity: 2)
        contextLength = unsafe Int(words[0])
        return unsafe (contextLength, Int(words[1]))
    }

    /// The block's drafts after its anchor and each one's probability under
    /// the drafter's head, from the last completed draft of `count` rows.
    func drafts(count: Int) -> (tokens: [Int32], confidences: [Float]) {
        let tokens = unsafe block.contents().bindMemory(to: UInt32.self, capacity: count)
        let confidence = unsafe head.rowsConfidence.contents().bindMemory(to: Float.self, capacity: count)
        return ((1..<count).map { unsafe Int32(bitPattern: tokens[$0]) },
                (0..<(count - 1)).map { unsafe confidence[$0] })
    }

    /// The block's anchor, as written by the last `encodeAccept` or
    /// host-driven draft.
    var anchor: Int32 { unsafe Int32(bitPattern: block.contents().load(as: UInt32.self)) }

    // MARK: - Host-driven steps

    /// Adds the features of positions `[contextLength, contextLength + rows)`,
    /// which the target has written to `featureTap`, to the context. Queued
    /// on the target's queue, not waited.
    public func appendContext(rows: Int) throws {
        guard rows > 0 else { return }
        guard contextLength + rows <= maxContext, rows <= featureTap.capacity else {
            throw DFlashError.invalidState("context update of \(rows) rows at \(contextLength) exceeds capacity")
        }
        GPUTrace.setScope("draft append")
        let cb = try ctx.makeCommandBuffer()
        cb.label = "dflash context ctx=\(contextLength) rows=\(rows)"
        try encodeContextUpdate(cb, position: contextLength, rows: rows)
        cb.commit()
        setContext(contextLength + rows)
    }

    /// Drafts `count - 1` tokens after `anchor`, which sits at position
    /// `contextLength`, and waits for them. Greedy over the drafter's logits.
    public func propose(anchor: Int32, count: Int) throws -> [Int32] {
        let traceStart = GPUTrace.begin()
        defer { GPUTrace.end("draft_propose", since: traceStart) }
        guard contextLength + count <= maxContext else { throw DFlashError.invalidState("context full") }
        setAnchor(anchor)
        let cb = try ctx.makeCommandBuffer()
        cb.label = "dflash draft ctx=\(contextLength) rows=\(count)"
        try encodeDraft(cb, rows: count)
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw DFlashError.invalidState("draft failed: \(error)") }
        return drafts(count: count).tokens
    }

    /// Sets the block's anchor and mask rows for a host-driven draft.
    private func setAnchor(_ anchor: Int32) {
        let ids = unsafe block.contents().bindMemory(to: UInt32.self, capacity: Self.maxRows)
        unsafe ids[0] = UInt32(bitPattern: anchor)
        for r in 1..<Self.maxRows { unsafe ids[r] = UInt32(bitPattern: config.maskTokenID) }
    }

    // MARK: - Encoding

    /// The context update for positions `position..<position + rows`, in
    /// chunks of `maxRows`: fused features, `fc`, `hidden_norm`, then per
    /// layer the K|V projection into the cache.
    func encodeContextUpdate(_ cb: any MTLCommandBuffer, position: Int, rows: Int) throws {
        let D = config.hiddenSize, layers = config.contextLayers.count
        let tap = featureTap
        var done = 0
        while done < rows {
            let n = min(Self.maxRows, rows - done), p0 = position + done
            dispatch(cb, gatherPSO, [tap.buffer, feature],
                     [UInt32(n), UInt32(layers), UInt32(D), UInt32(p0), UInt32(tap.capacity)],
                     grid: MTLSize(width: layers * D, height: n, depth: 1))
            try project(cb, weights.fc, x: feature, y: contextHidden, rows: n)
            norm(cb, normBF16PSO, x: contextHidden, weight: weights.hiddenNorm, out: contextHidden, rows: n)
            for (l, layer) in weights.layers.enumerated() {
                try project(cb, layer.qkv.rows(config.qWidth..<config.qkvWidth), x: contextHidden, y: contextKV, rows: n)
                guard let enc = cb.makeTracedComputeEncoder() else { throw DFlashError.invalidState("encoder") }
                enc.setComputePipelineState(storeKVPSO)
                enc.setBuffer(contextKV, offset: 0, index: 0)
                enc.setBuffer(weights.buffer, offset: layer.kNorm, index: 1)
                enc.setBuffer(kCache[l], offset: 0, index: 2)
                enc.setBuffer(vCache[l], offset: 0, index: 3)
                enc.setValue(UInt32(config.kvHeads), index: 4)
                enc.setValue(UInt32(p0), index: 5)
                enc.setValue(UInt32(cacheCapacity[l]), index: 6)
                enc.setValue(config.ropeTheta, index: 7)
                enc.setValue(config.rmsEps, index: 8)
                enc.dispatchThreadgroups(MTLSize(width: config.kvHeads, height: n, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                enc.endEncoding()
            }
            done += n
        }
    }

    /// A draft of `rows` rows from `block` (anchor, then mask tokens) at
    /// position `state.context`, leaving the drafts in `block[1..<rows]` and
    /// their probabilities in the head's `rowsConfidence`.
    func encodeDraft(_ cb: any MTLCommandBuffer, rows R: Int) throws {
        guard R >= 2, R <= config.blockSize else {
            throw DFlashError.invalidState("block of \(R) outside 2...\(config.blockSize)")
        }
        GPUTrace.setScope("draft")
        let D = config.hiddenSize, F = config.intermediateSize
        let emb = model.embedding
        if let enc = cb.makeTracedComputeEncoder() {
            enc.setComputePipelineState(embedPSO)
            enc.setBuffer(emb.buffer, offset: Int(emb.offset), index: 0)
            enc.setBuffer(emb.buffer, offset: Int(emb.scaleOffset), index: 1)
            enc.setBuffer(emb.buffer, offset: Int(emb.biasOffset), index: 2)
            enc.setBuffer(block, offset: 0, index: 3)
            enc.setBuffer(residual, offset: 0, index: 4)
            enc.setValue(UInt32(D), index: 5)
            enc.setValue(Float(D).squareRoot(), index: 6)
            enc.dispatchThreads(MTLSize(width: D, height: R, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }
        norm(cb, normF32PSO, x: residual, weight: weights.layers[0].inputNorm, out: normed, rows: R)
        for (l, layer) in weights.layers.enumerated() {
            try project(cb, layer.qkv, x: normed, y: qkv, rows: R)
            if let enc = cb.makeTracedComputeEncoder() {
                enc.setComputePipelineState(qkNormRopePSO)
                enc.setBuffer(qkv, offset: 0, index: 0)
                enc.setBuffer(weights.buffer, offset: layer.qNorm, index: 1)
                enc.setBuffer(weights.buffer, offset: layer.kNorm, index: 2)
                enc.setBuffer(state, offset: 0, index: 3)
                enc.setValue(UInt32(config.heads), index: 4)
                enc.setValue(UInt32(config.kvHeads), index: 5)
                enc.setValue(config.ropeTheta, index: 6)
                enc.setValue(config.rmsEps, index: 7)
                enc.dispatchThreadgroups(MTLSize(width: config.heads + config.kvHeads, height: R, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                enc.endEncoding()
            }
            if let enc = cb.makeTracedComputeEncoder() {
                enc.setComputePipelineState(attentionPSO)
                enc.setBuffer(qkv, offset: 0, index: 0)
                enc.setBuffer(kCache[l], offset: 0, index: 1)
                enc.setBuffer(vCache[l], offset: 0, index: 2)
                enc.setBuffer(attention, offset: 0, index: 3)
                enc.setBuffer(state, offset: 0, index: 4)
                let params: [UInt32] = [UInt32(config.heads), UInt32(config.kvHeads), UInt32(R),
                                        config.slidingLayers[l] ? 1 : 0,
                                        UInt32(min(config.slidingWindow, Int(UInt32.max))),
                                        UInt32(cacheCapacity[l])]
                for i in params.indices { enc.setValue(params[i], index: 5 + i) }
                enc.dispatchThreadgroups(MTLSize(width: config.heads, height: R, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                enc.endEncoding()
            }
            try project(cb, layer.o, x: attention, y: delta, rows: R)
            addNorm(cb, weight: layer.postAttentionNorm, rows: R)
            try project(cb, layer.gateUp, x: normed, y: gateUp, rows: R)
            dispatch(cb, siluPSO, [gateUp, act], [UInt32(F)], grid: MTLSize(width: F, height: R, depth: 1))
            try project(cb, layer.down, x: act, y: delta, rows: R)
            if l + 1 < weights.layers.count {
                addNorm(cb, weight: weights.layers[l + 1].inputNorm, rows: R)
            }
        }
        // Rows 1..<R predict the drafts; the head applies the final norm.
        dispatch(cb, addHeadRowsPSO, [residual, delta, headRows], [UInt32((R - 1) * D), UInt32(D)],
                 grid: MTLSize(width: (R - 1) * D, height: 1, depth: 1))
        if let hot = hotVocabulary {
            try head.encodeHotGreedyRows(commandBuffer: cb, hidden: headRows, rowCount: R - 1,
                                         normWeight: weights.buffer, normOffset: weights.norm, table: hot,
                                         weights: emb.buffer, weightsOffset: Int(emb.offset),
                                         scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                                         biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                                         outIndices: picks, d: UInt32(D), rmsEps: config.rmsEps,
                                         softcap: Float(model.config.finalLogitSoftcap))
        } else {
            try head.encodeGreedyRows(commandBuffer: cb, hidden: headRows, rowCount: R - 1,
                                      normWeight: weights.buffer, normOffset: weights.norm,
                                      weights: emb.buffer, weightsOffset: Int(emb.offset),
                                      scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                                      biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                                      outTokens: picks, d: UInt32(D), vocab: UInt32(model.config.vocabSize),
                                      rmsEps: config.rmsEps, softcap: Float(model.config.finalLogitSoftcap))
        }
        dispatch(cb, finishPSO, [picks, hotVocabulary?.ids ?? picks, block],
                 [UInt32(R - 1), hotVocabulary == nil ? 0 : 1], grid: MTLSize(width: R - 1, height: 1, depth: 1))
    }

    /// After a verify of `rows` rows of `block` starting at position
    /// `state.context`, with the verify's greedy tokens in `verified`:
    /// advances the context past the anchor and the accepted drafts and
    /// anchors the next block at the target's next token. The verify's
    /// context rows must already be in the cache (`encodeContextUpdate`).
    func encodeAccept(_ cb: any MTLCommandBuffer, verified: any MTLBuffer, rows: Int) {
        guard let enc = cb.makeTracedComputeEncoder() else { return }
        enc.setComputePipelineState(acceptPSO)
        enc.setBuffer(block, offset: 0, index: 0)
        enc.setBuffer(verified, offset: 0, index: 1)
        enc.setBuffer(state, offset: 0, index: 2)
        enc.setValue(UInt32(rows), index: 3)
        enc.setValue(UInt32(bitPattern: config.maskTokenID), index: 4)
        enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding()
    }

    // MARK: - Helpers

    private func project(_ cb: any MTLCommandBuffer, _ m: DFlashMatrix, x: any MTLBuffer, y: any MTLBuffer,
                         rows: Int) throws {
        try int4.encodeRows(commandBuffer: cb,
                            weights: weights.buffer, weightsOffset: m.weights,
                            scales: weights.buffer, scalesOffset: m.scales,
                            biases: weights.buffer, biasesOffset: m.biases,
                            x: x, y: y, m: UInt32(m.rows), n: UInt32(m.columns), rows: rows,
                            xStrideElements: m.columns, yStrideElements: m.rows, types: .bf16)
    }

    private func dispatch(_ cb: any MTLCommandBuffer, _ pso: any MTLComputePipelineState, _ buffers: [any MTLBuffer],
                          _ constants: [UInt32], grid: MTLSize) {
        guard let enc = cb.makeTracedComputeEncoder() else { return }
        enc.setComputePipelineState(pso)
        for (i, b) in buffers.enumerated() { enc.setBuffer(b, offset: 0, index: i) }
        for i in constants.indices { enc.setValue(constants[i], index: buffers.count + i) }
        enc.dispatchThreads(grid, threadsPerThreadgroup: MTLSize(width: min(256, grid.width), height: 1, depth: 1))
        enc.endEncoding()
    }

    private func norm(_ cb: any MTLCommandBuffer, _ pso: any MTLComputePipelineState, x: any MTLBuffer,
                      weight: Int, out: any MTLBuffer, rows: Int) {
        guard let enc = cb.makeTracedComputeEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(weights.buffer, offset: weight, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
        enc.setValue(UInt32(config.hiddenSize), index: 3)
        enc.setValue(config.rmsEps, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// residual += delta, normed = rmsnorm(residual) * weight.
    private func addNorm(_ cb: any MTLCommandBuffer, weight: Int, rows: Int) {
        guard let enc = cb.makeTracedComputeEncoder() else { return }
        enc.setComputePipelineState(addNormPSO)
        enc.setBuffer(residual, offset: 0, index: 0)
        enc.setBuffer(delta, offset: 0, index: 1)
        enc.setBuffer(weights.buffer, offset: weight, index: 2)
        enc.setBuffer(normed, offset: 0, index: 3)
        enc.setValue(UInt32(config.hiddenSize), index: 4)
        enc.setValue(config.rmsEps, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }
}
