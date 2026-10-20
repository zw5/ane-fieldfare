import Foundation
public import Metal

/// Greedy DFlash speculative decoding behind the ordinary producer interface,
/// so the shared generation loop's stopping, detokenizing and streaming apply
/// unchanged.
///
/// `produce(token:position:)` either pops a token the last verify already
/// accepted, whose KV row and drafter context row are in place, or runs a new
/// step anchored at `token`: a plain decode step or a drafted block verified
/// in one chunk, as `SpeculationPolicy` decides. The emitted tokens equal
/// plain greedy decoding up to the numerical difference between a batched
/// verify chunk and single-row decode.
///
/// A step's GPU work runs back to back with one host wait: the verify reads
/// its tokens from the drafter's block, and behind its final head the
/// drafter adds the verified rows to its context, accepts on the GPU and
/// drafts the next block from the target's next token. The host reads the
/// outcome and the next drafts together once all of it completes.
public final class SpeculativeDecoder: LogitProducer, ChunkedPrefillRunner, ContextWindowReporting,
    SamplingTokenProducer, PrefixReusingProducer {
    public let runner: RealForwardRunner
    public let drafter: DFlashDrafter
    public private(set) var policy: SpeculationPolicy
    public private(set) var stats = SpeculationStats()
    public private(set) var lastGreedyToken: UInt32 = 0
    public var usesFusedGreedyHead: Bool { true }
    /// With a rule set, every token is drawn under it with noise keyed by
    /// its position, by the target and the drafter alike: a draft is kept
    /// when it equals the target's draw, so the output is the target's own
    /// sampled text for the seed, as plain decoding would draw it.
    public var tokenSampling: TokenSampling? {
        didSet {
            runner.tokenSampling = tokenSampling
            chained = nil
        }
    }
    public var maxContext: Int { min(runner.maxContext, drafter.maxContext) }

    /// Accepted drafts not yet handed out: (position, token, its successor).
    private var accepted: [(position: Int, token: Int32, next: UInt32)] = []
    /// The last verify's greedy token per row, which the GPU accepts from.
    private let verified: any MTLBuffer
    /// A completed draft queued behind the last verify: anchored at
    /// `position`, `rows` rows, with its drafts and their confidences.
    private var chained: (position: Int, rows: Int, drafts: [Int32], confidences: [Float])?
    /// The step at `position` the last verify planned as a plain step.
    private var plannedPlain: Int?
    /// Feature rows past the drafter's context build up during plain steps;
    /// they are added before the ring could wrap onto them.
    private var backlogLimit: Int { drafter.featureTap.capacity - RealForwardRunner.maxVerifyTokens }

    public init(runner: RealForwardRunner, drafter: DFlashDrafter, maxBlock: Int? = nil) throws {
        guard runner.usesFusedGreedyHead else {
            throw DFlashError.invalidState("speculative decoding needs the fused greedy head")
        }
        guard let verified = runner.device.makeBuffer(length: RealForwardRunner.maxVerifyTokens * 4,
                                                      options: .storageModeShared) else {
            throw DFlashError.invalidState("verify token allocation failed")
        }
        verified.label = "dflash.verified"
        self.runner = runner
        self.drafter = drafter
        self.verified = verified
        policy = SpeculationPolicy(maxBlock: min(maxBlock ?? drafter.config.blockSize,
                                                 drafter.config.blockSize,
                                                 RealForwardRunner.maxVerifyTokens))
        runner.draftFeatureTap = drafter.featureTap
    }

    deinit {
        // A newer decoder on the same runner may have installed its own tap.
        if runner.draftFeatureTap?.buffer === drafter.featureTap.buffer { runner.draftFeatureTap = nil }
    }

    public func reset() {
        runner.reset()
        drafter.reset()
        accepted.removeAll()
        chained = nil
        plannedPlain = nil
    }

    public func canReuse(prefix position: Int) -> Bool { runner.canRewind(to: position) }

    /// Keeps the target's and the drafter's first `position` tokens; the
    /// next prefill starts there.
    public func rewindForReuse(to position: Int) throws {
        accepted.removeAll()
        chained = nil
        plannedPlain = nil
        try runner.rewindForReuse(to: position)
        if drafter.contextLength > position { try drafter.truncateContext(to: position) }
    }

    public func prefillChunked(tokens: ArraySlice<Int32>, startPosition: Int, outputMode: PrefillOutputMode,
                        config: PrefillRuntimeConfig, into logits: any MTLBuffer,
                        onProgress: (Int) -> Void) async throws -> PrefillResult {
        // Whatever was queued or accepted past `startPosition` belongs to an
        // abandoned continuation.
        chained = nil
        plannedPlain = nil
        accepted.removeAll()
        if drafter.contextLength > startPosition { try drafter.truncateContext(to: startPosition) }
        try catchUp(to: startPosition)
        // A chunk's rows all land in the ring before the drafter reads them.
        try setRing(rows: min(config.chunkTokens, PrefillRuntimeConfig.maxChunkTokens))
        defer { try? setRing(rows: DFlashDrafter.baseFeatureRows) }
        // A prompt's tokens are likely drafts.
        drafter.hotVocabulary?.insert(tokens.map { UInt32(bitPattern: $0) })
        var failure: (any Error)?
        let result = try await runner.prefillChunked(
            tokens: tokens, startPosition: startPosition, outputMode: outputMode, config: config,
            into: logits, onProgress: { [drafter] completed in
                // Each finished chunk's features are in the ring; queue them.
                do { try drafter.appendContext(rows: startPosition + completed - drafter.contextLength) }
                catch { failure = failure ?? error }
                onProgress(completed)
            })
        if let failure { throw failure }
        if case .greedyToken(let token) = result.seed { lastGreedyToken = token }
        return result
    }

    private func setRing(rows: Int) throws {
        if try drafter.resizeFeatureRing(rows: rows) { runner.draftFeatureTap = drafter.featureTap }
    }

    /// Adds every feature row the target has written past the drafter's
    /// context, up to `position`.
    private func catchUp(to position: Int) throws {
        let rows = position - drafter.contextLength
        guard rows > 0 else { return }
        let started = Date()
        try drafter.appendContext(rows: rows)
        stats.contextSeconds += Date().timeIntervalSince(started)
    }

    public func produce(token: Int32, position: Int, into logits: any MTLBuffer) async throws {
        if let next = accepted.first, next.position == position, next.token == token {
            accepted.removeFirst()
            lastGreedyToken = next.next
            return
        }
        accepted.removeAll()
        if runner.kvPosition > position { try runner.rewind(to: position) }
        if drafter.contextLength > position { try drafter.truncateContext(to: position) }
        let started = Date()
        // A draft queued behind the last verify is this step's when it is
        // anchored here.
        var draft = chained.flatMap { $0.position == position && drafter.anchor == token ? $0 : nil }
        let planned = plannedPlain == position
        chained = nil
        plannedPlain = nil
        if draft == nil {
            let length = planned ? 1 : policy.next(limit: maxContext - position)
            if length == 1 {
                try await plainStep(token: token, position: position, into: logits)
                policy.recordPlain(seconds: Date().timeIntervalSince(started))
                policy.recordThroughput(tokens: 1, seconds: Date().timeIntervalSince(started))
                return
            }
            try catchUp(to: position)
            let draftStart = Date()
            let tokens = try drafter.propose(anchor: token, count: length, sampling: tokenSampling)
            let seconds = Date().timeIntervalSince(draftStart)
            stats.draftSeconds += seconds
            policy.recordDraft(length: length, seconds: seconds)
            draft = (position, length, tokens, drafter.drafts(count: length).confidences)
        }
        guard let draft else { return }
        stats.draftedTokens += draft.drafts.count
        let rows = policy.verifyLength(confidences: draft.confidences)
        if rows < draft.rows { stats.truncatedSteps += 1 }
        if rows == 1 {
            try await plainStep(token: token, position: position, into: logits)
            policy.recordPlain(seconds: Date().timeIntervalSince(started))
            policy.recordThroughput(tokens: 1, seconds: Date().timeIntervalSince(started))
            return
        }
        try await verifyStep(draft: draft, rows: rows, started: started)
    }

    private func plainStep(token: Int32, position: Int, into logits: any MTLBuffer) async throws {
        try await runner.produce(token: token, position: position, into: logits)
        lastGreedyToken = runner.lastGreedyToken
        stats.plainSteps += 1
        // The step's feature row is in the ring; add the backlog before the
        // ring wraps onto it.
        if position + 1 - drafter.contextLength >= backlogLimit { try catchUp(to: position + 1) }
    }

    /// Verifies the first `rows` rows of `draft` (anchor and drafts) in one
    /// block and keeps the longest prefix the target agrees with, plus the
    /// target's next token; queues the next draft behind the verify.
    private func verifyStep(draft: (position: Int, rows: Int, drafts: [Int32], confidences: [Float]),
                            rows: Int, started: Date) async throws {
        let position = draft.position
        var nextLength = 1
        var update: (any MTLCommandBuffer)?, draftCB: (any MTLCommandBuffer)?
        let verifyStart = Date()
        // Behind the head: this block's rows into the drafter's context, the
        // acceptance, and the next block's draft from the target's next
        // token, which the next step verifies if it is anchored there. The
        // verify waits for the last of them, so the step has one host wait.
        try await runner.verifyBlock(rows: rows, startPosition: position, tokens: drafter.block, greedy: verified) {
            [self] in
            nextLength = policy.next(limit: maxContext - position - rows)
            let cb = try runner.makeCommandBuffer()
            cb.label = "dflash context+accept ctx=\(position) rows=\(rows)"
            try drafter.encodeContextUpdate(cb, position: position, rows: rows)
            try drafter.encodeAccept(cb, verified: verified, rows: rows)
            cb.commit()
            update = cb
            guard nextLength >= 2 else { return cb }
            let draft = try runner.makeCommandBuffer()
            draft.label = "dflash draft rows=\(nextLength)"
            try drafter.encodeDraft(draft, rows: nextLength, sampling: tokenSampling)
            draft.commit()
            draftCB = draft
            return draft
        }
        let stepSeconds = Date().timeIntervalSince(verifyStart)
        guard let update else { throw DFlashError.invalidState("verify ran no chained work") }
        for cb in [update, draftCB].compactMap({ $0 }) {
            if let error = cb.error { throw DFlashError.invalidState("chained drafter work failed: \(error)") }
        }
        // The verify's own time, without the drafter work queued behind it.
        let draftGPU = draftCB.map { max(0, $0.gpuEndTime - $0.gpuStartTime) } ?? 0
        let verifySeconds = max(0, stepSeconds - draftGPU - max(0, update.gpuEndTime - update.gpuStartTime))
        stats.verifySeconds += verifySeconds
        stats.contextSeconds += max(0, update.gpuEndTime - update.gpuStartTime)

        let out = unsafe verified.contents().bindMemory(to: UInt32.self, capacity: rows)
        let greedy = (0..<rows).map { unsafe Int32(bitPattern: out[$0]) }
        var count = 0
        while count < rows - 1, draft.drafts[count] == greedy[count] { count += 1 }
        let (context, gpuAccepted) = drafter.settle()
        guard gpuAccepted == count, context == position + count + 1 else {
            throw DFlashError.invalidState(
                "GPU accepted \(gpuAccepted) at context \(context); host \(count) at \(position + count + 1)")
        }
        // Rows [0, count] (anchor and accepted drafts) are now context.
        if count + 1 < rows { try runner.rewind(to: position + count + 1) }
        lastGreedyToken = UInt32(bitPattern: greedy[0])
        drafter.hotVocabulary?.insert(greedy.prefix(count + 1).map { UInt32(bitPattern: $0) })
        accepted = (0..<count).map { i in
            (position: position + 1 + i, token: draft.drafts[i], next: UInt32(bitPattern: greedy[i + 1]))
        }
        stats.speculativeSteps += 1
        stats.acceptedDrafts += count
        stats.draftSeconds += draftGPU
        if nextLength >= 2 {
            policy.recordDraft(length: nextLength, seconds: draftGPU)
            let next = drafter.drafts(count: nextLength)
            chained = (position + count + 1, nextLength, next.tokens, next.confidences)
        } else {
            plannedPlain = position + count + 1
        }
        policy.recordVerify(rows: rows, seconds: verifySeconds, accepted: count,
                            confidences: draft.confidences, stepTokens: count + 1,
                            stepSeconds: Date().timeIntervalSince(started))
    }
}
