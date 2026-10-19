import Foundation
public import Metal

/// Chooses each step's draft length and verify length from measured cost,
/// online acceptance and the drafter's own confidence.
///
/// On a streamed MoE target the binding cost of speculation is verification:
/// every verified row pulls its own experts, so a block's cost grows with the
/// union of experts it touches, and rejected rows pay in full. The policy
/// therefore prices rows in measured wall time.
///
/// Before drafting, it models drafts as accepted independently at rate `a`
/// (DISCO-style, fitted from recent accept/reject counts), so a block of `k`
/// rows yields (1 - a^k) / (1 - a) tokens, and drafts the length that
/// maximizes expected tokens per second, or none: length 1 is a plain decode
/// step, which still feeds the drafter's context. While plain wins it retries
/// a short block after `explore` plain steps, doubling that interval each time
/// the retry does not pay.
///
/// After drafting, it truncates the block DSpark-style: each draft's softmax
/// probability under the target head, calibrated online against what verify
/// accepts, gives the expected yield of every verify length, and the length
/// with the largest yield net of its verify time at the current throughput is
/// verified. A block whose drafts look unlikely is not verified at all.
public struct SpeculationPolicy: Sendable {
    public let maxBlock: Int
    /// Decayed counts of accepted drafts and of first rejections.
    private var accepts = 8.5
    private var rejects = 1.5
    private let decay = 0.97
    /// Decayed (accepted, seen) verified drafts per confidence decile.
    private var calibrationAccepted: [Double]
    private var calibrationSeen: [Double]
    private let calibrationDecay = 0.995
    public private(set) var decodeSeconds = 0.030
    /// Seconds per drafter call by block length (anchor plus drafts).
    public private(set) var draftSeconds: [Double]
    /// Seconds per verify by verified rows.
    public private(set) var verifySeconds: [Double]
    /// Recent realized tokens per second, the value of time in truncation.
    public private(set) var throughput = 1 / 0.030
    private let smoothing = 0.2
    private static let minExplore = 8, maxExplore = 128
    private var explore = SpeculationPolicy.minExplore
    private var plainSinceSpeculation = 0

    public init(maxBlock: Int) {
        precondition(maxBlock >= 2)
        self.maxBlock = maxBlock
        // Priors measured on an M5 (16 GB): ~30 ms per decode step, a drafter
        // call of ~4 ms + 1.2 ms per row, a verify of ~26 ms + 5.5 ms per row.
        draftSeconds = (0...maxBlock).map { Self.priorDraft($0) }
        verifySeconds = (0...maxBlock).map { Self.priorVerify($0) }
        // Two pseudo-observations per decile at its midpoint: trust the
        // drafter's probability until verify says otherwise.
        calibrationAccepted = (0..<10).map { (Double($0) + 0.5) / 10 * 2 }
        calibrationSeen = Array(repeating: 2, count: 10)
    }

    private static func priorDraft(_ length: Int) -> Double { 0.004 + 0.0012 * Double(length) }
    private static func priorVerify(_ length: Int) -> Double { 0.026 + 0.0055 * Double(length) }

    /// The per-draft acceptance rate the draft-length planner assumes.
    public var acceptanceRate: Double { accepts / (accepts + rejects) }

    /// Expected tokens a block of `length` rows yields at the planner's rate:
    /// the anchor's own next token plus each draft whose prefix survives.
    public func expectedTokens(length: Int) -> Double {
        let a = min(acceptanceRate, 0.999)
        return (1 - pow(a, Double(length))) / (1 - a)
    }

    /// P(a draft is accepted | its prefix was) for a drafter confidence.
    public func calibrated(_ confidence: Float) -> Double {
        let bin = min(9, max(0, Int(confidence * 10)))
        return calibrationAccepted[bin] / calibrationSeen[bin]
    }

    /// The block length to draft next, 1 for a plain decode step. `limit`
    /// caps it, e.g. by remaining context.
    public mutating func next(limit: Int) -> Int {
        let upper = min(maxBlock, limit)
        var best = 1
        var bestRate = 1 / decodeSeconds
        if upper >= 2 {
            for length in 2...upper {
                let rate = expectedTokens(length: length) / (draftSeconds[length] + verifySeconds[length])
                if rate > bestRate { best = length; bestRate = rate }
            }
        }
        if best > 1 {
            plainSinceSpeculation = 0
            explore = Self.minExplore
            return best
        }
        plainSinceSpeculation += 1
        guard upper >= 4, plainSinceSpeculation >= explore else { return 1 }
        plainSinceSpeculation = 0
        explore = min(explore * 2, Self.maxExplore)
        return 4
    }

    /// Rows to verify for drafts with `confidences` (anchor included, so 1
    /// means a plain decode step instead): the length whose expected tokens,
    /// less its verify time valued at the current throughput, is largest.
    public func verifyLength(confidences: [Float]) -> Int {
        var best = 1
        var bestValue = 1 - throughput * decodeSeconds
        var expected = 1.0, survive = 1.0
        for (i, confidence) in confidences.prefix(maxBlock - 1).enumerated() {
            survive *= calibrated(confidence)
            expected += survive
            let value = expected - throughput * verifySeconds[i + 2]
            if value > bestValue { best = i + 2; bestValue = value }
        }
        return best
    }

    public mutating func recordPlain(seconds: Double) {
        decodeSeconds += smoothing * (seconds - decodeSeconds)
    }

    public mutating func recordDraft(length: Int, seconds: Double) {
        Self.update(&draftSeconds, at: length, seconds: seconds, prior: Self.priorDraft, smoothing: smoothing)
    }

    /// Records a verify of `rows` rows that took `seconds` and accepted
    /// `accepted` of its `rows - 1` drafts, whose confidences were
    /// `confidences`, and the whole step's tokens and seconds.
    public mutating func recordVerify(rows: Int, seconds: Double, accepted: Int, confidences: [Float],
                                      stepTokens: Int, stepSeconds: Double) {
        Self.update(&verifySeconds, at: rows, seconds: seconds, prior: Self.priorVerify, smoothing: smoothing)
        accepts = decay * accepts + Double(accepted)
        rejects = decay * rejects + (accepted < rows - 1 ? 1 : 0)
        for i in 0..<min(accepted + 1, rows - 1) {
            let bin = min(9, max(0, Int(confidences[i] * 10)))
            for b in 0..<10 { calibrationAccepted[b] *= calibrationDecay; calibrationSeen[b] *= calibrationDecay }
            calibrationAccepted[bin] += i < accepted ? 1 : 0
            calibrationSeen[bin] += 1
        }
        recordThroughput(tokens: stepTokens, seconds: stepSeconds)
    }

    public mutating func recordThroughput(tokens: Int, seconds: Double) {
        guard seconds > 0 else { return }
        throughput += 0.05 * (Double(tokens) / seconds - throughput)
    }

    /// EMA at `length`; other lengths follow its ratio to the prior.
    private static func update(_ table: inout [Double], at length: Int, seconds: Double,
                               prior: (Int) -> Double, smoothing: Double) {
        table[length] += smoothing * (seconds - table[length])
        let ratio = table[length] / prior(length)
        for other in 2..<table.count where other != length {
            table[other] += 0.05 * (ratio * prior(other) - table[other])
        }
    }
}

/// Counters for one speculative generation.
public struct SpeculationStats: Sendable {
    public var plainSteps = 0
    public var speculativeSteps = 0
    /// Drafted blocks verified shorter than drafted, or not at all.
    public var truncatedSteps = 0
    public var draftedTokens = 0
    public var acceptedDrafts = 0
    public var draftSeconds = 0.0
    public var verifySeconds = 0.0
    public var contextSeconds = 0.0

    public var profile: [String: Any] {
        ["plain_steps": plainSteps, "speculative_steps": speculativeSteps, "truncated_steps": truncatedSteps,
         "drafted_tokens": draftedTokens, "accepted_drafts": acceptedDrafts,
         "mean_accepted_per_speculative_step": speculativeSteps > 0
            ? Double(acceptedDrafts) / Double(speculativeSteps) : 0,
         "draft_seconds": draftSeconds, "verify_seconds": verifySeconds,
         "context_seconds": contextSeconds]
    }
}

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
public final class SpeculativeDecoder: LogitProducer, ChunkedPrefillRunner, ContextWindowReporting,
    FusedGreedyProducer {
    public let runner: RealForwardRunner
    public let drafter: DFlashDrafter
    public private(set) var policy: SpeculationPolicy
    public private(set) var stats = SpeculationStats()
    public private(set) var lastGreedyToken: UInt32 = 0
    public var usesFusedGreedyHead: Bool { true }
    public var maxContext: Int { min(runner.maxContext, drafter.maxContext) }

    /// Accepted drafts not yet handed out: (position, token, its successor).
    private var accepted: [(position: Int, token: Int32, next: UInt32)] = []
    /// Captured feature rows from plain steps, appended to the drafter lazily.
    private var pendingContext: [Float16] = []
    private var pendingContextRows = 0

    public init(runner: RealForwardRunner, drafter: DFlashDrafter, maxBlock: Int? = nil) throws {
        guard runner.usesFusedGreedyHead else {
            throw DFlashError.invalidState("speculative decoding needs the fused greedy head")
        }
        self.runner = runner
        self.drafter = drafter
        policy = SpeculationPolicy(maxBlock: min(maxBlock ?? drafter.config.blockSize,
                                                 drafter.config.blockSize,
                                                 RealForwardRunner.maxVerifyTokens))
    }

    public func reset() {
        runner.reset()
        drafter.reset()
        accepted.removeAll()
        pendingContext.removeAll()
        pendingContextRows = 0
    }

    func prefillChunked(tokens: ArraySlice<Int32>, startPosition: Int, outputMode: PrefillOutputMode,
                        config: PrefillRuntimeConfig, into logits: any MTLBuffer,
                        onProgress: (Int) -> Void) async throws -> PrefillResult {
        var failure: any Error?
        let drafter = self.drafter
        runner.prefillCapture = PrefillCapture(hiddenLayers: drafter.config.contextLayers) { chunk in
            do {
                try drafter.appendContext(captured: chunk.hidden, stride: chunk.tokenCount, rows: chunk.tokenCount)
            } catch { failure = error }
        }
        defer { runner.prefillCapture = nil }
        let result = try await runner.prefillChunked(tokens: tokens, startPosition: startPosition,
                                                     outputMode: outputMode, config: config, into: logits,
                                                     onProgress: onProgress)
        if let failure { throw failure }
        if case .greedyToken(let token) = result.seed { lastGreedyToken = token }
        return result
    }

    public func produce(token: Int32, position: Int, into logits: any MTLBuffer) async throws {
        if let next = accepted.first, next.position == position, next.token == token {
            accepted.removeFirst()
            lastGreedyToken = next.next
            return
        }
        accepted.removeAll()
        if runner.kvPosition > position {
            try runner.rewind(to: position)
            try flushContext()
            try drafter.truncateContext(to: position)
        }
        let length = policy.next(limit: maxContext - position)
        let started = Date()
        if length == 1 {
            try await plainStep(token: token, position: position, into: logits)
            policy.recordPlain(seconds: Date().timeIntervalSince(started))
            policy.recordThroughput(tokens: 1, seconds: Date().timeIntervalSince(started))
            return
        }
        try flushContext()
        let draftStart = Date()
        let drafts = try drafter.propose(anchor: token, count: length)
        let confidences = drafter.lastConfidences
        let draftSeconds = Date().timeIntervalSince(draftStart)
        stats.draftSeconds += draftSeconds
        policy.recordDraft(length: length, seconds: draftSeconds)
        stats.draftedTokens += drafts.count
        let rows = policy.verifyLength(confidences: confidences)
        if rows < length { stats.truncatedSteps += 1 }
        if rows == 1 {
            let plainStart = Date()
            try await plainStep(token: token, position: position, into: logits)
            policy.recordPlain(seconds: Date().timeIntervalSince(plainStart))
            policy.recordThroughput(tokens: 1, seconds: Date().timeIntervalSince(started))
            return
        }
        let verifyStart = Date()
        let acceptedCount = try await verifyStep(anchor: token, drafts: Array(drafts.prefix(rows - 1)),
                                                 position: position)
        policy.recordVerify(rows: rows, seconds: Date().timeIntervalSince(verifyStart), accepted: acceptedCount,
                            confidences: confidences, stepTokens: acceptedCount + 1,
                            stepSeconds: Date().timeIntervalSince(started))
    }

    private func plainStep(token: Int32, position: Int, into logits: any MTLBuffer) async throws {
        let layers = drafter.config.contextLayers.count, D = drafter.config.hiddenSize
        runner.prefillCapture = PrefillCapture(hiddenLayers: drafter.config.contextLayers) { [self] chunk in
            pendingContext.append(contentsOf: chunk.hidden.prefix(layers * D))
            pendingContextRows += 1
        }
        defer { runner.prefillCapture = nil }
        try await runner.produce(token: token, position: position, into: logits)
        lastGreedyToken = runner.lastGreedyToken
        stats.plainSteps += 1
    }

    /// Hands plain steps' captured rows to the drafter, re-laid out from
    /// `[row][layer][D]` to the `[layer][row][D]` it reads.
    private func flushContext() throws {
        guard pendingContextRows > 0 else { return }
        let started = Date()
        let layers = drafter.config.contextLayers.count, D = drafter.config.hiddenSize
        let rows = pendingContextRows
        var laid = [Float16](repeating: 0, count: rows * layers * D)
        for row in 0..<rows {
            for layer in 0..<layers {
                let src = (row * layers + layer) * D, dst = (layer * rows + row) * D
                laid[dst..<(dst + D)] = pendingContext[src..<(src + D)]
            }
        }
        pendingContext.removeAll(keepingCapacity: true)
        pendingContextRows = 0
        try laid.withUnsafeBufferPointer {
            try drafter.appendContext(captured: $0, stride: rows, rows: rows)
        }
        stats.contextSeconds += Date().timeIntervalSince(started)
    }

    /// Verifies `drafts` after `anchor` in one chunk and keeps the longest
    /// prefix the target agrees with, plus the target's next token.
    private func verifyStep(anchor: Int32, drafts: [Int32], position: Int) async throws -> Int {
        let length = drafts.count + 1
        var verifyHidden: [Float16] = []
        runner.prefillCapture = PrefillCapture(hiddenLayers: drafter.config.contextLayers) { chunk in
            verifyHidden = Array(chunk.hidden)
        }
        defer { runner.prefillCapture = nil }
        var started = Date()
        let verified = try await runner.verify(tokens: [anchor] + drafts, startPosition: position)
        stats.verifySeconds += Date().timeIntervalSince(started)

        var count = 0
        while count < drafts.count, drafts[count] == verified[count] { count += 1 }
        // Rows [0, count] (anchor and accepted drafts) are now context.
        if count + 1 < length { try runner.rewind(to: position + count + 1) }
        started = Date()
        try verifyHidden.withUnsafeBufferPointer {
            try drafter.appendContext(captured: $0, stride: length, rows: count + 1)
        }
        stats.contextSeconds += Date().timeIntervalSince(started)

        lastGreedyToken = UInt32(bitPattern: verified[0])
        accepted = (0..<count).map { i in
            (position: position + 1 + i, token: drafts[i], next: UInt32(bitPattern: verified[i + 1]))
        }
        stats.speculativeSteps += 1
        stats.acceptedDrafts += count
        return count
    }
}
