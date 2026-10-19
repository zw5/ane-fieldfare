import Foundation
import Metal

/// Streaming callbacks from `runRawCompletion`. `.prefill` reports monotonic
/// producer-defined prompt progress; scalar replay reports per token, while a
/// prefill-capable producer may report per internal chunk. `.token` fires per
/// decoded non-stop token; `.tail` carries the detokenizer flush remainder at a
/// stop boundary.
public enum RawDecodeProgress: Sendable {
    case prefill(done: Int, total: Int)
    case token(index: Int, id: Int32, delta: String)
    case tail(String)
}

struct RawPrefillOutcome {
    let position: Int
    let seed: PrefillSeed?
    let prefillSeconds: Double
    let history: [Int32]
}

/// The prefill half of `runRawCompletion`.
func runRawPrefill(
    producer: some LogitProducer,
    promptIds: [Int32],
    multimodalInput: MultimodalPrefillInput?,
    prefillConfig requestedPrefillConfig: PrefillRuntimeConfig,
    outputMode: PrefillOutputMode,
    isPureGreedy: Bool,
    historyReserve: Int,
    scratch: RawCompletionScratch,
    onProgress: (RawDecodeProgress) -> Void
) async throws -> RawPrefillOutcome {
    if let multimodalInput {
        guard multimodalInput.effectiveTokenIDs == promptIds else {
            throw PrefillError.chunkedUnsupported(
                "multimodal effective token IDs do not match the prompt")
        }
    }
    // Image spans are served only by the complete chunked prefill path, and a
    // prefill config that cannot serve them is a performance setting, not a
    // decision to drop the images. Coercing here rather than at each call site
    // is what makes it total: the default argument is `.current`, which the
    // environment can turn `.off`, so a caller that passes no config at all
    // reaches this too. The call sites still coerce first, to report it.
    let prefillConfig = multimodalInput == nil
        ? requestedPrefillConfig
        : (requestedPrefillConfig.coercedForImagePrompt() ?? requestedPrefillConfig)

    func checkSeed(_ seed: PrefillSeed, path: String, requiresWrittenLogits: Bool) throws {
        if requiresWrittenLogits, outputMode == .logits, seed != .logitsWritten {
            throw PrefillError.unsupportedPrefillSeed(
                "RawCompletion \(path) prefill requested logits but producer returned \(seed)")
        }
        if case .greedyToken = seed, !isPureGreedy {
            throw PrefillError.unsupportedPrefillSeed(
                "RawCompletion \(path) prefill returned a greedy token for a sampling config")
        }
    }

    var history: [Int32] = []
    history.reserveCapacity(historyReserve)

    producer.reset()
    let prefillStart = Date()
    var position = 0
    var prefillSeed: PrefillSeed?
    let prefillTokens = promptIds[...]
    switch (multimodalInput, prefillConfig.mode) {
    case (.some(let input), .chunked):
        // The coercion above leaves an image prompt in chunked mode, so a
        // producer that cannot run image spans is the only way to fail here —
        // and no config change can fix that.
        guard let multimodal = producer as? any MultimodalPrefillRunner else {
            throw PrefillError.chunkedUnsupported(
                "multimodal prefill requires a MultimodalPrefillRunner-backed runtime")
        }
        let result = try await multimodal.prefillMultimodal(
            input: input,
            startPosition: position,
            outputMode: outputMode,
            config: prefillConfig,
            into: scratch.logits
        ) { done in
            onProgress(.prefill(done: done, total: promptIds.count))
        }
        try checkSeed(result.seed, path: "multimodal", requiresWrittenLogits: true)
        position = result.newPosition
        prefillSeed = result.seed
        history.append(contentsOf: prefillTokens)
    case (.some, _):
        throw PrefillError.chunkedUnsupported(
            "multimodal prefill requires a MultimodalPrefillRunner-backed runtime")
    case (.none, .chunked):
        guard let chunked = producer as? any ChunkedPrefillRunner else {
            throw PrefillError.chunkedUnsupported(
                PrefillError.chunkedRequiresChunkedRunnerReason)
        }
        let result = try await chunked.prefillChunked(tokens: prefillTokens,
                                                      startPosition: position,
                                                      outputMode: outputMode,
                                                      config: prefillConfig,
                                                      into: scratch.logits) { done in
            onProgress(.prefill(done: done, total: promptIds.count))
        }
        try checkSeed(result.seed, path: "chunked", requiresWrittenLogits: true)
        position = result.newPosition
        prefillSeed = result.seed
        history.append(contentsOf: prefillTokens)
    case (.none, _):
        for t in prefillTokens {
            try Task.checkCancellation()
            try await producer.produce(token: t, position: position, into: scratch.logits)
            position += 1
            history.append(t)
            onProgress(.prefill(done: position, total: promptIds.count))
        }
    }

    return RawPrefillOutcome(
        position: position,
        seed: prefillSeed,
        prefillSeconds: Date().timeIntervalSince(prefillStart),
        history: history)
}

public struct RawDecodeResult: Sendable {
    public let prefillTokens: Int
    public let prefillSeconds: Double
    public let newTokens: Int
    public let decodeSeconds: Double
    public let reason: StopReason
    public let kvPosition: Int
    public let kvBackedTokenIDs: [Int32]
    public let uncommittedBoundaryTokenIDs: [Int32]
    public let withheldTrailingKVTokens: Int

    public init(prefillTokens: Int,
                prefillSeconds: Double,
                newTokens: Int,
                decodeSeconds: Double,
                reason: StopReason,
                kvPosition: Int,
                kvBackedTokenIDs: [Int32],
                uncommittedBoundaryTokenIDs: [Int32],
                withheldTrailingKVTokens: Int = 0) {
        self.prefillTokens = prefillTokens
        self.prefillSeconds = prefillSeconds
        self.newTokens = newTokens
        self.decodeSeconds = decodeSeconds
        self.reason = reason
        self.kvPosition = kvPosition
        self.kvBackedTokenIDs = kvBackedTokenIDs
        self.uncommittedBoundaryTokenIDs = uncommittedBoundaryTokenIDs
        self.withheldTrailingKVTokens = withheldTrailingKVTokens
    }
}

/// Preallocated per-generation buffers (two 512 KiB vocab buffers plus a token
/// slot) and sampler. A warm session reuses them for every token, avoiding
/// per-token Metal buffer allocation. Not `Sendable`: one generation owns it
/// at a time, inside that generation's task.
public struct RawCompletionScratch {
    let logits: any MTLBuffer
    let probs: any MTLBuffer
    let outToken: any MTLBuffer
    let sampler: Sampler

    public init(context: MetalContext, vocab: Int) throws {
        guard let logits = context.device.makeBuffer(length: vocab * MemoryLayout<Float16>.size,
                                                     options: .storageModeShared),
              let probs = context.device.makeBuffer(length: vocab * MemoryLayout<Float16>.size,
                                                    options: .storageModeShared),
              let outToken = context.device.makeBuffer(length: MemoryLayout<UInt32>.size,
                                                       options: .storageModeShared)
        else {
            throw ModelError.residentBufferWrapFailed
        }
        self.logits = logits
        self.probs = probs
        self.outToken = outToken
        self.sampler = try Sampler(context: context, vocab: vocab)
    }
}

extension GenerationConfig {
    /// A pure-greedy config can use the fused head's GPU argmax
    /// (`RealForwardRunner.lastGreedyToken`) instead of sampling from the
    /// logits buffer. Anything else needs real logits.
    public var isPureGreedy: Bool {
        temperature == 0 && repetitionPenalty == 1
    }

}

/// Raw-completion prefill + decode loop shared by the CLI and the Mac app.
/// Consumes pre-encoded `promptIds` (BOS + verbatim encode upstream — no chat
/// template). Stop handling, detokenizer flush ordering, and history append
/// ordering are shared by both front ends.
///
/// When the producer runs the fused lm_head (`RealForwardRunner` default) the
/// logits buffer is never written; the loop then requires a pure-greedy config
/// and reads `lastGreedyToken`. Callers with sampling configs must construct
/// the runner with `forceLogitsHead: true`.
public func runRawCompletion(producer: some LogitProducer,
                             tokenizer: GFTokenizer,
                             promptIds: [Int32],
                             multimodalInput: MultimodalPrefillInput? = nil,
                             config: GenerationConfig,
                             context: MetalContext,
                             scratch: RawCompletionScratch,
                             prefillConfig: PrefillRuntimeConfig = .defaultChunked,
                             shouldStop: () -> Bool = { false },
                             onProgress: (RawDecodeProgress) -> Void) async throws -> RawDecodeResult {
    try config.validate()
    guard !promptIds.isEmpty else {
        throw GeneratorError.emptyPrompt
    }
    // Present exactly when the producer's fused head leaves a GPU argmax
    // instead of logits.
    let fusedGreedyRunner = (producer as? any FusedGreedyProducer).flatMap {
        $0.usesFusedGreedyHead ? $0 : nil
    }
    let fusedGreedy = fusedGreedyRunner != nil
    guard !fusedGreedy || config.isPureGreedy else {
        throw PrefillError.unsupportedPrefillSeed(
            "the fused-head producer cannot serve this sampling configuration; use a logits head")
    }

    var detok = GFDetokenizer(tokenizer: tokenizer,
                              barrierTokenIDs: tokenizer.structuralMarkerIDs)

    if let context = producer as? any ContextWindowReporting,
       promptIds.count + config.maxNewTokens > context.maxContext {
        throw GeneratorError.contextOverflow(prompt: promptIds.count,
                                             maxNew: config.maxNewTokens,
                                             maxContext: context.maxContext)
    }

    let prefill = try await runRawPrefill(
        producer: producer,
        promptIds: promptIds,
        multimodalInput: multimodalInput,
        prefillConfig: prefillConfig,
        outputMode: fusedGreedy ? .greedyIfAvailable : .logits,
        isPureGreedy: config.isPureGreedy,
        historyReserve: promptIds.count + config.maxNewTokens,
        scratch: scratch,
        onProgress: onProgress)
    var history = prefill.history
    var position = prefill.position
    let prefillSeed = prefill.seed

    let decodeStart = Date()
    let prefillSeconds = prefill.prefillSeconds
    var stopMatcher = StreamingStopMatcher(stops: config.stopStrings)
    var generated = 0
    var reason: StopReason = .maxTokens
    var uncommittedBoundaryTokenIDs: [Int32] = []
    var trailingInvisibleTokens = 0

    while true {
        try Task.checkCancellation()

        let tokenID: Int32
        if generated == 0, let seed = prefillSeed {
            switch seed {
            case .greedyToken(let token):
                tokenID = Int32(bitPattern: token)
            case .logitsWritten:
                tokenID = try sampleOnce(scratch: scratch, context: context,
                                         history: history, config: config, position: generated)
            }
        } else if let fusedGreedyRunner {
            tokenID = Int32(bitPattern: fusedGreedyRunner.lastGreedyToken)
        } else {
            tokenID = try sampleOnce(scratch: scratch, context: context,
                                     history: history, config: config, position: generated)
        }
        generated += 1
        uncommittedBoundaryTokenIDs = [tokenID]

        if tokenizer.stopTokenIDs.contains(tokenID) || config.extraStopTokens.contains(tokenID) {
            if tokenID == tokenizer.endOfTurnID {
                reason = .endOfTurn
            } else if tokenID == tokenizer.toolResponseID {
                reason = .toolCalls
            } else {
                reason = .eos
            }
            let tail = stopMatcher.push(detok.flush()) + stopMatcher.finish()
            if !tail.isEmpty { onProgress(.tail(tail)) }
            break
        }

        let delta = detok.push(tokenID)
        let visible = stopMatcher.push(delta)
        onProgress(.token(index: generated - 1, id: tokenID, delta: visible))

        // Cancellation is not a stop-string match: reporting it as one made a
        // user pressing Stop indistinguishable from a configured stop string,
        // and `stopStringFiltered` is computed from `isStopped` rather than the
        // reason, so the two disagreed about the same run.
        let hitStopString = stopMatcher.isStopped
        let cancelled = !hitStopString && shouldStop()
        let hitMax = generated >= config.maxNewTokens
        if hitStopString || cancelled || hitMax {
            if !visible.isEmpty { trailingInvisibleTokens = 0 }
            let tail = stopMatcher.push(detok.flush()) + stopMatcher.finish()
            if !tail.isEmpty { onProgress(.tail(tail)) }
            reason = hitStopString ? .stopString : (cancelled ? .cancelled : .maxTokens)
            break
        }

        history.append(tokenID)
        trailingInvisibleTokens = visible.isEmpty ? trailingInvisibleTokens + 1 : 0
        try await producer.produce(token: tokenID, position: position, into: scratch.logits)
        position += 1
        uncommittedBoundaryTokenIDs.removeAll(keepingCapacity: true)
    }

    return RawDecodeResult(prefillTokens: promptIds.count,
                           prefillSeconds: prefillSeconds,
                           newTokens: generated,
                           decodeSeconds: Date().timeIntervalSince(decodeStart),
                           reason: reason,
                           kvPosition: position,
                           kvBackedTokenIDs: history,
                           uncommittedBoundaryTokenIDs: uncommittedBoundaryTokenIDs,
                           withheldTrailingKVTokens: reason == .stopString
                               ? trailingInvisibleTokens : 0)
}

private func sampleOnce(scratch: RawCompletionScratch, context: MetalContext,
                        history: [Int32], config: GenerationConfig, position: Int) throws -> Int32 {
    let cb = try context.makeCommandBuffer()
    scratch.sampler.sample(commandBuffer: cb, logits: scratch.logits, probs: scratch.probs,
                           history: history, config: config, position: position,
                           outToken: scratch.outToken)
    cb.commit(); cb.waitUntilCompleted()
    try checkCommandBufferError(cb)
    return unsafe Int32(bitPattern: scratch.outToken.contents().load(as: UInt32.self))
}
