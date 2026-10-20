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
    /// Prompt tokens kept from the producer's cache (`reusingPrefix`); the
    /// rest of the prompt was prefilled.
    public let reusedPrefixTokens: Int
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

/// Raw-completion prefill + decode loop shared by the CLI and the server.
/// Consumes pre-encoded `promptIds` (no chat template).
///
/// A fused-head producer never writes logits: it serves pure-greedy configs
/// and, as a `SamplingTokenProducer`, any config without a repetition
/// penalty; other configs need a logits head.
public func runRawCompletion(producer: some LogitProducer,
                             tokenizer: GFTokenizer,
                             promptIds: [Int32],
                             multimodalInput: MultimodalPrefillInput? = nil,
                             config: GenerationConfig,
                             context: MetalContext,
                             scratch: RawCompletionScratch,
                             prefillConfig requestedPrefillConfig: PrefillRuntimeConfig = .defaultChunked,
                             reusingPrefix: Int = 0,
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
    // A sampling producer draws on the GPU itself, for any config that does
    // not need the logits on the host (a repetition penalty does).
    let gpuSampling = TokenSampling(config: config)
    let samplingProducer = fusedGreedy ? producer as? any SamplingTokenProducer : nil
    let sampledOnGPU = gpuSampling != nil && samplingProducer != nil
    guard !fusedGreedy || config.isPureGreedy || sampledOnGPU else {
        throw PrefillError.unsupportedPrefillSeed(
            "the fused-head producer cannot serve this sampling configuration; use a logits head")
    }
    if sampledOnGPU { samplingProducer?.tokenSampling = gpuSampling }
    defer { if sampledOnGPU { samplingProducer?.tokenSampling = nil } }

    var detok = GFDetokenizer(tokenizer: tokenizer,
                              barrierTokenIDs: tokenizer.structuralMarkerIDs)

    if let context = producer as? any ContextWindowReporting,
       promptIds.count + config.maxNewTokens > context.maxContext {
        throw GeneratorError.contextOverflow(prompt: promptIds.count,
                                             maxNew: config.maxNewTokens,
                                             maxContext: context.maxContext)
    }

    // Image spans are served only by the chunked prefill path, so an image
    // prompt runs chunked whatever prefill config was asked for.
    let prefillConfig = multimodalInput == nil
        ? requestedPrefillConfig
        : (requestedPrefillConfig.coercedForImagePrompt() ?? requestedPrefillConfig)
    if let multimodalInput {
        guard multimodalInput.effectiveTokenIDs == promptIds else {
            throw PrefillError.chunkedUnsupported(
                "multimodal effective token IDs do not match the prompt")
        }
    }
    let outputMode: PrefillOutputMode = fusedGreedy ? .greedyIfAvailable : .logits
    let producerPicksTokens = config.isPureGreedy || sampledOnGPU
    func checkSeed(_ seed: PrefillSeed, path: String) throws {
        if outputMode == .logits, seed != .logitsWritten {
            throw PrefillError.unsupportedPrefillSeed(
                "RawCompletion \(path) prefill requested logits but producer returned \(seed)")
        }
        if case .greedyToken = seed, !producerPicksTokens {
            throw PrefillError.unsupportedPrefillSeed(
                "RawCompletion \(path) prefill returned a greedy token for a sampling config")
        }
    }

    var history: [Int32] = []
    history.reserveCapacity(promptIds.count + config.maxNewTokens)

    // A prefix still in the producer's cache is kept; at least the prompt's
    // last token is prefilled, for the seed.
    let reused = multimodalInput == nil && reusingPrefix > 0 && reusingPrefix < promptIds.count
        && (producer as? any PrefixReusingProducer)?.canReuse(prefix: reusingPrefix) == true
        ? reusingPrefix : 0
    if reused > 0, let reusing = producer as? any PrefixReusingProducer {
        try reusing.rewindForReuse(to: reused)
        history.append(contentsOf: promptIds[..<reused])
    } else {
        producer.reset()
    }
    let prefillStart = Date()
    var position = reused
    var prefillSeed: PrefillSeed?
    let prefillTokens = promptIds[reused...]
    switch (multimodalInput, prefillConfig.mode) {
    case (.some(let input), .chunked):
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
        try checkSeed(result.seed, path: "multimodal")
        position = result.newPosition
        prefillSeed = result.seed
        history.append(contentsOf: prefillTokens)
    case (.some, _):
        throw PrefillError.chunkedUnsupported(
            "multimodal prefill requires a MultimodalPrefillRunner-backed runtime")
    case (.none, .chunked):
        guard let chunked = producer as? any ChunkedPrefillRunner else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill requires a ChunkedPrefillRunner-backed runtime")
        }
        let result = try await chunked.prefillChunked(tokens: prefillTokens,
                                                      startPosition: position,
                                                      outputMode: outputMode,
                                                      config: prefillConfig,
                                                      into: scratch.logits) { done in
            onProgress(.prefill(done: reused + done, total: promptIds.count))
        }
        try checkSeed(result.seed, path: "chunked")
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
    let prefillSeconds = Date().timeIntervalSince(prefillStart)

    let decodeStart = Date()
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

        // Cancellation is reported as `.cancelled`, never as a stop string.
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
                           withheldTrailingKVTokens: reason == .stopString ? trailingInvisibleTokens : 0,
                           reusedPrefixTokens: reused)
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
