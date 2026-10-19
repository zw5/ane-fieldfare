public import Metal

/// Host-side sampling on a runner built with the fused head.
///
/// A fused head picks its token on the device, greedy or under a
/// `TokenSampling` rule, and never writes logits; only a configuration the
/// GPU cannot sample (a repetition penalty reads the history) needs the
/// logits on the host. A process that serves every configuration from one
/// runner, as the server does, wraps the runner in this for those requests:
/// every prefill and decode step runs with
/// `PrefillOutputMode.logits`, which takes the same logits head a
/// `forceLogitsHead` runner takes. The runner's fused-head state (its hot
/// vocabulary) is not read or changed.
///
/// Deliberately not a `FusedGreedyProducer`, so `runRawCompletion` samples
/// from the logits buffer.
public final class LogitsHeadProducer: LogitProducer, ChunkedPrefillRunner, MultimodalPrefillRunner,
    ContextWindowReporting {
    public let runner: RealForwardRunner
    public var maxContext: Int { runner.maxContext }

    public init(runner: RealForwardRunner) {
        self.runner = runner
    }

    public func reset() {
        runner.reset()
    }

    public func produce(token: Int32, position: Int, into logits: any MTLBuffer) async throws {
        try await runner.produceLogits(token: token, position: position, into: logits)
    }

    func prefillChunked(tokens: ArraySlice<Int32>, startPosition: Int, outputMode: PrefillOutputMode,
                        config: PrefillRuntimeConfig, into logits: any MTLBuffer,
                        onProgress: (Int) -> Void) async throws -> PrefillResult {
        try await runner.prefillChunked(tokens: tokens, startPosition: startPosition, outputMode: .logits,
                                        config: config, into: logits, onProgress: onProgress)
    }

    func prefillMultimodal(input: MultimodalPrefillInput, startPosition: Int, outputMode: PrefillOutputMode,
                           config: PrefillRuntimeConfig, into logits: any MTLBuffer,
                           onProgress: (Int) -> Void) async throws -> PrefillResult {
        try await runner.prefillMultimodal(input: input, startPosition: startPosition, outputMode: .logits,
                                           config: config, into: logits, onProgress: onProgress)
    }
}

/// The producer a generation decodes with, decided the same way for the CLI
/// and the server.
///
/// - A speculative decoder, when the caller built one (text prompts, greedy
///   or sampled without a repetition penalty).
/// - Plain greedy text decoding with chunked prefill guesses from the hot
///   vocabulary and checks every token against the full head; the output is
///   the full head's.
/// - A configuration with a repetition penalty on a fused-head runner
///   samples on the host through `LogitsHeadProducer`.
/// - Otherwise the runner itself, which samples on the GPU.
public func selectLogitProducer(runner: RealForwardRunner,
                                speculative: SpeculativeDecoder?,
                                config: GenerationConfig,
                                prefillConfig: PrefillRuntimeConfig,
                                hasImages: Bool) -> any LogitProducer {
    if let speculative {
        return speculative
    }
    if config.repetitionPenalty != 1 {
        return runner.usesFusedGreedyHead ? LogitsHeadProducer(runner: runner) : runner
    }
    if !config.isPureGreedy {
        return runner
    }
    if !hasImages, prefillConfig.mode == .chunked, let hotHead = HotHeadDecoder(runner: runner) {
        return hotHead
    }
    return runner
}
