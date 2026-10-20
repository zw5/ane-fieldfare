public import Foundation
public import ANEFieldfareHotpath
import ANEFieldfareHotpathCLICore

/// The loaded Gemma model: one runner, its sampling scratch, and the optional
/// DFlash drafter, generating through the same `runRawCompletion` loop and
/// `selectLogitProducer` decision as the CLI.
///
/// The runner is built with the fused head, so greedy requests take the
/// hot-vocabulary head and sampled requests sample on the GPU, both through
/// the drafter when one is loaded, exactly as the CLI does; a request with a
/// repetition penalty goes through `LogitsHeadProducer`, the logits head a
/// CLI run with a penalty takes.
public final class ModelEngine: CompletionEngine {
    public struct Options: Sendable {
        public var modelURL: URL
        public var drafterURL: URL?
        public var maxContext: Int
        public var backend: RuntimeBackend
        public var faultPolicy: RuntimeFaultPolicy
        public var earlyExpertRead: RuntimeEarlyExpertRead

        public init(modelURL: URL, drafterURL: URL?, maxContext: Int, backend: RuntimeBackend,
                    faultPolicy: RuntimeFaultPolicy, earlyExpertRead: RuntimeEarlyExpertRead) {
            self.modelURL = modelURL
            self.drafterURL = drafterURL
            self.maxContext = maxContext
            self.backend = backend
            self.faultPolicy = faultPolicy
            self.earlyExpertRead = earlyExpertRead
        }
    }

    public let tokenizer: GFTokenizer
    public let maxContext: Int
    /// The largest prefill chunk a request may use; the KV ring is sized for it.
    public let maxPrefillChunkTokens: Int
    private let context: MetalContext
    private let runner: RealForwardRunner
    private let scratch: RawCompletionScratch
    private let drafter: DFlashDrafter?
    /// The tokens in the runner's KV cache after the last request that
    /// completed, and the decode path that wrote them; nil after a failed or
    /// cancelled request, whose cache state is unknown. Requests run one at a
    /// time, so this is never read and written concurrently.
    private var cached: (tokens: [Int32], path: String)?

    public var hasDrafter: Bool { drafter != nil }

    public init(options: Options, tokenizer: GFTokenizer) async throws {
        // Every request's chunk is chosen from its prompt length as the CLI's
        // `--prefill-chunk-tokens auto` chooses it; the runner's KV ring has to
        // hold the largest one the context allows.
        let maxChunk = PrefillRuntimeConfig.autoChunkTokens(promptTokens: options.maxContext)
        // The CLI's own validation and environment handling, with the CLI's
        // greedy head: `forceLogitsHead` false.
        let runtime = try Args.runtimeConfiguration(
            backend: options.backend, prefillChunkTokens: maxChunk, faultPolicy: options.faultPolicy,
            earlyExpertRead: options.earlyExpertRead, forceLogitsHead: false,
            environment: ProcessInfo.processInfo.environment)
        let context = try MetalContext()
        let model = try Model.load(directoryURL: options.modelURL, device: context.device)
        let runner = try RealForwardRunner(model: model, context: context, maxContext: options.maxContext,
                                           runtimeConfiguration: runtime)
        self.scratch = try RawCompletionScratch(context: context, vocab: model.config.vocabSize)
        if let drafterURL = options.drafterURL {
            self.drafter = try await DFlashDrafter(directory: drafterURL, model: model, context: context,
                                             maxContext: options.maxContext)
        } else {
            self.drafter = nil
        }
        self.tokenizer = tokenizer
        self.maxContext = options.maxContext
        self.maxPrefillChunkTokens = maxChunk
        self.context = context
        self.runner = runner
    }

    public func complete(promptIDs: [Int32], config: GenerationConfig,
                         onProgress: (RawDecodeProgress) -> Void) async throws -> EngineCompletion {
        let prefill = PrefillRuntimeConfig.production(
            chunkTokens: PrefillRuntimeConfig.autoChunkTokens(promptTokens: promptIDs.count,
                                                              cap: maxPrefillChunkTokens))
        // A decoder per request: its policy starts fresh, as a CLI run's does,
        // and releasing it detaches the drafter's feature tap, which requests
        // without it must not pay for. Greedy and sampled requests both use
        // it; a repetition penalty needs host logits and cannot.
        let speculative: SpeculativeDecoder?
        if config.repetitionPenalty == 1, let drafter {
            speculative = try SpeculativeDecoder(runner: runner, drafter: drafter)
        } else {
            speculative = nil
        }
        let producer = selectLogitProducer(runner: runner, speculative: speculative, config: config,
                                           prefillConfig: prefill, hasImages: false)
        let path: String
        switch producer {
        case is SpeculativeDecoder: path = "speculative"
        case is HotHeadDecoder: path = "hot-head"
        case is LogitsHeadProducer: path = "sampled"
        default: path = config.isPureGreedy ? "plain" : "sampled"
        }
        // A chat re-sends its whole history: keep the prefix still cached and
        // prefill only what follows. Only on the path that wrote the cache —
        // a speculative request also needs the drafter's context rows, which
        // other paths do not write.
        var reuse = 0
        if let cached, cached.path == path {
            // The prompt's last token is always prefilled, for its seed.
            reuse = min(zip(cached.tokens, promptIDs).prefix { $0.0 == $0.1 }.count, promptIDs.count - 1)
        }
        cached = nil
        let result = try await runRawCompletion(
            producer: producer,
            tokenizer: tokenizer,
            promptIds: promptIDs,
            config: config,
            context: context,
            scratch: scratch,
            prefillConfig: prefill,
            reusingPrefix: reuse,
            onProgress: onProgress)
        if result.kvBackedTokenIDs.count == result.kvPosition {
            cached = (result.kvBackedTokenIDs, path)
        }
        return EngineCompletion(result: result, decodePath: path)
    }
}
