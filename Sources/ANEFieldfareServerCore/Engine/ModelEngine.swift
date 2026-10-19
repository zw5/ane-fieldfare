public import Foundation
public import ANEFieldfareHotpath
import ANEFieldfareHotpathCLICore

/// The loaded Gemma model: one runner, its sampling scratch, and the optional
/// DFlash drafter, generating through the same `runRawCompletion` loop and
/// `selectLogitProducer` decision as the CLI.
///
/// The runner is built with the fused greedy head, so greedy requests take
/// the hot-vocabulary head (or the drafter) exactly as the CLI's
/// `--temperature 0` does, and sampled requests go through
/// `LogitsHeadProducer`, which takes the logits head a CLI sampling run's
/// `forceLogitsHead` runner takes. Every request starts from a reset KV
/// cache: `runRawCompletion` resets the producer before its prefill, and no
/// prefix is reused across requests.
public final class ModelEngine: CompletionEngine {
    public struct Options: Sendable {
        public var modelURL: URL
        public var drafterURL: URL?
        public var maxContext: Int
        public var backend: RuntimeBackend
        public var faultPolicy: RuntimeFaultPolicy
        public var earlyExpertRead: RuntimeEarlyExpertRead

        public init(modelURL: URL, drafterURL: URL? = nil, maxContext: Int,
                    backend: RuntimeBackend = .metal,
                    faultPolicy: RuntimeFaultPolicy = RuntimeConfiguration.production.faultPolicy,
                    earlyExpertRead: RuntimeEarlyExpertRead = RuntimeConfiguration.production.earlyExpertRead) {
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
    public let runtime: RuntimeConfiguration
    private let context: MetalContext
    private let runner: RealForwardRunner
    private let scratch: RawCompletionScratch
    private let drafter: DFlashDrafter?

    public var hasDrafter: Bool { drafter != nil }

    public init(options: Options, tokenizer: GFTokenizer,
                environment: [String: String] = ProcessInfo.processInfo.environment) async throws {
        // Every request's chunk is chosen from its prompt length as the CLI's
        // `--prefill-chunk-tokens auto` chooses it; the runner's KV ring has to
        // hold the largest one the context allows.
        let maxChunk = PrefillRuntimeConfig.autoChunkTokens(promptTokens: options.maxContext)
        // The CLI's own validation and environment handling, with the CLI's
        // greedy head: `forceLogitsHead` false.
        let args = Args(model: options.modelURL.path,
                        backend: options.backend,
                        maxContext: options.maxContext,
                        temperature: 0,
                        prefillChunkTokens: maxChunk,
                        prefillChunkTokensAuto: false,
                        faultPolicy: options.faultPolicy,
                        earlyExpertRead: options.earlyExpertRead,
                        drafter: options.drafterURL?.path)
        let runtime = try args.resolvedRuntimeConfiguration(forceLogitsHead: false, environment: environment)
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
        self.runtime = runtime
        self.context = context
        self.runner = runner
    }

    public func complete(promptIDs: [Int32], config: GenerationConfig,
                         onProgress: (RawDecodeProgress) -> Void) async throws -> EngineCompletion {
        let prefill = PrefillRuntimeConfig.production(
            chunkTokens: PrefillRuntimeConfig.autoChunkTokens(promptTokens: promptIDs.count,
                                                              cap: maxPrefillChunkTokens))
        // A decoder per request: its policy starts fresh, as a CLI run's does,
        // and releasing it detaches the drafter's feature tap, which plain and
        // sampled requests must not pay for.
        let speculative: SpeculativeDecoder?
        if config.isPureGreedy, let drafter {
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
        let result = try await runRawCompletion(
            producer: producer,
            tokenizer: tokenizer,
            promptIds: promptIDs,
            config: config,
            context: context,
            scratch: scratch,
            prefillConfig: prefill,
            onProgress: onProgress)
        return EngineCompletion(result: result, decodePath: path)
    }
}
