import Foundation
public import ANEFieldfareHotpath

public struct Args: Equatable, Sendable {
    public var model: String
    public var backend: RuntimeBackend
    public var promptFile: String?
    public var countTokens: Bool
    public var expertTopK: Int
    public var prefillExpertCap: Int?
    public var prompt: String?
    public var chatPrompt: String?
    public var messagesFile: String?
    public var images: [String]
    public var visionPack: String?
    public var visionResidency: VisionResidencyPolicy
    public var maxNew: Int
    public var maxContext: Int
    public var temperature: Float
    public var topK: Int?
    public var topP: Float?
    public var repetitionPenalty: Float
    public var seed: UInt64?
    public var stops: [String]
    public var quiet: Bool
    public var prefillPolicy: RuntimePrefillPolicy
    public var prefillChunkTokens: Int
    /// `--prefill-chunk-tokens auto`: the size is decided once the prompt length
    /// is known, which needs the tokenizer and, for images, their geometry.
    public var prefillChunkTokensAuto: Bool
    public var faultPolicy: RuntimeFaultPolicy
    public var earlyExpertRead: RuntimeEarlyExpertRead
    /// DFlash drafter directory; enables speculative decoding.
    public var drafter: String?
    public var drafterOrigin: DrafterOrigin

    /// How `drafter` was chosen. Speculation is the default: without
    /// `--drafter` the drafter is looked for beside the model.
    public enum DrafterOrigin: Equatable, Sendable {
        /// Named with `--drafter <dir>`; conflicting options are errors.
        case named
        /// Found by `DrafterLocation`; a request it cannot serve decodes plainly.
        case found
        /// Looked for and not there.
        case missing
        /// `--drafter off`, or a request speculation cannot serve.
        case off
    }

    public init(model: String,
                backend: RuntimeBackend = .hybridANE,
                promptFile: String? = nil,
                countTokens: Bool = false,
                expertTopK: Int = 8,
                prefillExpertCap: Int? = nil,
                prompt: String? = nil,
                chatPrompt: String? = nil,
                messagesFile: String? = nil,
                images: [String] = [],
                visionPack: String? = nil,
                visionResidency: VisionResidencyPolicy = .defaultPolicy,
                maxNew: Int = 1_024,
                maxContext: Int = OptionDefaults.maxContext,
                temperature: Float = OptionDefaults.temperature,
                topK: Int? = OptionDefaults.topK,
                topP: Float? = OptionDefaults.topP,
                repetitionPenalty: Float = 1.0,
                seed: UInt64? = nil,
                stops: [String] = [],
                quiet: Bool = false,
                prefillPolicy: RuntimePrefillPolicy = RuntimeConfiguration.production.prefillPolicy,
                prefillChunkTokens: Int = RuntimeConfiguration.production.prefillChunkTokens,
                prefillChunkTokensAuto: Bool = true,
                faultPolicy: RuntimeFaultPolicy = RuntimeConfiguration.production.faultPolicy,
                earlyExpertRead: RuntimeEarlyExpertRead = RuntimeConfiguration.production.earlyExpertRead,
                drafter: String? = nil,
                drafterOrigin: DrafterOrigin? = nil) {
        self.model = model
        self.drafter = drafter
        self.drafterOrigin = drafterOrigin ?? (drafter == nil ? .off : .named)
        self.earlyExpertRead = earlyExpertRead
        self.backend = backend
        self.faultPolicy = faultPolicy
        self.promptFile = promptFile
        self.countTokens = countTokens
        self.expertTopK = expertTopK
        self.prefillExpertCap = prefillExpertCap
        self.prompt = prompt
        self.chatPrompt = chatPrompt
        self.messagesFile = messagesFile
        self.images = images
        self.visionPack = visionPack
        self.visionResidency = visionResidency
        self.maxNew = maxNew
        self.maxContext = maxContext
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
        self.stops = stops
        self.quiet = quiet
        self.prefillPolicy = prefillPolicy
        self.prefillChunkTokens = prefillChunkTokens
        self.prefillChunkTokensAuto = prefillChunkTokensAuto
    }
}

public enum ArgsError: Error, Equatable, CustomStringConvertible {
    case helpRequested
    case unknownFlag(String)
    case missingValue(flag: String)
    case invalidValue(flag: String, value: String)
    case requiredMissing(String)
    case mutuallyExclusive(String, String)
    case modeMissing

    public var description: String {
        switch self {
        case .helpRequested: return "help requested"
        case .unknownFlag(let flag): return "unknown flag: \(flag)"
        case .missingValue(let flag): return "missing value for \(flag)"
        case .invalidValue(let flag, let value): return "invalid value for \(flag): \(value)"
        case .requiredMissing(let flag): return "required flag missing: \(flag)"
        case .mutuallyExclusive(let a, let b): return "\(a) and \(b) are mutually exclusive"
        case .modeMissing:
            return "one of --prompt, --prompt-file, --chat-prompt or --messages-file is required"
        }
    }
}

extension Args {
    /// Whether `--prefill-chunk-tokens auto` has a size to resolve at all.
    ///
    /// Under `--prefill off` the config is `.off` and the size is inert, and an
    /// image turn there is coerced to the runtime's own chunked default rather
    /// than to a resolved size, so resolving one would only announce a number
    /// nothing goes on to use.
    public var resolvesPrefillChunkAuto: Bool {
        prefillChunkTokensAuto && prefillPolicy != .off
    }

    public static let usage = """
    ANEFieldfareHotpathCLI — Gemma 4 26B-A4B text generation

    usage: ANEFieldfareHotpathCLI --model <dir>
           (--prompt <string> | --prompt-file <path> | --chat-prompt <string> | --messages-file <path>) [options]

    required:
      --model <dir>             Path to a .gturbo model directory.
      --prompt <string>         Raw-completion prompt.
      --chat-prompt <string>    Single-turn instruction chat; pairs with --image
      --messages-file <path>    JSON chat messages with role and content fields.
      --image <path>            Attach an image; repeatable. Needs --chat-prompt
                                and an installed companion pack.
      --vision-pack <dir>       Companion pack (default beside the model).
      --vision-residency <on-demand|keep-ready>
                                Routed-expert residency during vision (default on-demand).

    options:
      --backend <metal|hybrid-ane> Execution backend (default hybrid-ane; metal when
                                 --expert-top-k, --prefill-expert-cap or
                                 --prefill off rule the ANE out).
      --fault-policy <fail-closed|failover-metal>
                                 On an ANE guest fault, abort (default) or
                                 evict the guest and continue on Metal.
      --prompt-file <path>       Read a raw-completion prompt from UTF-8 text.
      --count-tokens             Print text prompt token count and exit.
      --expert-top-k <n>         Routed experts per token, 1...8 (default 8).
      --prefill-expert-cap <n>   Optional lossy expert-union cap, 1...128.
      --max-new <int>            Generated-token limit (default 1024).
      --max-context <int>        Context limit in tokens (default 32768).
      --temperature <float>      Sampling temperature (default 0.2; 0 = greedy).
      --top-k <int>              Top-k truncation, 1...256 (default 64; 0 = off).
      --top-p <float>            Nucleus truncation (default 0.95).
      --repetition-penalty <f>   Repetition penalty (default 1.0).
      --seed <uint64>            Deterministic sampling seed (default off).
      --stop <string>            Stop substring (repeatable).
      --quiet                    Suppress the timing footer.
      --early-expert-read <s>    Decode early read of the next layer's likeliest
                                 experts in idle storage time: off, router or
                                 fitted (default fitted).
      --drafter <dir|off>        DFlash drafter checkpoint for speculative
                                 decoding, greedy or sampled, with each step's
                                 verify length chosen from measured cost and
                                 acceptance. Default: found beside the model or
                                 in ../dflash/\(DrafterLocation.directoryName).
                                 Image prompts and a repetition penalty decode
                                 without it.
      --prefill on|off           Enable or disable chunked prompt prefill (default on).
      --prefill-chunk-tokens <n|auto>
                                 Prefill chunk size: \(RuntimeConfiguration.allowedValueList(RuntimeConfiguration.allowedPrefillChunkTokens, alsoAccepting: ["auto"]))
                                 (default auto). Each chunk re-reads the routed
                                 expert pool, so larger chunks read less; auto
                                 picks the smallest size that covers the prompt.
      --help                     Show this message.
    """

    public func resolvedRuntimeConfiguration(
        forceLogitsHead: Bool,
        environment: [String: String] = [:]) throws -> RuntimeConfiguration {
        try Self.runtimeConfiguration(
            backend: backend, expertTopK: expertTopK, prefillExpertCap: prefillExpertCap,
            prefillPolicy: prefillPolicy, prefillChunkTokens: prefillChunkTokens, faultPolicy: faultPolicy,
            earlyExpertRead: earlyExpertRead, forceLogitsHead: forceLogitsHead, environment: environment)
    }

    /// The runtime configuration these options select, validated as the
    /// CLI's flags are. The server builds its runner through this too.
    public static func runtimeConfiguration(
        backend: RuntimeBackend,
        expertTopK: Int = 8,
        prefillExpertCap: Int? = nil,
        prefillPolicy: RuntimePrefillPolicy = RuntimeConfiguration.production.prefillPolicy,
        prefillChunkTokens: Int,
        faultPolicy: RuntimeFaultPolicy,
        earlyExpertRead: RuntimeEarlyExpertRead,
        forceLogitsHead: Bool,
        environment: [String: String] = [:]) throws -> RuntimeConfiguration {
        let activeExperts = expertTopK
        let expertCap = prefillExpertCap ?? (activeExperts < 8 ? activeExperts * 8 : nil)
        guard (1...8).contains(activeExperts), expertCap.map({ (1...128).contains($0) }) ?? true else {
            throw ArgsError.invalidValue(flag: "--expert-top-k/--prefill-expert-cap", value: "out of range")
        }
        let attentionName = environment["ANE_FIELDFARE_PREFILL_ATTENTION"] ?? "causal-simd"
        guard let attention = RuntimePrefillAttentionPath(rawValue: attentionName) else {
            throw ArgsError.invalidValue(flag: "ANE_FIELDFARE_PREFILL_ATTENTION", value: attentionName)
        }
        guard backend != .hybridANE || (activeExperts == 8 && expertCap == nil) else {
            throw ArgsError.invalidValue(flag: "--backend", value: "hybrid-ane requires all eight routed experts and no prefill expert cap")
        }
        guard RuntimeConfiguration.allowedPrefillChunkTokens.contains(prefillChunkTokens) else {
            throw ArgsError.invalidValue(
                flag: "--prefill-chunk-tokens", value: "\(prefillChunkTokens)")
        }
        return RuntimeConfiguration(
            backend: backend,
            prefillEnabled: prefillPolicy == .chunked,
            prefillChunkTokens: prefillChunkTokens,
            prefillAttentionPath: attention,
            forceLogitsHead: forceLogitsHead,
            activeRoutedExperts: activeExperts,
            prefillExpertUnionCap: expertCap,
            faultPolicy: faultPolicy,
            earlyExpertRead: earlyExpertRead)
    }

    public static func parse(_ argv: [String]) throws -> Args {
        var backend: RuntimeBackend?
        var faultPolicy = RuntimeConfiguration.production.faultPolicy
        var promptFile: String?
        var countTokens = false
        var expertTopK = 8
        var prefillExpertCap: Int?
        var model: String?
        var prompt: String?
        var chatPrompt: String?
        var messagesFile: String?
        var images: [String] = []
        var visionPack: String?
        var visionResidency: VisionResidencyPolicy = .defaultPolicy
        var maxNew = 1_024
        var maxContext = OptionDefaults.maxContext
        var temperature = OptionDefaults.temperature
        var topK = OptionDefaults.topK
        var topP = OptionDefaults.topP
        var repetitionPenalty: Float = 1.0
        var seed: UInt64?
        var stops: [String] = []
        var quiet = false
        let runtimeDefaults = RuntimeConfiguration.production
        var prefillPolicy = runtimeDefaults.prefillPolicy
        var prefillChunkTokens = runtimeDefaults.prefillChunkTokens
        var prefillChunkTokensAuto = true
        var earlyExpertRead = runtimeDefaults.earlyExpertRead
        var drafter: String?

        var cursor = ArgumentCursor(argv)
        while let flag = cursor.nextFlag() {
            switch flag {
            case "--help", "-h":
                throw ArgsError.helpRequested
            case "--backend":
                backend = try cursor.choice(for: flag, as: RuntimeBackend.self)
            case "--fault-policy":
                faultPolicy = try cursor.choice(for: flag, as: RuntimeFaultPolicy.self)
            case "--prompt-file":
                promptFile = try cursor.value(for: flag)
            case "--count-tokens":
                countTokens = true
            case "--expert-top-k":
                expertTopK = try cursor.value(for: flag, as: Int.self) { (1...8).contains($0) }
            case "--prefill-expert-cap":
                prefillExpertCap = try cursor.value(for: flag, as: Int.self) { (1...128).contains($0) }
            case "--quiet":
                quiet = true
            case "--model":
                model = try cursor.value(for: flag)
            case "--prompt":
                prompt = try cursor.value(for: flag)
            case "--chat-prompt":
                chatPrompt = try cursor.value(for: flag)
            case "--image":
                images.append(try cursor.value(for: flag))
            case "--vision-pack":
                visionPack = try cursor.value(for: flag)
            case "--vision-residency":
                visionResidency = try cursor.choice(for: flag, as: VisionResidencyPolicy.self)
            case "--messages-file":
                messagesFile = try cursor.value(for: flag)
            case "--max-new":
                maxNew = try cursor.value(for: flag, as: Int.self) { $0 > 0 }
            case "--max-context":
                maxContext = try cursor.maxContext(for: flag)
            case "--temperature":
                temperature = try cursor.temperature(for: flag)
            case "--top-k":
                topK = try cursor.topK(for: flag)
            case "--top-p":
                topP = try cursor.topP(for: flag)
            case "--repetition-penalty":
                repetitionPenalty = try cursor.value(for: flag, as: Float.self) { $0.isFinite && $0 > 0 }
            case "--seed":
                seed = try cursor.value(for: flag, as: UInt64.self)
            case "--stop":
                stops.append(try cursor.value(for: flag))
            case "--early-expert-read":
                earlyExpertRead = try cursor.choice(for: flag, as: RuntimeEarlyExpertRead.self)
            case "--drafter":
                drafter = try cursor.value(for: flag)
            case "--prefill":
                let value = try cursor.value(for: flag)
                switch value {
                case "on": prefillPolicy = .chunked
                case "off": prefillPolicy = .off
                default: throw ArgsError.invalidValue(flag: flag, value: value)
                }
            case "--prefill-chunk-tokens":
                let value = try cursor.value(for: flag)
                if value == "auto" {
                    prefillChunkTokensAuto = true
                    break
                }
                guard let parsed = Int(value),
                      RuntimeConfiguration.allowedPrefillChunkTokens.contains(parsed) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                // Repeated flags are last-one-wins, `auto` included.
                prefillChunkTokensAuto = false
                prefillChunkTokens = parsed
            default:
                throw ArgsError.unknownFlag(flag)
            }
        }

        guard let model else { throw ArgsError.requiredMissing("--model") }
        if prompt != nil && messagesFile != nil {
            throw ArgsError.mutuallyExclusive("--prompt", "--messages-file")
        }
        if prompt != nil, chatPrompt != nil {
            throw ArgsError.mutuallyExclusive("--prompt", "--chat-prompt")
        }
        if messagesFile != nil, chatPrompt != nil {
            throw ArgsError.mutuallyExclusive("--chat-prompt", "--messages-file")
        }
        if prompt != nil, !images.isEmpty {
            throw ArgsError.mutuallyExclusive("--prompt", "--image")
        }
        if !images.isEmpty, chatPrompt == nil {
            // Name the flag the user left out, unless they passed the one
            // images conflict with.
            throw messagesFile == nil
                ? ArgsError.requiredMissing("--chat-prompt")
                : ArgsError.mutuallyExclusive("--image", "--messages-file")
        }
        if prompt == nil && promptFile == nil && chatPrompt == nil && messagesFile == nil {
            throw ArgsError.modeMissing
        }
        if promptFile != nil {
            for (flag, present) in [("--prompt", prompt != nil), ("--chat-prompt", chatPrompt != nil),
                                    ("--messages-file", messagesFile != nil), ("--image", !images.isEmpty)] where present {
                throw ArgsError.mutuallyExclusive("--prompt-file", flag)
            }
        }
        // Without prefill every prompt token would go through the speculative
        // decoder one at a time, drafting and verifying each of them.
        let speculationFits = repetitionPenalty == 1 && images.isEmpty && prefillPolicy != .off
        let drafterOrigin: Args.DrafterOrigin
        switch drafter {
        case "off"?:
            drafter = nil
            drafterOrigin = .off
        case .some:
            if repetitionPenalty != 1 {
                throw ArgsError.invalidValue(
                    flag: "--drafter", value: "speculative decoding cannot apply a repetition penalty")
            }
            if !images.isEmpty {
                throw ArgsError.mutuallyExclusive("--drafter", "--image")
            }
            if prefillPolicy == .off {
                throw ArgsError.mutuallyExclusive("--drafter", "--prefill off")
            }
            drafterOrigin = .named
        case nil where !speculationFits:
            drafterOrigin = .off
        case nil:
            drafter = DrafterLocation.find(model: model)?.path
            drafterOrigin = drafter == nil ? .missing : .found
        }
        try validateTruncation(temperature: temperature, topK: topK, topP: topP)
        // The ANE rail serves prefill with all eight routed experts, uncapped.
        let aneFits = expertTopK == 8 && prefillExpertCap == nil && prefillPolicy != .off
        let arguments = Args(model: model,
                             backend: backend ?? (aneFits ? .hybridANE : .metal),
                             promptFile: promptFile,
                             countTokens: countTokens,
                             expertTopK: expertTopK,
                             prefillExpertCap: prefillExpertCap,
                             prompt: prompt,
                             chatPrompt: chatPrompt,
                             messagesFile: messagesFile,
                             images: images,
                             visionPack: visionPack,
                             visionResidency: visionResidency,
                             maxNew: maxNew,
                             maxContext: maxContext,
                             temperature: temperature,
                             topK: topK,
                             topP: topP,
                             repetitionPenalty: repetitionPenalty,
                             seed: seed,
                             stops: stops,
                             quiet: quiet,
                             prefillPolicy: prefillPolicy,
                             prefillChunkTokens: prefillChunkTokens,
                             prefillChunkTokensAuto: prefillChunkTokensAuto,
                             faultPolicy: faultPolicy,
                             earlyExpertRead: earlyExpertRead,
                             drafter: drafter,
                             drafterOrigin: drafterOrigin)
        _ = try arguments.resolvedRuntimeConfiguration(forceLogitsHead: false)
        return arguments
    }
}
