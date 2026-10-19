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
    /// DFlash drafter directory; enables speculative greedy decoding.
    public var drafter: String?

    public init(model: String,
                backend: RuntimeBackend = .metal,
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
                maxContext: Int = 32_768,
                temperature: Float = 0.2,
                topK: Int? = 64,
                topP: Float? = 0.95,
                repetitionPenalty: Float = 1.0,
                seed: UInt64? = nil,
                stops: [String] = [],
                quiet: Bool = false,
                prefillPolicy: RuntimePrefillPolicy = RuntimeConfiguration.production.prefillPolicy,
                prefillChunkTokens: Int = RuntimeConfiguration.production.prefillChunkTokens,
                prefillChunkTokensAuto: Bool = true,
                faultPolicy: RuntimeFaultPolicy = RuntimeConfiguration.production.faultPolicy,
                earlyExpertRead: RuntimeEarlyExpertRead = RuntimeConfiguration.production.earlyExpertRead,
                drafter: String? = nil) {
        self.model = model
        self.drafter = drafter
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
      --backend <metal|hybrid-ane> Execution backend (default metal).
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
      --drafter <dir>            DFlash drafter checkpoint: speculative greedy
                                 decoding, with each step's verify length chosen
                                 from measured cost and acceptance. Requires
                                 --temperature 0 and a text prompt.
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
        imagePrompt: Bool = false,
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
        var backend = RuntimeBackend.metal
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
        var maxContext = 32_768
        var temperature: Float = 0.2
        var topK: Int? = 64
        var topP: Float? = 0.95
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

        var index = 0
        while index < argv.count {
            let flag = argv[index]
            switch flag {
            case "--help":
                throw ArgsError.helpRequested
            case "--backend":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = RuntimeBackend(rawValue: value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                backend = parsed
            case "--fault-policy":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = RuntimeFaultPolicy(rawValue: value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                faultPolicy = parsed
            case "--prompt-file":
                promptFile = try takeValue(argv, &index, flag: flag)
            case "--count-tokens":
                countTokens = true
                index += 1
            case "--expert-top-k", "--prefill-expert-cap":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value), (1...(flag == "--expert-top-k" ? 8 : 128)).contains(parsed) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                if flag == "--expert-top-k" { expertTopK = parsed } else { prefillExpertCap = parsed }
            case "--quiet":
                quiet = true
                index += 1
            case "--model":
                model = try takeValue(argv, &index, flag: flag)
            case "--prompt":
                prompt = try takeValue(argv, &index, flag: flag)
            case "--chat-prompt":
                chatPrompt = try takeValue(argv, &index, flag: flag)
            case "--image":
                images.append(try takeValue(argv, &index, flag: flag))
            case "--vision-pack":
                visionPack = try takeValue(argv, &index, flag: flag)
            case "--vision-residency":
                let value = try takeValue(argv, &index, flag: flag)
                guard let policy = VisionResidencyPolicy(rawValue: value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                visionResidency = policy
            case "--messages-file":
                messagesFile = try takeValue(argv, &index, flag: flag)
            case "--max-new":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value), parsed > 0 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                maxNew = parsed
            case "--max-context":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value), parsed > 0 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                maxContext = parsed
            case "--temperature":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Float(value), parsed >= 0 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                temperature = parsed
            case "--top-k":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value), (0...256).contains(parsed) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                topK = parsed == 0 ? nil : parsed
            case "--top-p":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Float(value), parsed > 0, parsed <= 1 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                topP = parsed
            case "--repetition-penalty":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Float(value), parsed > 0 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                repetitionPenalty = parsed
            case "--seed":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = UInt64(value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                seed = parsed
            case "--stop":
                stops.append(try takeValue(argv, &index, flag: flag))
            case "--early-expert-read":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = RuntimeEarlyExpertRead(rawValue: value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                earlyExpertRead = parsed
            case "--drafter":
                drafter = try takeValue(argv, &index, flag: flag)
            case "--prefill":
                let value = try takeValue(argv, &index, flag: flag)
                switch value {
                case "on": prefillPolicy = .chunked
                case "off": prefillPolicy = .off
                default: throw ArgsError.invalidValue(flag: flag, value: value)
                }
            case "--prefill-chunk-tokens", "--prefill-chunk":
                let value = try takeValue(argv, &index, flag: flag)
                if value == "auto" {
                    prefillChunkTokensAuto = true
                    break
                }
                guard let parsed = Int(value),
                      RuntimeConfiguration.allowedPrefillChunkTokens.contains(parsed) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                // A repeated flag is last-one-wins in both directions: without
                // this clear, an explicit size after `auto` parses and is then
                // overridden at run time by the resolved size.
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
            // Naming the flag the user actually left out. Reporting a conflict
            // with `--messages-file` sent them looking for a flag they never
            // passed, in the most likely first-run mistake there is.
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
        if drafter != nil, temperature != 0 || repetitionPenalty != 1 {
            throw ArgsError.invalidValue(
                flag: "--drafter", value: "speculative decoding is greedy: use --temperature 0")
        }
        if drafter != nil, !images.isEmpty {
            throw ArgsError.mutuallyExclusive("--drafter", "--image")
        }
        // Without prefill every prompt token goes through the speculative
        // decoder one at a time, which would draft and verify each of them.
        if drafter != nil, prefillPolicy == .off {
            throw ArgsError.mutuallyExclusive("--drafter", "--prefill off")
        }
        if temperature > 0, topK == nil, let topP, topP < 1 {
            throw ArgsError.invalidValue(
                flag: "--top-p",
                value: "\(topP) requires --top-k between 1 and 256")
        }
        let arguments = Args(model: model,
                             backend: backend,
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
                             drafter: drafter)
        _ = try arguments.resolvedRuntimeConfiguration(forceLogitsHead: false)
        return arguments
    }

    private static func takeValue(_ argv: [String],
                                  _ index: inout Int,
                                  flag: String) throws -> String {
        guard index + 1 < argv.count else { throw ArgsError.missingValue(flag: flag) }
        let value = argv[index + 1]
        index += 2
        return value
    }
}
