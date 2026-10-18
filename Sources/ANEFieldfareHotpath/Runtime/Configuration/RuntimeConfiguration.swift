public enum RuntimeHeadPath: String, Codable, Sendable {
    case fusedRows = "fused-rows"
    case logits
}

public enum RuntimeBackend: String, Codable, Sendable {
    case metal
    /// Routed top-8 prefill on ANE; attention, shared FFN, logits and decode
    /// remain on the measured Metal companion rail.
    case hybridANE = "hybrid-ane"
}

public enum RuntimePrefillPolicy: String, Codable, Sendable {
    case off
    case chunked
}

public enum RuntimePrefillAttentionPath: String, Codable, Sendable {
    case causalTiled = "causal-tiled"
    case causalSIMD = "causal-simd"
    case fullTensorOps2DPreferred = "full-tensorops-2d-preferred"
    case fullTensorOps2DValidityV2 = "full-tensorops-2d-validity-v2"
}

public enum RuntimeExpertCachePolicy: String, Codable, Sendable {
    case lfu
    case lru
}

public enum RuntimeWidePrefillTransport: String, Codable, Sendable {
    /// Use the same bounded LFU carriers as recurrent decode.
    case cachedPread = "cached-pread"
    /// Route-authoritative, exact compact expert reads through two transient
    /// eight-expert banks. Retained only as a rejected experimental rail.
    case preadDoubleBuffered = "pread-double-buffered"
    /// Historical mapping control; rejected for its large working set.
    case mappedControl = "mapped-control"
}

public struct RuntimeConfiguration: Sendable, Equatable {
    public static let defaultExpertCacheSlots = 16
    public static let defaultPrefillChunkTokens = 1024
    // Cache capacity is separate from prefill scratch and ANE carriers. Check
    // the complete process footprint on the intended prompt and context size.
    public static let allowedExpertCacheSlots = [8, 16, 17, 18, 24, 32]
    public static let allowedPrefillChunkTokens = PrefillRuntimeConfig.allowedChunkTokens
    public static let minimumExpertCacheSlotsForChunkedPrefill = 16
    /// The one rendering shared by every help text and every rejection that
    /// names one of the arrays above, so neither can name a value the guard
    /// does not accept: the hardcoded "32, 64, or 128" outlived the widening of
    /// the allowed set and told users 256 was illegal while the guard accepted
    /// it. It lives here, beside the arrays, because the CLI's usage and the
    /// server's usage and rejections all render the same sets.
    ///
    /// `alsoAccepting` appends parse-level aliases after the integers, so
    /// `--prefill-chunk-tokens` renders "32, 64, 128, 256, or auto". They come
    /// last because they are not members of the array the guards test, and
    /// because the help line's integers are read back by
    /// `usageNamesExactlyTheAllowedValues`.
    public static func allowedValueList(_ values: [Int],
                                        alsoAccepting aliases: [String] = []) -> String {
        let words = values.map(String.init) + aliases
        switch words.count {
        case 0: return ""
        case 1: return words[0]
        case 2: return "\(words[0]) or \(words[1])"
        default:
            return words.dropLast().joined(separator: ", ") + ", or " + words[words.count - 1]
        }
    }

    public let expertCacheSlots: Int
    public let backend: RuntimeBackend
    public let expertCachePolicy: RuntimeExpertCachePolicy
    public let rdadvisePolicy: RDAdvicePolicyMode
    public let prefillPolicy: RuntimePrefillPolicy
    public let prefillChunkTokens: Int
    public let prefillAttentionPath: RuntimePrefillAttentionPath
    public let headPath: RuntimeHeadPath
    /// Number of routed experts actually evaluated. Values below the
    /// checkpoint's top-8 are an explicit lossy inference mode.
    public let activeRoutedExperts: Int
    /// Hard distinct-expert budget for one prefill layer/tile. Nil preserves
    /// every routed identity. Lossy profiles use this to prevent a 128-token
    /// tile from expanding back to nearly all 128 experts.
    public let prefillExpertUnionCap: Int?
    public let widePrefillTransport: RuntimeWidePrefillTransport
    /// Predictive expert fetch for the next layer. This remains an explicit
    /// experimental rail; the runner consumes the typed decision without
    /// consulting process state inside the recurrent loop.
    public let routePrefetchEnabled: Bool
    /// What the device supervisor does when a guest rail faults.
    public let faultPolicy: RuntimeFaultPolicy

    public init(backend: RuntimeBackend = .metal,
                expertCacheSlots: Int = RuntimeConfiguration.defaultExpertCacheSlots,
                expertCachePolicy: RuntimeExpertCachePolicy = .lfu,
                rdadvisePolicy: RDAdvicePolicyMode = .off,
                prefillEnabled: Bool = true,
                prefillChunkTokens: Int = RuntimeConfiguration.defaultPrefillChunkTokens,
                prefillAttentionPath: RuntimePrefillAttentionPath = .causalSIMD,
                forceLogitsHead: Bool = false,
                activeRoutedExperts: Int = 8,
                prefillExpertUnionCap: Int? = nil,
                widePrefillTransport: RuntimeWidePrefillTransport = .cachedPread,
                routePrefetchEnabled: Bool = false,
                faultPolicy: RuntimeFaultPolicy = .failClosed) {
        precondition(Self.allowedExpertCacheSlots.contains(expertCacheSlots),
                     "unsupported expert-cache slot count")
        precondition(Self.allowedPrefillChunkTokens.contains(prefillChunkTokens),
                     "unsupported prefill chunk size")
        precondition((1...8).contains(activeRoutedExperts),
                     "active routed experts must be in 1...8")
        precondition(prefillExpertUnionCap == nil ||
                     (1...128).contains(prefillExpertUnionCap!),
                     "prefill expert union cap must be in 1...128")
        self.expertCacheSlots = expertCacheSlots
        self.backend = backend
        self.expertCachePolicy = expertCachePolicy
        self.rdadvisePolicy = rdadvisePolicy
        self.prefillPolicy = prefillEnabled ? .chunked : .off
        self.prefillChunkTokens = prefillChunkTokens
        self.prefillAttentionPath = prefillAttentionPath
        self.headPath = forceLogitsHead ? .logits : .fusedRows
        self.activeRoutedExperts = activeRoutedExperts
        self.prefillExpertUnionCap = prefillExpertUnionCap
        self.widePrefillTransport = widePrefillTransport
        self.routePrefetchEnabled = routePrefetchEnabled
        self.faultPolicy = faultPolicy
    }

    public static var production: RuntimeConfiguration {
        RuntimeConfiguration()
    }

    public var fp16RingEnabled: Bool { true }
    public var rdadviseEnabled: Bool { rdadvisePolicy != .off }
    public var prefillConfig: PrefillRuntimeConfig {
        switch prefillPolicy {
        case .off:
            return .off
        case .chunked:
            return .production(chunkTokens: prefillChunkTokens)
        }
    }
    public var modelExpertCachePolicy: ExpertCachePolicy {
        expertCachePolicy == .lru ? .lru : .lfu
    }
}
