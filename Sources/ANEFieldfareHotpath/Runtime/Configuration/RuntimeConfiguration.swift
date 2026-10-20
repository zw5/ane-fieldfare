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

/// What scores the next layer's experts for the decode early read, or `off`.
/// `router` uses the next layer's own router; `fitted` uses the fitted guess
/// shipped for this model, or the router for a model with none. See
/// `NextLayerExpertPrefetcher`.
public enum RuntimeEarlyExpertRead: String, Codable, Sendable, CaseIterable {
    case off
    case router
    case fitted
}

/// A runtime environment variable set to a value it does not take.
public struct RuntimeEnvironmentError: Error, CustomStringConvertible {
    public let variable: String
    public let value: String
    public let expected: String

    public var description: String { "\(variable)=\(value) must be \(expected)" }
}

public struct RuntimeConfiguration: Sendable, Equatable {
    public static let defaultPrefillChunkTokens = 1024
    public static let allowedPrefillChunkTokens = PrefillRuntimeConfig.allowedChunkTokens
    /// Renders `values` as "32, 64, or 128" for help text and rejections,
    /// followed by parse-level `aliases` that are not members of `values`.
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

    public let backend: RuntimeBackend
    public let prefillPolicy: RuntimePrefillPolicy
    public let prefillChunkTokens: Int
    public let headPath: RuntimeHeadPath
    /// Number of routed experts actually evaluated. Values below the
    /// checkpoint's top-8 are an explicit lossy inference mode.
    public let activeRoutedExperts: Int
    /// Hard distinct-expert budget for one prefill layer/tile. Nil preserves
    /// every routed identity. Lossy profiles use this to prevent a 128-token
    /// tile from expanding back to nearly all 128 experts.
    public let prefillExpertUnionCap: Int?
    /// What the device supervisor does when a guest rail faults.
    public let faultPolicy: RuntimeFaultPolicy
    /// Decode early read of the next layer's likeliest experts in idle storage time.
    public let earlyExpertRead: RuntimeEarlyExpertRead

    public init(backend: RuntimeBackend = .metal,
                prefillPolicy: RuntimePrefillPolicy = .chunked,
                prefillChunkTokens: Int = RuntimeConfiguration.defaultPrefillChunkTokens,
                headPath: RuntimeHeadPath = .fusedRows,
                activeRoutedExperts: Int = 8,
                prefillExpertUnionCap: Int? = nil,
                faultPolicy: RuntimeFaultPolicy = .failClosed,
                earlyExpertRead: RuntimeEarlyExpertRead = .fitted) {
        precondition(Self.allowedPrefillChunkTokens.contains(prefillChunkTokens),
                     "unsupported prefill chunk size")
        precondition((1...8).contains(activeRoutedExperts),
                     "active routed experts must be in 1...8")
        precondition(prefillExpertUnionCap.map { (1...128).contains($0) } ?? true,
                     "prefill expert union cap must be in 1...128")
        self.backend = backend
        self.prefillPolicy = prefillPolicy
        self.prefillChunkTokens = prefillChunkTokens
        self.headPath = headPath
        self.activeRoutedExperts = activeRoutedExperts
        self.prefillExpertUnionCap = prefillExpertUnionCap
        self.faultPolicy = faultPolicy
        self.earlyExpertRead = earlyExpertRead
    }

    public var prefillConfig: PrefillRuntimeConfig {
        switch prefillPolicy {
        case .off:
            return .off
        case .chunked:
            return .production(chunkTokens: prefillChunkTokens)
        }
    }
}
