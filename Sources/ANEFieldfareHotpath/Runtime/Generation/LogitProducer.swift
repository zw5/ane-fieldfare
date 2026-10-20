public import Metal

/// Produces next-token logits for `runRawCompletion`. The production
/// implementation is `RealForwardRunner`; tests use scripted logits so decode
/// behavior stays independent of the kernel stack.
///
/// Not `Sendable`: a producer holds one generation's GPU state and is driven
/// from the single task running that generation.
public protocol LogitProducer: AnyObject {
    /// Clear any per-generation state, such as KV cache.
    func reset()
    /// Run one token at `position`, leaving FP16 logits in `logits`.
    func produce(token: Int32, position: Int, into logits: any MTLBuffer) async throws
}

/// A producer whose head leaves the greedy next token on the device instead of
/// writing logits; the generation loop reads `lastGreedyToken` after each
/// `produce`.
public protocol FusedGreedyProducer: LogitProducer {
    var usesFusedGreedyHead: Bool { get }
    var lastGreedyToken: UInt32 { get }
}

/// A fused-head producer that also samples on the GPU: with `tokenSampling`
/// set, `lastGreedyToken` is its sampled token instead of the argmax.
public protocol SamplingTokenProducer: FusedGreedyProducer {
    var tokenSampling: TokenSampling? { get set }
}

/// A producer that can keep the first tokens of its cache for the next
/// generation instead of resetting: a chat's next turn re-sends the whole
/// conversation, and only what follows the shared prefix needs a prefill.
public protocol PrefixReusingProducer: LogitProducer {
    /// Whether `rewindForReuse(to:)` can keep the first `position` tokens.
    func canReuse(prefix position: Int) -> Bool
    /// Abandons everything past the first `position` cached tokens.
    func rewindForReuse(to position: Int) throws
}

protocol ContextWindowReporting {
    var maxContext: Int { get }
}

public enum PrefillOutputMode: Sendable, Equatable {
    case logits
    case greedyIfAvailable
}

public enum PrefillSeed: Sendable, Equatable {
    case logitsWritten
    case greedyToken(UInt32)
}

public struct PrefillResult: Sendable, Equatable {
    public let newPosition: Int
    public let seed: PrefillSeed

    public init(newPosition: Int, seed: PrefillSeed) {
        self.newPosition = newPosition
        self.seed = seed
    }
}

protocol ChunkedPrefillRunner: LogitProducer {
    /// Prefill a prompt slice using the chunked production runtime.
    func prefillChunked(tokens: ArraySlice<Int32>,
                        startPosition: Int,
                        outputMode: PrefillOutputMode,
                        config: PrefillRuntimeConfig,
                        into logits: any MTLBuffer,
                        onProgress: (Int) -> Void) async throws -> PrefillResult
}

protocol MultimodalPrefillRunner: LogitProducer {
    func prefillMultimodal(input: MultimodalPrefillInput,
                           startPosition: Int,
                           outputMode: PrefillOutputMode,
                           config: PrefillRuntimeConfig,
                           into logits: any MTLBuffer,
                           onProgress: (Int) -> Void) async throws -> PrefillResult
}
