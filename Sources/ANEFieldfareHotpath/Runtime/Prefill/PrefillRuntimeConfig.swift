import Foundation

public enum PrefillError: Error, CustomStringConvertible, Equatable {
    case chunkedUnsupported(String)
    case chunkedRunnerDirty(String)
    case prefillCursorMismatch(String)
    case unsupportedPrefillSeed(String)

    public var description: String {
        switch self {
        case .chunkedUnsupported(let reason),
             .chunkedRunnerDirty(let reason),
             .prefillCursorMismatch(let reason),
             .unsupportedPrefillSeed(let reason):
            return reason
        }
    }
}

/// Tracks a prefill chunk that has written KV rows but not yet advanced the
/// cursor; a runner left in that state must be reset before reuse.
struct PrefillChunkCommitState {
    private var inFlight: Range<Int>?

    mutating func markDirty(startPosition: Int, tokenCount: Int) {
        precondition(startPosition >= 0, "prefill dirty startPosition must be non-negative")
        precondition(tokenCount > 0, "prefill dirty tokenCount must be positive")
        inFlight = startPosition..<(startPosition + tokenCount)
    }

    mutating func markCommitted() {
        inFlight = nil
    }

    func requireClean(operation: String) throws {
        guard let inFlight else { return }
        throw PrefillError.chunkedRunnerDirty(
            "\(operation) rejected because a previous chunked prefill wrote KV rows for in-flight chunk [\(inFlight.lowerBound), \(inFlight.upperBound)) but did not commit; call reset() before reusing the runner")
    }
}

struct PrefillChunkSpan: Sendable, Equatable {
    let tokenOffset: Int
    let tokenCount: Int
    let startPosition: Int
    let completedCount: Int
}

/// One item of a multimodal prefill turn: either a run of text small enough to
/// fit the chunk scratch, or exactly one image span, which executes whole.
struct PrefillWorkItem: Sendable, Equatable {
    let range: Range<Int>
    let imageIndex: Int?

    var isImage: Bool { imageIndex != nil }
}

enum PrefillChunkPlanner {
    /// Splits a multimodal prompt into the items a turn executes, in order.
    ///
    /// Text is cut at the same clamp the scratch layout applies, so the split
    /// and the buffer it runs against can never disagree; image spans pass
    /// through whole, because an image's features are one indivisible block.
    /// `imageRanges` must be sorted, non-overlapping, and inside `tokenCount`.
    static func multimodalWork(tokenCount: Int,
                               imageRanges: [Range<Int>],
                               chunkTokens: Int) -> [PrefillWorkItem] {
        precondition(tokenCount >= 0, "prefill tokenCount must be non-negative")
        let chunk = max(1, min(chunkTokens, PrefillRuntimeConfig.maxChunkTokens))
        var work: [PrefillWorkItem] = []

        func appendText(_ range: Range<Int>) {
            guard !range.isEmpty else { return }
            var offset = range.lowerBound
            while offset < range.upperBound {
                let end = min(range.upperBound, offset + chunk)
                work.append(PrefillWorkItem(range: offset..<end, imageIndex: nil))
                offset = end
            }
        }

        var cursor = 0
        for (index, imageRange) in imageRanges.enumerated() {
            appendText(cursor..<imageRange.lowerBound)
            work.append(PrefillWorkItem(range: imageRange, imageIndex: index))
            cursor = imageRange.upperBound
        }
        appendText(cursor..<tokenCount)
        return work
    }

    static func spans(tokenCount: Int,
                      startPosition: Int,
                      config: PrefillRuntimeConfig) -> [PrefillChunkSpan] {
        precondition(tokenCount >= 0, "prefill tokenCount must be non-negative")
        precondition(startPosition >= 0, "prefill startPosition must be non-negative")
        let chunk = max(1, min(config.chunkTokens, PrefillRuntimeConfig.maxChunkTokens))
        guard tokenCount > 0 else { return [] }

        var spans: [PrefillChunkSpan] = []
        spans.reserveCapacity((tokenCount + chunk - 1) / chunk)
        var offset = 0
        while offset < tokenCount {
            let count = min(chunk, tokenCount - offset)
            let completed = offset + count
            spans.append(PrefillChunkSpan(tokenOffset: offset,
                                          tokenCount: count,
                                          startPosition: startPosition + offset,
                                          completedCount: completed))
            offset = completed
        }
        return spans
    }
}

public struct PrefillRuntimeConfig: Sendable, Equatable {
    public enum Mode: String, Sendable, Equatable {
        case off
        case chunked
    }

    public static let maxChunkTokens = 4096
    public static let allowedChunkTokens = [32, 64, 128, 256, 512, 1024, 2048, 4096]

    /// The smallest allowed chunk that covers `promptTokens`, capped.
    ///
    /// One chunk means each layer's experts are read once for the whole
    /// prefill, which is the floor, so this asks for the smallest size that
    /// reaches it and never more than the prompt needs.
    public static func autoChunkTokens(
        promptTokens: Int,
        cap: Int = maxChunkTokens
    ) -> Int {
        let ceiling = min(cap, maxChunkTokens)
        for candidate in allowedChunkTokens where candidate >= promptTokens {
            return min(candidate, ceiling)
        }
        return ceiling
    }

    public let mode: Mode
    public let chunkTokens: Int
    private init(mode: Mode, chunkTokens: Int) {
        self.mode = mode
        self.chunkTokens = chunkTokens
    }

    public static var off: PrefillRuntimeConfig {
        PrefillRuntimeConfig(mode: .off, chunkTokens: 64)
    }

    public static var defaultChunked: PrefillRuntimeConfig {
        production(chunkTokens: 64)
    }

    /// This config at a different chunk size, all other settings intact. The
    /// multimodal path runs each work item at the size it was planned at.
    public func replacingChunkTokens(_ tokens: Int) -> PrefillRuntimeConfig {
        PrefillRuntimeConfig(mode: mode, chunkTokens: max(1, tokens))
    }

    /// Whether the chunked path an image prompt needs is switched on.
    public var servesImagePrompt: Bool { mode == .chunked }

    /// The config an image prompt can run under, or nil when it already runs.
    /// Prefill mode is a performance choice, so an image turn is coerced rather
    /// than refused; callers report the coercion.
    public func coercedForImagePrompt() -> PrefillRuntimeConfig? {
        servesImagePrompt ? nil : .defaultChunked
    }

    public static func production(chunkTokens: Int) -> PrefillRuntimeConfig {
        precondition(RuntimeConfiguration.allowedPrefillChunkTokens.contains(chunkTokens),
                     "unsupported prefill chunk size")
        return PrefillRuntimeConfig(mode: .chunked, chunkTokens: chunkTokens)
    }
}
