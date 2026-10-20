public import Metal

/// Plain greedy decoding whose head reads only the hot vocabulary on most
/// tokens, with output bitwise equal to the full head's.
///
/// Each decode step guesses its token from `HotVocabulary`, the head rows of
/// the tokens a decode likely emits, and feeds the guess to the next step.
/// After `LMHeadChainInt4.maxCheckRows` steps, one pass of the full head over
/// the saved normalized rows gives each step's true greedy token, the same
/// bits the single-row head computes. Guesses up to the first wrong one are
/// handed out; the wrong one is replaced by the true token and the steps
/// after it are rewound. Each check also adds its tokens to the table.
///
/// The full head reads its 262k rows once per block instead of once per
/// token. A wrong guess costs the steps after it in its block; the tokens
/// handed out are the full head's either way.
public final class HotHeadDecoder: LogitProducer, ChunkedPrefillRunner, ContextWindowReporting,
    FusedGreedyProducer, PrefixReusingProducer {
    public let runner: RealForwardRunner
    public private(set) var lastGreedyToken: UInt32 = 0
    public var usesFusedGreedyHead: Bool { true }
    public var maxContext: Int { runner.maxContext }

    /// Checked tokens not yet handed out: (position, token, its successor).
    private var checked: [(position: Int, token: Int32, next: UInt32)] = []

    /// nil when `runner` has no hot vocabulary, i.e. ships no list for its
    /// vocabulary or runs the logits head.
    public init?(runner: RealForwardRunner) {
        guard runner.usesFusedGreedyHead, runner.hotVocabulary != nil else { return nil }
        self.runner = runner
    }

    public func reset() {
        runner.reset()
        checked.removeAll()
    }

    public func canReuse(prefix position: Int) -> Bool { runner.canRewind(to: position) }

    public func rewindForReuse(to position: Int) throws {
        checked.removeAll()
        try runner.rewindForReuse(to: position)
    }

    func prefillChunked(tokens: ArraySlice<Int32>, startPosition: Int, outputMode: PrefillOutputMode,
                        config: PrefillRuntimeConfig, into logits: any MTLBuffer,
                        onProgress: (Int) -> Void) async throws -> PrefillResult {
        // A prompt's tokens are likely in its answer.
        runner.hotVocabulary?.insert(tokens.map { UInt32(bitPattern: $0) })
        let result = try await runner.prefillChunked(tokens: tokens, startPosition: startPosition,
                                                     outputMode: outputMode, config: config, into: logits,
                                                     onProgress: onProgress)
        if case .greedyToken(let token) = result.seed {
            lastGreedyToken = token
            runner.hotVocabulary?.insert(CollectionOfOne(token))
        }
        return result
    }

    public func produce(token: Int32, position: Int, into logits: any MTLBuffer) async throws {
        if let next = checked.first, next.position == position, next.token == token {
            checked.removeFirst()
            lastGreedyToken = next.next
            return
        }
        checked.removeAll()
        if runner.kvPosition > position { try runner.rewind(to: position) }
        let steps = min(LMHeadChainInt4.maxCheckRows, maxContext - position)
        var guesses: [UInt32] = []
        var input = token
        defer { runner.hotHead = nil }
        for row in 0..<steps {
            // The last step's guess would only be compared, so it checks instead.
            runner.hotHead = (row, row == steps - 1)
            try await runner.produce(token: input, position: position + row, into: logits)
            guesses.append(runner.lastGreedyToken)
            input = Int32(bitPattern: runner.lastGreedyToken)
        }
        let truth = runner.checkedTokens
        runner.hotVocabulary?.insert(truth)
        // Step i ran on a correct input while every earlier guess was right.
        var good = 0
        while good < steps, guesses[good] == truth[good] { good += 1 }
        let kept = min(good + 1, steps)
        if kept < steps { try runner.rewind(to: position + kept) }
        lastGreedyToken = truth[0]
        checked = (1..<kept).map { i in
            (position: position + i, token: Int32(bitPattern: truth[i - 1]), next: truth[i])
        }
    }
}
