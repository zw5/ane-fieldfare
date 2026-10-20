import Testing
import Foundation
import Metal
@testable import ANEFieldfareHotpath
import ANEFieldfareValidationSupport

/// Shared raw-completion loop validation: stops, withheld KV accounting,
/// chunked prefill seeding, and cancellation, via `ScriptedLogitProducer`
/// (kernel-independent).
@Suite struct RawCompletionLoopTests {

    func automaton(_ seq: [Int32], end: Int32) -> @Sendable (Int32, Int) -> ScriptedLogitProducer.Step {
        let next: [Int32: Int32] = {
            var n: [Int32: Int32] = [:]
            for i in 0..<max(0, seq.count - 1) { n[seq[i]] = seq[i + 1] }
            if let last = seq.last { n[last] = end }
            return n
        }()
        let first = seq.first ?? end
        return { input, _ in .argmax(next[input] ?? first) }
    }

    struct Collected {
        var tokens: [(Int, Int32, String)] = []
        var tails: [String] = []
    }

    /// Counts calls only: the cases using it must be refused before either.
    final class CountingProducer: LogitProducer, ContextWindowReporting {
        let maxContext: Int
        private(set) var resetCalls = 0
        private(set) var produceCalls = 0

        init(maxContext: Int) { self.maxContext = maxContext }

        func reset() { resetCalls += 1 }

        func produce(token: Int32, position: Int, into logits: any MTLBuffer) async throws {
            produceCalls += 1
        }
    }

    final class ChunkedTestProducer: LogitProducer, ChunkedPrefillRunner {
        let vocabSize: Int
        private let firstToken: Int32
        private let seed: PrefillSeed
        private(set) var resetCalls = 0
        private(set) var produceCalls = 0
        private(set) var chunkedCalls = 0
        private(set) var lastOutputMode: PrefillOutputMode?
        private(set) var lastConfig: PrefillRuntimeConfig?

        init(vocabSize: Int,
             firstToken: Int32,
             seed: PrefillSeed = .logitsWritten) {
            self.vocabSize = vocabSize
            self.firstToken = firstToken
            self.seed = seed
        }

        func reset() {
            resetCalls += 1
            produceCalls = 0
            chunkedCalls = 0
            lastOutputMode = nil
            lastConfig = nil
        }

        func produce(token: Int32, position: Int, into logits: any MTLBuffer) async throws {
            produceCalls += 1
        }

        func prefillChunked(tokens: ArraySlice<Int32>,
                            startPosition: Int,
                            outputMode: PrefillOutputMode,
                            config: PrefillRuntimeConfig,
                            into logits: any MTLBuffer,
                            onProgress: (Int) -> Void) async throws -> PrefillResult {
            chunkedCalls += 1
            lastOutputMode = outputMode
            lastConfig = config
            let ptr = unsafe logits.contents().bindMemory(to: Float16.self, capacity: vocabSize)
            for i in 0..<vocabSize { unsafe ptr[i] = Float16(-30.0) }
            unsafe ptr[Int(firstToken)] = Float16(30.0)
            onProgress(tokens.count)
            return PrefillResult(newPosition: startPosition + tokens.count,
                                 seed: seed)
        }
    }

    func runLoop(seq: [Int32], end: Int32,
                 config: GenerationConfig) async throws -> (Collected, RawDecodeResult) {
        let ctx = try MetalContext()
        let tok = try await GFTokenizer.load()
        let producer = ScriptedLogitProducer(vocabSize: tok.vocabSize,
                                             step: automaton(seq, end: end))
        let promptIds = tok.encode("go", addBOS: true)
        let scratch = try RawCompletionScratch(context: ctx, vocab: tok.vocabSize)
        var collected = Collected()
        let result = try await runRawCompletion(producer: producer, tokenizer: tok,
                                                promptIds: promptIds, config: config,
                                                context: ctx, scratch: scratch,
                                                prefillConfig: .off) { progress in
            switch progress {
            case .prefill: break
            case .token(let index, let id, let delta): collected.tokens.append((index, id, delta))
            case .tail(let text): collected.tails.append(text)
            }
        }
        return (collected, result)
    }

}
