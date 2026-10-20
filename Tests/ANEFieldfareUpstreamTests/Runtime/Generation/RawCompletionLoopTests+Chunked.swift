import Testing

@testable import ANEFieldfareHotpath

extension RawCompletionLoopTests {
    @Test func chunkedLogitsSeedProducesFirstTokenThroughTheChunkedEntryPoint() async throws {
        let context = try MetalContext()
        let tokenizer = try await GFTokenizer.load()
        let tokenA = tokenizer.encode("a", addBOS: false).first!
        let producer = ChunkedTestProducer(vocabSize: tokenizer.vocabSize, firstToken: tokenA)
        let promptIDs = tokenizer.encode("go", addBOS: true)
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)
        var prefills: [(Int, Int)] = []
        var tokens: [Int32] = []

        let result = try await runRawCompletion(
            producer: producer,
            tokenizer: tokenizer,
            promptIds: promptIDs,
            config: GenerationConfig(maxNewTokens: 1, temperature: 0),
            context: context,
            scratch: scratch,
            prefillConfig: .production(chunkTokens: 32)) { progress in
                switch progress {
                case .prefill(let done, let total): prefills.append((done, total))
                case .token(_, let id, _): tokens.append(id)
                case .tail: break
                }
            }

        #expect(result.newTokens == 1)
        #expect(tokens == [tokenA])
        #expect(producer.chunkedCalls == 1)
        #expect(producer.produceCalls == 0)
        #expect(producer.lastOutputMode == .logits)
        #expect(producer.lastConfig == .production(chunkTokens: 32))
        #expect(prefills.count == 1)
        #expect(prefills.first?.0 == promptIDs.count)
        #expect(prefills.first?.1 == promptIDs.count)
    }

    @Test func chunkedPrefillRejectsGreedySeedWhenLogitsRequested() async throws {
        let context = try MetalContext()
        let tokenizer = try await GFTokenizer.load()
        let tokenA = tokenizer.encode("a", addBOS: false).first!
        let producer = ChunkedTestProducer(
            vocabSize: tokenizer.vocabSize,
            firstToken: tokenA,
            seed: .greedyToken(UInt32(bitPattern: tokenA)))
        let promptIDs = tokenizer.encode("go", addBOS: true)
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)

        do {
            _ = try await runRawCompletion(
                producer: producer,
                tokenizer: tokenizer,
                promptIds: promptIDs,
                config: GenerationConfig(maxNewTokens: 1, temperature: 0.7),
                context: context,
                scratch: scratch,
                prefillConfig: .production(chunkTokens: 32)) { _ in }
            Issue.record("expected unsupported chunked prefill seed")
        } catch let error as PrefillError {
            guard case .unsupportedPrefillSeed(let reason) = error else {
                Issue.record("unexpected PrefillError \(error)")
                return
            }
            #expect(reason.contains("RawCompletion chunked prefill requested logits"))
        }

        #expect(producer.chunkedCalls == 1)
        #expect(producer.produceCalls == 0)
        #expect(producer.lastOutputMode == .logits)
    }
}
