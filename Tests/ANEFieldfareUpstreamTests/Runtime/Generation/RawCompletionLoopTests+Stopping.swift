import Foundation
import Metal
import Testing
import ANEFieldfareValidationSupport

@testable import ANEFieldfareHotpath
import Tokenizers

extension RawCompletionLoopTests {
  @Test func stopsOnEOS() async throws {
    let tok = try await GFTokenizer.load()
    let idA = tok.encode("a", addBOS: false).first!
    let idB = tok.encode("b", addBOS: false).first!
    let (collected, result) = try await runLoop(
      seq: [idA, idB], end: tok.eosID,
      config: GenerationConfig(maxNewTokens: 50, temperature: 0))
    #expect(result.reason == .eos)
    #expect(result.newTokens == 3)
    #expect(collected.tokens.map(\.1) == [idA, idB])
  }

  @Test func stopsOnMaxTokensAndCountsExactly() async throws {
    let tok = try await GFTokenizer.load()
    let idA = tok.encode("a", addBOS: false).first!
    let (collected, result) = try await runLoop(
      seq: [idA, idA], end: idA,
      config: GenerationConfig(maxNewTokens: 5, temperature: 0))
    #expect(result.reason == .maxTokens)
    #expect(result.newTokens == 5)
    #expect(collected.tokens.count == 5)
    #expect(collected.tokens.map(\.0) == [0, 1, 2, 3, 4])
  }

  @Test func stopStringSpanningTokensReportsWithheldKVTokens() async throws {
    let tok = try await GFTokenizer.load()
    let idA = tok.encode("a", addBOS: false).first!
    let textA = tok.decode([idA], skipSpecialTokens: true)
    let (collected, result) = try await runLoop(
      seq: [idA, idA, idA], end: idA,
      config: GenerationConfig(maxNewTokens: 50, temperature: 0,
                               stopStrings: [textA + textA]))
    #expect(result.reason == .stopString)
    #expect(result.withheldTrailingKVTokens == 1)
    #expect(result.kvBackedTokenIDs.count == result.kvPosition)
    #expect(result.kvBackedTokenIDs.last == idA)
    #expect(result.uncommittedBoundaryTokenIDs == [idA])
    #expect(collected.tails.isEmpty)
  }

  @Test func stopMatchThatFlushesHeldTextDoesNotOvercountWithheldTokens() async throws {
    let tok = try await GFTokenizer.load()
    let idA = try #require(tok.tokenizer.convertTokenToId("a").map(Int32.init))
    let idCab = try #require(tok.tokenizer.convertTokenToId("cab").map(Int32.init))
    let (collected, result) = try await runLoop(
      seq: [idA, idCab], end: idA,
      config: GenerationConfig(maxNewTokens: 50, temperature: 0, stopStrings: ["ab"]))
    #expect(result.reason == .stopString)
    #expect(result.withheldTrailingKVTokens == 0)
    #expect(result.kvBackedTokenIDs.last == idA)
    #expect(collected.tokens.map(\.2) == ["", "ac"])
  }

  @Test func shouldStopEndsDecodeWithCancelledReason() async throws {
    let ctx = try MetalContext()
    let tok = try await GFTokenizer.load()
    let idA = tok.encode("a", addBOS: false).first!
    let producer = ScriptedLogitProducer(vocabSize: tok.vocabSize,
                                         step: automaton([idA, idA], end: idA))
    let promptIds = tok.encode("go", addBOS: true)
    let scratch = try RawCompletionScratch(context: ctx, vocab: tok.vocabSize)
    var seen = 0
    var cancel = false
    let result = try await runRawCompletion(
      producer: producer, tokenizer: tok, promptIds: promptIds,
      config: GenerationConfig(maxNewTokens: 50, temperature: 0),
      context: ctx, scratch: scratch, prefillConfig: .off,
      shouldStop: { cancel }) { progress in
        if case .token = progress {
          seen += 1
          if seen == 2 { cancel = true }
        }
      }
    #expect(result.reason == .cancelled)
    #expect(result.newTokens == 2)
    #expect(result.withheldTrailingKVTokens == 0)
    #expect(result.kvBackedTokenIDs.count == promptIds.count + 1)
    #expect(result.uncommittedBoundaryTokenIDs.count == 1)
  }

  @Test func cancellationPropagatesMidDecode() async throws {
    let ctx = try MetalContext()
    let tok = try await GFTokenizer.load()
    let idA = tok.encode("a", addBOS: false).first!
    let producer = ScriptedLogitProducer(
      vocabSize: tok.vocabSize,
      step: automaton([idA, idA], end: idA))
    let promptIds = tok.encode("go", addBOS: true)
    let scratch = try RawCompletionScratch(context: ctx, vocab: tok.vocabSize)

    let task = Task {
      try await runRawCompletion(
        producer: producer, tokenizer: tok,
        promptIds: promptIds,
        config: GenerationConfig(maxNewTokens: 100_000, temperature: 0),
        context: ctx, scratch: scratch,
        prefillConfig: .off
      ) { progress in
        if case .token(let index, _, _) = progress, index == 2 {
          unsafe withUnsafeCurrentTask { unsafe $0?.cancel() }
        }
      }
    }
    await #expect(throws: CancellationError.self) {
      _ = try await task.value
    }
  }

  @Test func rawCompletionRejectsContextOverflowBeforeReset() async throws {
    let context = try MetalContext()
    let tokenizer = try await GFTokenizer.load()
    let promptIDs = tokenizer.encode("one two three", addBOS: true)
    let producer = CountingProducer(maxContext: promptIDs.count)
    let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)

    do {
      _ = try await runRawCompletion(
        producer: producer,
        tokenizer: tokenizer,
        promptIds: promptIDs,
        config: GenerationConfig(maxNewTokens: 1, temperature: 0),
        context: context,
        scratch: scratch,
        prefillConfig: .off) { _ in }
      Issue.record("expected context overflow")
    } catch let error as GeneratorError {
      guard case .contextOverflow(let prompt, let maxNew, let maxContext) = error else {
        Issue.record("unexpected GeneratorError \(error)")
        return
      }
      #expect(prompt == promptIDs.count)
      #expect(maxNew == 1)
      #expect(maxContext == promptIDs.count)
    }

    #expect(producer.resetCalls == 0)
    #expect(producer.produceCalls == 0)
  }
}
