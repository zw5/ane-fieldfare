import Foundation
import Metal
import Testing

@testable import ANEFieldfareHotpath

let installedDrafterURL: URL? = {
    let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("../../../../checkpoints/dflash/gemma-4-26B-A4B-it-DFlash").standardizedFileURL
    return FileManager.default.fileExists(atPath: url.appendingPathComponent("model.safetensors").path) ? url : nil
}()

/// "Write a Python function that merges overlapping intervals in a list, with
/// a docstring and a short example." in the chat template.
private let chatPrompt: [Int32] = [2, 105, 2364, 107, 6974, 496, 17856, 1292, 600, 130450, 48490, 20312, 528,
                                   496, 1694, 236764, 607, 496, 6484, 2383, 532, 496, 2822, 2591, 236761, 106,
                                   107, 105, 4368, 107, 100, 45518, 107, 101]

/// Prefill `prompt[start...]`, then decode greedily (or under the producer's
/// sampling rule) until `count` tokens.
private func decode(_ producer: any FusedGreedyProducer & ChunkedPrefillRunner, prompt: [Int32],
                    from start: Int = 0, count: Int, logits: any MTLBuffer) async throws -> [Int32] {
    let prefill = try await producer.prefillChunked(
        tokens: prompt[start...], startPosition: start, outputMode: .greedyIfAvailable,
        config: .production(chunkTokens: 64), into: logits, onProgress: { _ in })
    guard case .greedyToken(let seed) = prefill.seed else {
        Issue.record("prefill did not seed a greedy token")
        return []
    }
    var tokens = [Int32(bitPattern: seed)]
    while tokens.count < count {
        try await producer.produce(token: tokens[tokens.count - 1],
                                   position: prompt.count + tokens.count - 1, into: logits)
        tokens.append(Int32(bitPattern: producer.lastGreedyToken))
    }
    return tokens
}

@Suite(.serialized, .enabled(if: installedCheckpointURL != nil && installedDrafterURL != nil,
                             "installed checkpoint and drafter required"))
struct DFlashSpeculativeTests {
    /// Speculative decoding end to end against plain decoding on a chat
    /// prompt, greedy and sampled with the same rule: accepted drafts handed
    /// out one per call, chained drafts, truncated verifies and plain steps
    /// must reproduce the plain output.
    /// The two may legitimately part at a near-tie (batched and single-row
    /// arithmetic round differently); this prompt's first one is at token
    /// 102, past the tokens compared.
    @Test func speculativeOutputIsPlainGreedyOutput() async throws {
        let context = try MetalContext()
        let model = try Model.load(directoryURL: try #require(installedCheckpointURL), device: context.device)
        let runner = try RealForwardRunner(model: model, context: context, maxContext: 1024)
        let drafter = try await DFlashDrafter(directory: try #require(installedDrafterURL), model: model,
                                              context: context, maxContext: 1024)
        let logits = try #require(context.device.makeBuffer(length: model.config.vocabSize * 2,
                                                             options: .storageModeShared))
        for sampling in [nil, TokenSampling(temperature: 0.8, topK: 64, topP: 0.95, seed: 5)] {
            runner.reset()
            runner.tokenSampling = sampling
            let plain = try await decode(runner, prompt: chatPrompt, count: 64, logits: logits)
            runner.reset()
            let decoder = try SpeculativeDecoder(runner: runner, drafter: drafter)
            decoder.tokenSampling = sampling
            let speculative = try await decode(decoder, prompt: chatPrompt, count: 64, logits: logits)
            decoder.tokenSampling = nil
            let first = zip(plain, speculative).enumerated().first { $0.element.0 != $0.element.1 }?.offset
            #expect(plain.count == 64)
            #expect(speculative == plain,
                    "sampling \(String(describing: sampling)): first difference at \(String(describing: first))")
            #expect(decoder.stats.speculativeSteps > 0 && decoder.stats.acceptedDrafts > 0)
        }
    }

    /// A chat's next turn keeps the cached conversation and prefills only
    /// its new tokens (`PrefixReusingProducer`): the reply equals a fresh
    /// prefill of the whole conversation, for the plain runner and the
    /// speculative decoder.
    @Test func reusedPrefixGivesTheFreshPrefillsReply() async throws {
        let context = try MetalContext()
        let model = try Model.load(directoryURL: try #require(installedCheckpointURL), device: context.device)
        let runner = try RealForwardRunner(model: model, context: context, maxContext: 1024)
        let drafter = try await DFlashDrafter(directory: try #require(installedDrafterURL), model: model,
                                              context: context, maxContext: 1024)
        let logits = try #require(context.device.makeBuffer(length: model.config.vocabSize * 2,
                                                             options: .storageModeShared))
        for speculative in [false, true] {
            let decoder = speculative ? try SpeculativeDecoder(runner: runner, drafter: drafter) : nil
            let producer: any FusedGreedyProducer & ChunkedPrefillRunner & PrefixReusingProducer =
                decoder ?? runner
            // Turn 1, then a follow-up appended to everything the cache holds.
            producer.reset()
            let reply = try await decode(producer, prompt: chatPrompt, count: 24, logits: logits)
            let cachedCount = chatPrompt.count + reply.count - 1
            let second = chatPrompt + reply.dropLast() + [106, 107, 105, 2364, 107, 236840, 106, 107, 105, 4368,
                                                          107, 100, 45518, 107, 101]
            #expect(producer.canReuse(prefix: cachedCount))
            try producer.rewindForReuse(to: cachedCount)
            let reused = try await decode(producer, prompt: second, from: cachedCount, count: 24, logits: logits)
            producer.reset()
            let fresh = try await decode(producer, prompt: second, count: 24, logits: logits)
            #expect(reused == fresh, "speculative \(speculative)")
        }
    }
}
