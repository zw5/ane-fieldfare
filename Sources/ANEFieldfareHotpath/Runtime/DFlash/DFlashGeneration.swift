import Foundation
import Metal

/// Counters for one speculative generation.
public struct DFlashGenerationStats: Sendable {
    public var steps = 0
    public var draftedTokens = 0
    public var acceptedDrafts = 0
    public var generatedTokens = 0
    public var prefillSeconds = 0.0
    public var draftSeconds = 0.0
    public var verifySeconds = 0.0
    public var contextSeconds = 0.0
    public var decodeSeconds = 0.0
    /// Tokens produced per verify step, the acceptance length.
    public var meanAcceptanceLength: Double { steps > 0 ? Double(generatedTokens - 1) / Double(steps) : 0 }

    public var profile: [String: Any] {
        ["steps": steps, "drafted_tokens": draftedTokens, "accepted_drafts": acceptedDrafts,
         "generated_tokens": generatedTokens, "mean_acceptance_length": meanAcceptanceLength,
         "prefill_seconds": prefillSeconds, "draft_seconds": draftSeconds,
         "verify_seconds": verifySeconds, "context_seconds": contextSeconds,
         "decode_seconds": decodeSeconds,
         "decode_tokens_per_second": decodeSeconds > 0 ? Double(generatedTokens - 1) / decodeSeconds : 0]
    }
}

/// Greedy DFlash speculative decoding: the drafter proposes a block after the
/// last committed token, the target verifies the block in one chunk, and the
/// longest prefix matching the target's greedy choices is kept with the
/// target's next token. Output equals plain greedy decoding up to the
/// numerical difference between a batched verify chunk and single-token decode.
public enum DFlashGeneration {
    public static func generateGreedy(
        runner: RealForwardRunner,
        drafter: DFlashDrafter,
        context: MetalContext,
        prompt: [Int32],
        maxNewTokens: Int,
        blockSize: Int,
        stopTokens: Set<Int32> = [],
        prefillChunkTokens: Int,
        onTokens: ([Int32]) -> Void = { _ in }
    ) async throws -> (tokens: [Int32], stats: DFlashGenerationStats) {
        guard !prompt.isEmpty, maxNewTokens > 0 else { return ([], DFlashGenerationStats()) }
        guard blockSize >= 2, blockSize <= min(drafter.config.blockSize, RealForwardRunner.maxVerifyTokens) else {
            throw DFlashError.invalidState("block size \(blockSize) unsupported")
        }
        var stats = DFlashGenerationStats()
        let layers = drafter.config.contextLayers
        runner.reset()
        drafter.reset()
        defer { runner.prefillCapture = nil }

        // Prefill: every prompt row's features enter the drafter context.
        var captureError: Error?
        runner.prefillCapture = PrefillCapture(hiddenLayers: layers) { chunk in
            let started = Date()
            do {
                try drafter.appendContext(captured: chunk.hidden, stride: chunk.tokenCount, rows: chunk.tokenCount)
            } catch { captureError = error }
            stats.contextSeconds += Date().timeIntervalSince(started)
        }
        guard let logits = context.device.makeBuffer(length: drafter.vocabSize * 2,
                                                          options: .storageModeShared) else {
            throw DFlashError.invalidState("logits allocation failed")
        }
        let prefillStart = Date()
        let prefill = try await runner.prefillChunked(
            tokens: prompt[...], startPosition: 0, outputMode: .greedyIfAvailable,
            config: .production(chunkTokens: prefillChunkTokens), into: logits, onProgress: { _ in })
        stats.prefillSeconds = Date().timeIntervalSince(prefillStart)
        if let captureError { throw captureError }
        guard case .greedyToken(let first) = prefill.seed else {
            throw DFlashError.invalidState("prefill did not produce a greedy token; the fused head is required")
        }

        var tokens = [Int32(bitPattern: first)]
        onTokens(tokens)
        stats.generatedTokens = 1
        if stopTokens.contains(tokens[0]) { return (tokens, stats) }

        var verifyHidden = [Float16]()
        runner.prefillCapture = PrefillCapture(hiddenLayers: layers) { chunk in
            verifyHidden = Array(chunk.hidden)
        }
        var position = prompt.count
        let decodeStart = Date()
        while tokens.count < maxNewTokens {
            let count = min(blockSize, maxNewTokens - tokens.count + 1, runner.maxContext - position)
            guard count >= 2 else { break }
            let anchor = tokens[tokens.count - 1]

            var started = Date()
            let drafts = try drafter.propose(anchor: anchor, count: count)
            stats.draftSeconds += Date().timeIntervalSince(started)

            started = Date()
            let verified = try await runner.verify(tokens: [anchor] + drafts, startPosition: position)
            stats.verifySeconds += Date().timeIntervalSince(started)

            var accepted = 0
            while accepted < drafts.count, drafts[accepted] == verified[accepted] { accepted += 1 }
            var produced = Array(drafts[0..<accepted]) + [verified[accepted]]
            stats.steps += 1
            stats.draftedTokens += drafts.count
            stats.acceptedDrafts += accepted

            // Rows [0, accepted] (anchor and accepted drafts) are now context.
            if accepted + 1 < count { try runner.rewind(to: position + accepted + 1) }
            started = Date()
            try verifyHidden.withUnsafeBufferPointer {
                try drafter.appendContext(captured: $0, stride: count, rows: accepted + 1)
            }
            stats.contextSeconds += Date().timeIntervalSince(started)
            position += accepted + 1

            if let stop = produced.firstIndex(where: { stopTokens.contains($0) }) {
                produced = Array(produced[...stop])
            }
            produced = Array(produced.prefix(maxNewTokens - tokens.count))
            tokens += produced
            stats.generatedTokens = tokens.count
            onTokens(produced)
            if let last = produced.last, stopTokens.contains(last) { break }
        }
        stats.decodeSeconds = Date().timeIntervalSince(decodeStart)
        return (tokens, stats)
    }
}
