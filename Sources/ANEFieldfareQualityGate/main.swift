import ANEFieldfareHotpath
import Foundation
import Metal
import Darwin

private struct Fixture: Decodable {
    let prompt_token_ids: [Int32]
    let continuation_token_ids: [Int32]
}

/// Writes full-vocabulary logits under a fixed continuation, including reads
/// from KV produced by prefill. Arms run in separate processes. Comparison is
/// deliberately external, so generated candidate tokens cannot change inputs.
@main private enum QualityGate {
    static func main() async { await run() }

    nonisolated static func run() async {
        do {
            let args = CommandLine.arguments
            guard args.count == 5, let chunk = Int(args[3]),
                  RuntimeConfiguration.allowedPrefillChunkTokens.contains(chunk) else {
                throw NSError(domain: "QualityGate", code: 2, userInfo: [NSLocalizedDescriptionKey:
                    "usage: ANEFieldfareQualityGate <model-directory> <token-fixture.json> <chunk> <output-prefix>"])
            }
            let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: URL(fileURLWithPath: args[2])))
            guard !fixture.prompt_token_ids.isEmpty, !fixture.continuation_token_ids.isEmpty,
                  fixture.prompt_token_ids.count + fixture.continuation_token_ids.count < 4096,
                  (fixture.prompt_token_ids + fixture.continuation_token_ids).allSatisfy({ $0 >= 0 && $0 < 262144 }) else {
                throw NSError(domain: "QualityGate", code: 2, userInfo: [NSLocalizedDescriptionKey: "invalid token fixture"])
            }
            let outputURL = URL(fileURLWithPath: args[4] + ".logits.f16")
            guard !FileManager.default.fileExists(atPath: outputURL.path),
                  !FileManager.default.fileExists(atPath: args[4] + ".json") else {
                throw NSError(domain: "QualityGate", code: 3, userInfo: [NSLocalizedDescriptionKey: "output already exists"])
            }
            let requestedAttention = ProcessInfo.processInfo.environment["ANE_FIELDFARE_PREFILL_ATTENTION"] ?? "full-tensorops-2d-preferred"
            guard let attention = RuntimePrefillAttentionPath(rawValue: requestedAttention) else {
                throw NSError(domain: "QualityGate", code: 2, userInfo: [NSLocalizedDescriptionKey: "invalid attention path"])
            }
            let context = try MetalContext()
            let model = try Model.load(directoryURL: URL(fileURLWithPath: args[1]), device: context.device,
                streamingMode: .pread(slotCount: 16), expertCachePolicy: .lfu, integrityPolicy: .sizeCheckTrustedReceipt)
            let runtime = RuntimeConfiguration(expertCacheSlots: 16, prefillChunkTokens: chunk,
                prefillAttentionPath: attention,
                forceLogitsHead: true)
            let runner = try RealForwardRunner(model: model, context: context, maxContext: 4096, runtimeConfiguration: runtime)
            guard let logits = context.device.makeBuffer(length: model.config.vocabSize * 2, options: .storageModeShared) else {
                throw PrefillError.chunkedUnsupported("logits allocation failed")
            }
            try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            guard FileManager.default.createFile(atPath: outputURL.path, contents: nil) else {
                throw NSError(domain: "QualityGate", code: 3, userInfo: [NSLocalizedDescriptionKey: "cannot create logits output"])
            }
            let output = try FileHandle(forWritingTo: outputURL)
            defer { try? output.close() }
            let started = Date()
            _ = try await runner.prefillChunked(tokens: fixture.prompt_token_ids[...], startPosition: 0,
                outputMode: .logits, config: .production(chunkTokens: chunk), into: logits, onProgress: { _ in })
            let prefillSeconds = Date().timeIntervalSince(started)
            var finite = true
            var greedy: [Int] = []
            func appendLogits() throws {
                let values = logits.contents().assumingMemoryBound(to: Float16.self)
                var best = 0
                for i in 0..<model.config.vocabSize {
                    if !values[i].isFinite { finite = false }
                    if values[i] > values[best] { best = i }
                }
                greedy.append(best)
                try output.write(contentsOf: Data(bytes: values, count: logits.length))
            }
            try appendLogits()
            for (offset, token) in fixture.continuation_token_ids.enumerated() {
                try await runner.produce(token: token, position: fixture.prompt_token_ids.count + offset, into: logits)
                try appendLogits()
            }
            let receipt: [String: Any] = [
                "schema": "ane-fieldfare.fixed-continuation-logits.v1", "status": finite ? "accepted" : "rejected",
                "command": args, "host": ProcessInfo.processInfo.hostName,
                "os": ProcessInfo.processInfo.operatingSystemVersionString,
                "attention_path": attention.rawValue, "chunk_tokens": chunk, "prompt_token_ids": fixture.prompt_token_ids,
                "continuation_token_ids": fixture.continuation_token_ids,
                "rows": greedy.count, "vocabulary": model.config.vocabSize, "greedy_token_ids": greedy,
                "all_values_finite": finite, "prefill_seconds": prefillSeconds,
                "ane_shared_ffn": runner.aneSharedFFNProfile,
                "ane_routed_prefill": runner.aneRoutedPrefillProfile,
                "logits_path": outputURL.path,
                "claim_boundary": "Full logits under identical prompt and forced continuation inputs; independent cross-arm comparison required. No general quality claim."]
            let data = try JSONSerialization.data(withJSONObject: receipt, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: URL(fileURLWithPath: args[4] + ".json"), options: .atomic)
            print("\(greedy.count) full-vocabulary rows; finite=\(finite); prefill=\(prefillSeconds)s")
            if !finite { exit(1) }
        } catch {
            FileHandle.standardError.write(Data("error: \(error)\n".utf8)); exit(1)
        }
    }
}
