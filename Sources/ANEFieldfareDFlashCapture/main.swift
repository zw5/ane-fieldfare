import ANEFieldfareHotpath
import Foundation
import Darwin

private struct Generation: Decodable {
    let prompt_token_ids: [Int32]
    let generated_token_ids: [Int32]
}

/// Inputs for the offline DFlash acceptance gate. `capture` prefills a prompt
/// plus its greedy continuation and records the drafter's context layers and
/// every layer's routes; `embedding` writes the tied INT4 embedding table so
/// an external drafter reference shares the target's input and output head.
@main private enum DFlashCapture {
    static let contextLayers = [1, 6, 11, 17, 22, 27]

    static func main() async { await run() }

    static func fail(_ message: String, _ code: Int32 = 2) -> Never {
        FileHandle.standardError.write(Data("error: \(message)\n".utf8)); exit(code)
    }

    nonisolated static func run() async {
        let args = CommandLine.arguments
        do {
            if args.count == 4, args[1] == "embedding" {
                try dumpEmbedding(modelPath: args[2], prefix: args[3])
            } else if args.count == 5, args[1] == "capture" {
                try await capture(modelPath: args[2], generationPath: args[3], prefix: args[4])
            } else {
                fail("usage: ANEFieldfareDFlashCapture capture <model-directory> <generation.json> <output-prefix>\n"
                    + "       ANEFieldfareDFlashCapture embedding <model-directory> <output-prefix>")
            }
        } catch {
            fail("\(error)", 1)
        }
    }

    static func loadModel(_ path: String, context: MetalContext) throws -> Model {
        try Model.load(directoryURL: URL(fileURLWithPath: path), device: context.device,
                       streamingMode: .pread(slotCount: 16), expertCachePolicy: .lfu,
                       integrityPolicy: .sizeCheckTrustedReceipt)
    }

    static func write(_ pointer: UnsafeRawPointer, count: Int, to path: String) throws {
        guard !FileManager.default.fileExists(atPath: path) else { fail("output exists: \(path)", 3) }
        try Data(bytes: pointer, count: count).write(to: URL(fileURLWithPath: path))
    }

    static func dumpEmbedding(modelPath: String, prefix: String) throws {
        let context = try MetalContext()
        let model = try loadModel(modelPath, context: context)
        let view = model.embedding
        let base = UnsafeRawPointer(view.buffer.contents())
        try write(base + Int(view.offset), count: Int(view.length), to: prefix + ".q4")
        try write(base + Int(view.scaleOffset), count: Int(view.scaleLength), to: prefix + ".scales.bf16")
        try write(base + Int(view.biasOffset), count: Int(view.biasLength), to: prefix + ".biases.bf16")
        print("embedding vocab=\(model.config.vocabSize) d=\(model.config.hiddenSize) bytes=\(view.length)")
    }

    static func capture(modelPath: String, generationPath: String, prefix: String) async throws {
        let generation = try JSONDecoder().decode(
            Generation.self, from: Data(contentsOf: URL(fileURLWithPath: generationPath)))
        let tokens = generation.prompt_token_ids + generation.generated_token_ids
        let chunk = 1024
        guard !generation.generated_token_ids.isEmpty, tokens.count < 4096 else { fail("invalid generation") }

        let context = try MetalContext()
        let model = try loadModel(modelPath, context: context)
        let cfg = model.config
        let D = cfg.hiddenSize, layers = cfg.numLayers, topK = cfg.topKExperts
        let runner = try RealForwardRunner(
            model: model, context: context, maxContext: 4096,
            runtimeConfiguration: RuntimeConfiguration(expertCacheSlots: 16, prefillChunkTokens: chunk,
                                                       forceLogitsHead: true))
        var hidden = [Float16](repeating: 0, count: tokens.count * contextLayers.count * D)
        var routes = [UInt8](repeating: 0, count: tokens.count * layers * topK)
        var chunks: [[Int]] = []
        runner.prefillCapture = PrefillCapture(hiddenLayers: contextLayers) { c in
            chunks.append([c.startPosition, c.tokenCount])
            for slot in contextLayers.indices {
                for row in 0..<c.tokenCount {
                    let source = (slot * c.tokenCount + row) * D
                    let target = ((c.startPosition + row) * contextLayers.count + slot) * D
                    for i in 0..<D { hidden[target + i] = c.hidden[source + i] }
                }
            }
            for layer in 0..<layers {
                for row in 0..<c.tokenCount {
                    for k in 0..<topK {
                        routes[((c.startPosition + row) * layers + layer) * topK + k] =
                            UInt8(c.routes[layer][row * topK + k])
                    }
                }
            }
        }
        guard let logits = context.device.makeBuffer(length: cfg.vocabSize * 2, options: .storageModeShared) else {
            fail("logits allocation failed", 1)
        }
        let started = Date()
        _ = try await runner.prefillChunked(tokens: tokens[...], startPosition: 0, outputMode: .logits,
                                            config: .production(chunkTokens: chunk), into: logits,
                                            onProgress: { _ in })
        let seconds = Date().timeIntervalSince(started)
        try hidden.withUnsafeBytes { try write($0.baseAddress!, count: $0.count, to: prefix + ".hidden.f16") }
        try routes.withUnsafeBytes { try write($0.baseAddress!, count: $0.count, to: prefix + ".routes.u8") }
        let receipt: [String: Any] = [
            "schema": "ane-fieldfare.dflash-capture.v1", "status": "accepted",
            "command": args(), "host": ProcessInfo.processInfo.hostName,
            "os": ProcessInfo.processInfo.operatingSystemVersionString,
            "positions": tokens.count, "prompt_tokens": generation.prompt_token_ids.count,
            "context_layers": contextLayers, "hidden_layout": "[position][context_layer][hidden] FP16",
            "routes_layout": "[position][layer][top_k] UInt8", "chunks": chunks, "prefill_seconds": seconds,
            "claim_boundary": "Prefill-path residual stream and routes for a fixed token sequence; no drafter or acceptance claim.",
        ]
        try JSONSerialization.data(withJSONObject: receipt, options: [.prettyPrinted, .sortedKeys])
            .write(to: URL(fileURLWithPath: prefix + ".json"))
        print("captured \(tokens.count) positions in \(chunks.count) chunks; prefill=\(seconds)s")
    }

    static func args() -> [String] { CommandLine.arguments }
}
