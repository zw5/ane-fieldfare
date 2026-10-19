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
            } else if args.count == 6, args[1] == "generate", let block = Int(args[5]) {
                try await generate(modelPath: args[2], drafterPath: args[3], generationPath: args[4], block: block)
            } else if args.count == 6, args[1] == "draft-check" {
                try await draftCheck(modelPath: args[2], drafterPath: args[3], generationPath: args[4],
                                     anchors: args[5].split(separator: ",").compactMap { Int($0) })
            } else if args.count == 2, args[1] == "bench-gemv" {
                let context = try MetalContext()
                for (m, n) in [(4096, 2816), (2048, 2816), (2816, 4096), (2112, 2816), (2816, 2112), (8192, 2816)] {
                    print(try KernelBench.coldGEMV(context: context, m: m, n: n))
                }
            } else if args.count == 4, args[1] == "verify-check" {
                try await verifyCheck(modelPath: args[2], generationPath: args[3])
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
            runtimeConfiguration: RuntimeConfiguration(prefillChunkTokens: chunk,
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

    /// Teacher-forces a greedy generation through `verify` in full blocks and
    /// counts rows whose greedy token disagrees with the recorded next token.
    /// With `rewind`, a junk block is verified and rewound before each real one.
    static func verifyCheck(modelPath: String, generationPath: String) async throws {
        let generation = try JSONDecoder().decode(
            Generation.self, from: Data(contentsOf: URL(fileURLWithPath: generationPath)))
        let context = try MetalContext()
        let model = try loadModel(modelPath, context: context)
        let block = RealForwardRunner.maxVerifyTokens
        for useRewind in [false] {
            let runner = try RealForwardRunner(
                model: model, context: context, maxContext: 4096,
                runtimeConfiguration: RuntimeConfiguration(prefillChunkTokens: 1024))
            guard let logits = context.device.makeBuffer(length: model.config.vocabSize * 2,
                                                         options: .storageModeShared) else { fail("alloc", 1) }
            let prompt = generation.prompt_token_ids
            let result = try await runner.prefillChunked(tokens: prompt[...], startPosition: 0,
                outputMode: .greedyIfAvailable, config: .production(chunkTokens: 1024), into: logits,
                onProgress: { _ in })
            let gen = generation.generated_token_ids
            var mismatches = 0, rows = 0, firstMismatch = -1
            if case .greedyToken(let t) = result.seed, Int32(t) != gen[0] { mismatches += 1; firstMismatch = 0 }
            var position = prompt.count
            var i = 0
            print("  after prompt: tile_fetch=\(String(format: "%.2f", Double(runner.prefillTileFetchNanos) / 1e9)) "
                + "tile_hits=\(runner.prefillTileHits) tile_misses=\(runner.prefillTileMisses) "
                + "routed_wall=\(String(format: "%.2f", Double(runner.prefillRoutedWallNanos) / 1e9))")
            let started = Date()
            while i + 1 < gen.count {
                let n = min(block, gen.count - 1 - i)
                if useRewind {
                    _ = try await runner.verify(tokens: Array(repeating: 236761, count: block),
                                                startPosition: position)
                    try runner.rewind(to: position)
                }
                let out = try await runner.verify(tokens: Array(gen[i..<(i + n)]), startPosition: position)
                for r in 0..<n where out[r] != gen[i + r + 1] {
                    mismatches += 1
                    if firstMismatch < 0 { firstMismatch = i + r + 1 }
                }
                rows += n; i += n; position += n
            }
            let seconds = Date().timeIntervalSince(started)
            func sec(_ n: UInt64) -> String { String(format: "%.2f", Double(n) / 1e9) }
            print("  io=\(sec(runner.totalIoNanos)) routed_wall=\(sec(runner.prefillRoutedWallNanos)) "
                + "pre_router_gpu=\(sec(runner.prefillPreRouterGPUNanos)) shared_gpu=\(sec(runner.prefillSharedGPUNanos)) "
                + "routed_gpu=\(sec(runner.prefillRoutedGPUNanos)) hits=\(runner.routedExpertCacheHits) "
                + "misses=\(runner.routedExpertCacheMisses) union=\(runner.prefillUnionExpertsBeforeCap) "
                + "tile_fetch=\(sec(runner.prefillTileFetchNanos)) tile_drain=\(sec(runner.prefillTileDrainNanos)) "
                + "tile_hits=\(runner.prefillTileHits) tile_misses=\(runner.prefillTileMisses)")
            print("rewind=\(useRewind) rows=\(rows) mismatches=\(mismatches) first=\(firstMismatch) "
                + "verify_seconds=\(String(format: "%.2f", seconds)) per_block_ms="
                + String(format: "%.1f", seconds * 1000 / Double((rows + block - 1) / block) / (useRewind ? 2 : 1)))
        }
    }

    /// Speculative greedy generation of as many tokens as the recorded plain
    /// greedy generation, reporting agreement and timing.
    static func generate(modelPath: String, drafterPath: String, generationPath: String, block: Int) async throws {
        let generation = try JSONDecoder().decode(
            Generation.self, from: Data(contentsOf: URL(fileURLWithPath: generationPath)))
        let context = try MetalContext()
        let model = try loadModel(modelPath, context: context)
        let runner = try RealForwardRunner(
            model: model, context: context, maxContext: 4096,
            runtimeConfiguration: RuntimeConfiguration(prefillChunkTokens: 1024))
        let drafter = try DFlashDrafter(directory: URL(fileURLWithPath: drafterPath), model: model,
                                        context: context, maxContext: 4096)
        let reference = generation.generated_token_ids
        let (tokens, stats) = try await DFlashGeneration.generateGreedy(
            runner: runner, drafter: drafter, context: context, prompt: generation.prompt_token_ids,
            maxNewTokens: reference.count, blockSize: block, prefillChunkTokens: 1024)
        let firstDiff = zip(tokens, reference).enumerated().first { $0.element.0 != $0.element.1 }?.offset ?? -1
        var profile = stats.profile
        profile["block"] = block
        profile["tokens_match_plain_greedy"] = tokens == reference
        profile["first_divergence"] = firstDiff
        profile["generated_token_ids"] = tokens.map(Int.init)
        func sec(_ n: UInt64) -> Double { Double(n) / 1e9 }
        profile["runner_including_prompt"] = [
            "pre_router_gpu": sec(runner.prefillPreRouterGPUNanos), "shared_gpu": sec(runner.prefillSharedGPUNanos),
            "routed_gpu": sec(runner.prefillRoutedGPUNanos), "routed_wall": sec(runner.prefillRoutedWallNanos),
            "tile_fetch": sec(runner.prefillTileFetchNanos), "tile_drain": sec(runner.prefillTileDrainNanos),
            "tile_hits": runner.prefillTileHits, "tile_misses": runner.prefillTileMisses,
            "union_experts": runner.prefillUnionExpertsBeforeCap, "layer_tiles": runner.prefillUnionLayerTiles]
        let data = try JSONSerialization.data(withJSONObject: profile, options: [.sortedKeys])
        print(String(decoding: data, as: UTF8.self))
    }

    /// Drafts at given anchor positions over teacher-forced context, for
    /// comparison with the numpy reference (`Scripts/dflash/dflash_ref.py`).
    static func draftCheck(modelPath: String, drafterPath: String, generationPath: String, anchors: [Int]) async throws {
        let generation = try JSONDecoder().decode(
            Generation.self, from: Data(contentsOf: URL(fileURLWithPath: generationPath)))
        let tokens = generation.prompt_token_ids + generation.generated_token_ids
        let context = try MetalContext()
        let model = try loadModel(modelPath, context: context)
        let runner = try RealForwardRunner(
            model: model, context: context, maxContext: 4096,
            runtimeConfiguration: RuntimeConfiguration(prefillChunkTokens: 1024))
        let drafter = try DFlashDrafter(directory: URL(fileURLWithPath: drafterPath), model: model,
                                        context: context, maxContext: 4096)
        var failure: Error?
        runner.prefillCapture = PrefillCapture(hiddenLayers: drafter.config.contextLayers) { chunk in
            do { try drafter.appendContext(captured: chunk.hidden, stride: chunk.tokenCount, rows: chunk.tokenCount) }
            catch { failure = error }
        }
        guard let logits = context.device.makeBuffer(length: model.config.vocabSize * 2, options: .storageModeShared) else {
            fail("alloc", 1)
        }
        _ = try await runner.prefillChunked(tokens: tokens[...], startPosition: 0, outputMode: .logits,
                                            config: .production(chunkTokens: 1024), into: logits, onProgress: { _ in })
        if let failure { throw failure }
        var result: [String: [Int]] = [:]
        for anchor in anchors {
            try drafter.truncateContext(to: anchor)
            let started = Date()
            let drafts = try drafter.propose(anchor: tokens[anchor], count: drafter.config.blockSize)
            let ms = Date().timeIntervalSince(started) * 1000
            result[String(anchor)] = drafts.map(Int.init)
            FileHandle.standardError.write(Data("anchor \(anchor) propose_ms=\(String(format: "%.1f", ms))\n".utf8))
        }
        let data = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
        print(String(decoding: data, as: UTF8.self))
    }

    static func args() -> [String] { CommandLine.arguments }
}
