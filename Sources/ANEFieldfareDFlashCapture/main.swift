import ANEFieldfareHotpath
import Foundation
import Darwin
import Metal

/// `x` with `digits` fractional digits, in the POSIX locale, without C varargs.
func fixed(_ x: Double, _ digits: Int) -> String {
    x.formatted(.number.precision(.fractionLength(digits)).grouping(.never)
        .locale(Locale(identifier: "en_US_POSIX")))
}

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
            } else if (4...5).contains(args.count), args[1] == "verify-check" {
                try await verifyCheck(modelPath: args[2], generationPath: args[3],
                                      block: args.count == 5 ? Int(args[4]) ?? 0 : RealForwardRunner.maxVerifyTokens)
            } else if args.count == 5, args[1] == "capture" {
                try await capture(modelPath: args[2], generationPath: args[3], prefix: args[4])
            } else {
                fail("usage: ANEFieldfareDFlashCapture capture <model-directory> <generation.json> <output-prefix>\n"
                    + "       ANEFieldfareDFlashCapture embedding <model-directory> <output-prefix>\n"
                    + "       ANEFieldfareDFlashCapture verify-check <model-directory> <generation.json> [rows]\n"
                    + "       ANEFieldfareDFlashCapture draft-check <model-directory> <drafter> <generation.json> <anchors>\n"
                    + "       ANEFieldfareDFlashCapture generate <model-directory> <drafter> <generation.json> <block>")
            }
            if let path = try GPUTrace.finish() { FileHandle.standardError.write(Data("[gpu trace] wrote \(path)\n".utf8)) }
        } catch {
            fail("\(error)", 1)
        }
    }

    static func loadModel(_ path: String, context: MetalContext) throws -> Model {
        try Model.load(directoryURL: URL(fileURLWithPath: path), device: context.device)
    }

    /// `ANE_FIELDFARE_VERIFY_SAMPLE=temperature,topK,topP,seed` checks a
    /// sampled generation drawn under that rule instead of a greedy one.
    static func verifySampling() -> TokenSampling? {
        guard let text = ProcessInfo.processInfo.environment["ANE_FIELDFARE_VERIFY_SAMPLE"] else { return nil }
        let fields = text.split(separator: ",", omittingEmptySubsequences: false)
        guard fields.count == 4, let temperature = Float(fields[0]), let topK = Int(fields[1]),
              let topP = Float(fields[2]), let seed = UInt64(fields[3]) else {
            fail("ANE_FIELDFARE_VERIFY_SAMPLE must be temperature,topK,topP,seed: \(text)")
        }
        return TokenSampling(temperature: temperature, topK: topK, topP: topP, seed: seed)
    }

    static func write(_ bytes: UnsafeRawBufferPointer, to path: String) throws {
        guard !FileManager.default.fileExists(atPath: path) else { fail("output exists: \(path)", 3) }
        unsafe try Data(bytes).write(to: URL(fileURLWithPath: path))
    }

    static func dumpEmbedding(modelPath: String, prefix: String) throws {
        let context = try MetalContext()
        let model = try loadModel(modelPath, context: context)
        let view = model.embedding
        let base = unsafe UnsafeRawPointer(view.buffer.contents())
        func bytes(at offset: UInt64, count: UInt64) -> UnsafeRawBufferPointer {
            unsafe UnsafeRawBufferPointer(start: base + Int(offset), count: Int(count))
        }
        unsafe try write(bytes(at: view.offset, count: view.length), to: prefix + ".q4")
        unsafe try write(bytes(at: view.scaleOffset, count: view.scaleLength), to: prefix + ".scales.bf16")
        unsafe try write(bytes(at: view.biasOffset, count: view.biasLength), to: prefix + ".biases.bf16")
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
            runtimeConfiguration: RuntimeConfiguration(prefillChunkTokens: chunk, headPath: .logits))
        var hidden = [Float16](repeating: 0, count: tokens.count * contextLayers.count * D)
        var routes = [UInt8](repeating: 0, count: tokens.count * layers * topK)
        var chunks: [[Int]] = []
        runner.prefillCapture = PrefillCapture(hiddenLayers: contextLayers) { c in
            chunks.append([c.startPosition, c.tokenCount])
            c.withHidden { captured in
                for slot in contextLayers.indices {
                    for row in 0..<c.tokenCount {
                        let source = (slot * c.tokenCount + row) * D
                        let target = ((c.startPosition + row) * contextLayers.count + slot) * D
                        for i in 0..<D { hidden[target + i] = captured[source + i] }
                    }
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
        try hidden.withUnsafeBytes { unsafe try write($0, to: prefix + ".hidden.f16") }
        try routes.withUnsafeBytes { unsafe try write($0, to: prefix + ".routes.u8") }
        let receipt: [String: Any] = [
            "schema": "ane-fieldfare.dflash-capture.v1", "status": "accepted",
            "command": CommandLine.arguments, "host": ProcessInfo.processInfo.hostName,
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
    static func verifyCheck(modelPath: String, generationPath: String, block: Int) async throws {
        let generation = try JSONDecoder().decode(
            Generation.self, from: Data(contentsOf: URL(fileURLWithPath: generationPath)))
        let context = try MetalContext()
        let model = try loadModel(modelPath, context: context)
        guard (1...RealForwardRunner.maxVerifyTokens).contains(block) else { fail("block \(block)") }
        let runner = try RealForwardRunner(
            model: model, context: context, maxContext: 4096,
            runtimeConfiguration: RuntimeConfiguration(prefillChunkTokens: 1024))
        runner.tokenSampling = verifySampling()
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
        let started = Date()
        while i + 1 < gen.count {
            let n = min(block, gen.count - 1 - i)
            let out = try await runner.verify(tokens: Array(gen[i..<(i + n)]), startPosition: position)
            for r in 0..<n where out[r] != gen[i + r + 1] {
                mismatches += 1
                if firstMismatch < 0 { firstMismatch = i + r + 1 }
            }
            rows += n; i += n; position += n
        }
        let seconds = Date().timeIntervalSince(started)
        print("rows=\(rows) mismatches=\(mismatches) first=\(firstMismatch) "
            + "verify_seconds=\(fixed(seconds, 2)) per_block_ms="
            + fixed(seconds * 1000 / Double((rows + block - 1) / block), 1))
    }

    /// Speculative greedy generation of as many tokens as the recorded plain
    /// greedy generation, with blocks of at most `block` rows, reporting
    /// agreement and timing.
    static func generate(modelPath: String, drafterPath: String, generationPath: String, block: Int) async throws {
        let generation = try JSONDecoder().decode(
            Generation.self, from: Data(contentsOf: URL(fileURLWithPath: generationPath)))
        let context = try MetalContext()
        let model = try loadModel(modelPath, context: context)
        let runner = try RealForwardRunner(
            model: model, context: context, maxContext: 4096,
            runtimeConfiguration: RuntimeConfiguration(prefillChunkTokens: 1024))
        let drafter = try await DFlashDrafter(directory: URL(fileURLWithPath: drafterPath), model: model,
                                        context: context, maxContext: 4096)
        let decoder = try SpeculativeDecoder(runner: runner, drafter: drafter, maxBlock: block)
        decoder.tokenSampling = verifySampling()
        guard let logits = context.device.makeBuffer(length: model.config.vocabSize * 2,
                                                     options: .storageModeShared) else { fail("alloc", 1) }
        let prompt = generation.prompt_token_ids, reference = generation.generated_token_ids
        let prefillStart = Date()
        let prefill = try await decoder.prefillChunked(
            tokens: prompt[...], startPosition: 0, outputMode: .greedyIfAvailable,
            config: .production(chunkTokens: 1024), into: logits, onProgress: { _ in })
        let prefillSeconds = Date().timeIntervalSince(prefillStart)
        guard case .greedyToken(let first) = prefill.seed else { fail("prefill gave no greedy token", 1) }
        var token = Int32(bitPattern: first)
        var tokens = [token]
        let decodeStart = Date()
        while tokens.count < reference.count {
            try await decoder.produce(token: token, position: prompt.count + tokens.count - 1, into: logits)
            token = Int32(bitPattern: decoder.lastGreedyToken)
            tokens.append(token)
        }
        let decodeSeconds = Date().timeIntervalSince(decodeStart)
        let firstDiff = zip(tokens, reference).enumerated().first { $0.element.0 != $0.element.1 }?.offset ?? -1
        var profile = decoder.stats.profile
        profile["block"] = block
        profile["prefill_seconds"] = prefillSeconds
        profile["decode_seconds"] = decodeSeconds
        profile["decode_tokens_per_second"] = Double(tokens.count - 1) / decodeSeconds
        profile["tokens_match_plain_greedy"] = tokens == reference
        profile["first_divergence"] = firstDiff
        profile["generated_token_ids"] = tokens.map(Int.init)
        let data = try JSONSerialization.data(withJSONObject: profile, options: [.sortedKeys])
        print(String(decoding: data, as: UTF8.self))
    }

    /// Drafts at given anchor positions (descending) over teacher-forced
    /// context, for comparison with the numpy reference
    /// (`Scripts/dflash/dflash_ref.py`).
    static func draftCheck(modelPath: String, drafterPath: String, generationPath: String, anchors: [Int]) async throws {
        let generation = try JSONDecoder().decode(
            Generation.self, from: Data(contentsOf: URL(fileURLWithPath: generationPath)))
        let tokens = generation.prompt_token_ids + generation.generated_token_ids
        let context = try MetalContext()
        let model = try loadModel(modelPath, context: context)
        let runner = try RealForwardRunner(
            model: model, context: context, maxContext: 4096,
            runtimeConfiguration: RuntimeConfiguration(prefillChunkTokens: 1024))
        let drafter = try await DFlashDrafter(directory: URL(fileURLWithPath: drafterPath), model: model,
                                        context: context, maxContext: 4096)
        let decoder = try SpeculativeDecoder(runner: runner, drafter: drafter)
        guard let logits = context.device.makeBuffer(length: model.config.vocabSize * 2, options: .storageModeShared) else {
            fail("alloc", 1)
        }
        // Prefill puts every position's target features in the drafter's
        // context.
        _ = try await decoder.prefillChunked(tokens: tokens[...], startPosition: 0, outputMode: .logits,
                                             config: .production(chunkTokens: 1024), into: logits,
                                             onProgress: { _ in })
        var result: [String: [Int]] = [:]
        for anchor in anchors {
            try drafter.truncateContext(to: anchor)
            let started = Date()
            let drafts = try drafter.propose(anchor: tokens[anchor], count: drafter.config.blockSize)
            let ms = Date().timeIntervalSince(started) * 1000
            result[String(anchor)] = drafts.map(Int.init)
            FileHandle.standardError.write(Data("anchor \(anchor) propose_ms=\(fixed(ms, 1))\n".utf8))
        }
        let data = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
        print(String(decoding: data, as: UTF8.self))
    }

}
