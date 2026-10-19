import ANEFieldfareHotpath
import Darwin
import Foundation
import Metal

/// The optimization loop's benchmark: one model load, a fixed suite, raw
/// samples out. `Scripts/bench/compare.py` turns one or more result files per
/// build into statistics, token agreement and an HTML report.
///
/// usage: ANEFieldfareBench --model <dir> [--drafter <dir>] [--tokens N] [--reps N]
///                          [--suite quick|full] [--label name] [--out file.json]
///
/// Cases run the CLI's own generation loop (`runRawCompletion`), greedy, with a
/// fresh KV cache per case and a fresh speculation policy per speculative case.
/// Step samples time one decode step and verifies of 1, 2, 4 and 8 rows at a
/// fixed context, rewinding after each.
@main private enum Bench {
    struct Options {
        var model = "checkpoints/gturbo/gemma4.gturbo"
        var drafter: String?
        var tokens = 192
        var reps = 2
        var suite = "quick"
        var label = "run"
        var out: String?
    }

    static let prompts: [(String, String)] = [
        ("code", "Write a Python function that parses an ISO 8601 date string without using datetime, with tests."),
        ("math", "Solve step by step: a train leaves at 3:15pm at 72 km/h, a second leaves the same station at "
            + "4:05pm at 96 km/h on the same track. When and where does the second catch the first? Show all arithmetic."),
        ("explain", "Explain how a hash map works, including collisions and resizing."),
        ("chat", "I'm planning a weekend trip to Lisbon. What should I see and eat?"),
        ("essay", "Write a detailed essay on the history of the printing press and its effects on European politics."),
    ]

    static func main() async {
        do {
            try await run(parse(Array(CommandLine.arguments.dropFirst())))
        } catch {
            FileHandle.standardError.write(Data("error: \(error)\n".utf8))
            exit(1)
        }
    }

    static func parse(_ argv: [String]) throws -> Options {
        var options = Options()
        var i = 0
        func value() throws -> String {
            guard i + 1 < argv.count else { throw BenchError("missing value for \(argv[i])") }
            i += 2
            return argv[i - 1]
        }
        while i < argv.count {
            switch argv[i] {
            case "--model": options.model = try value()
            case "--drafter": options.drafter = try value()
            case "--tokens": options.tokens = Int(try value()) ?? options.tokens
            case "--reps": options.reps = Int(try value()) ?? options.reps
            case "--suite": options.suite = try value()
            case "--label": options.label = try value()
            case "--out": options.out = try value()
            default: throw BenchError("unknown flag \(argv[i])")
            }
        }
        return options
    }

    struct BenchError: Error, CustomStringConvertible {
        let description: String
        init(_ description: String) { self.description = description }
    }

    static func log(_ line: String) { FileHandle.standardError.write(Data((line + "\n").utf8)) }

    static func run(_ options: Options) async throws {
        let started = Date()
        let modelURL = URL(fileURLWithPath: options.model)
        let context = try MetalContext()
        let model = try Model.load(directoryURL: modelURL, device: context.device,
                                   integrityPolicy: .sizeCheckTrustedReceipt)
        let tokenizer = try await GFTokenizer.load(forModelDirectory: modelURL)
        let maxContext = 4096
        let runner = try RealForwardRunner(model: model, context: context, maxContext: maxContext,
                                           runtimeConfiguration: RuntimeConfiguration())
        let drafter = try options.drafter.map {
            try DFlashDrafter(directory: URL(fileURLWithPath: $0), model: model, context: context,
                              maxContext: maxContext)
        }
        let scratch = try RawCompletionScratch(context: context, vocab: model.config.vocabSize)
        let promptSet = options.suite == "full" ? prompts : Array(prompts.prefix(3))
        let modes = drafter == nil ? ["plain"] : ["plain", "spec"]
        log(unsafe "[bench \(options.label)] load \(String(format: "%.1f", Date().timeIntervalSince(started)))s; "
            + "\(promptSet.count) prompts x \(modes.count) modes x \(options.reps) reps, \(options.tokens) tokens")

        // Plain decode guesses its head from the hot vocabulary unless
        // ANE_FIELDFARE_HOT_HEAD=0; output is the same either way.
        let useHotHead = ProcessInfo.processInfo.environment["ANE_FIELDFARE_HOT_HEAD"] != "0"
        // Warm-up: compiles pipelines and fills the page cache the same way for
        // both builds before anything is measured.
        _ = try await generate(prompt: promptSet[0].1, tokens: 48,
                               producer: (useHotHead ? HotHeadDecoder(runner: runner) : nil) ?? runner,
                               tokenizer: tokenizer,
                               context: context, scratch: scratch)
        if let drafter {
            _ = try await generate(prompt: promptSet[0].1, tokens: 48,
                                   producer: try SpeculativeDecoder(runner: runner, drafter: drafter),
                                   tokenizer: tokenizer, context: context, scratch: scratch)
        }

        var cases: [[String: Any]] = []
        var stepTokens: [Int32] = []
        var stepPrompt: [Int32] = []
        for rep in 0..<options.reps {
            for (name, text) in promptSet {
                // Alternate mode order per rep so drift does not favour one mode.
                for mode in rep % 2 == 0 ? modes : modes.reversed() {
                    let before = Counters(runner)
                    let decoder = try drafter.flatMap { drafter in
                        try mode == "spec" ? SpeculativeDecoder(runner: runner, drafter: drafter) : nil
                    }
                    let hot = useHotHead ? HotHeadDecoder(runner: runner) : nil
                    let plain: any LogitProducer = hot ?? runner
                    let result = try await generate(prompt: text, tokens: options.tokens,
                                                    producer: decoder ?? plain, tokenizer: tokenizer,
                                                    context: context, scratch: scratch)
                    let delta = Counters(runner).minus(before)
                    let decodeTPS = result.result.decodeSeconds > 0
                        ? Double(result.result.newTokens) / result.result.decodeSeconds : 0
                    var record: [String: Any] = [
                        "prompt": name, "mode": mode, "rep": rep,
                        "prefill_tokens": result.result.prefillTokens,
                        "prefill_seconds": result.result.prefillSeconds,
                        "new_tokens": result.result.newTokens,
                        "decode_seconds": result.result.decodeSeconds,
                        "decode_tps": decodeTPS,
                        "generated_ids": result.ids.map(Int.init),
                        "counters": delta.json,
                    ]
                    if let hot {
                        record["hot_head"] = ["guessed_steps": hot.guessedSteps, "wrong_guesses": hot.wrongGuesses]
                    }
                    if let decoder {
                        var spec = decoder.stats.profile
                        spec["acceptance_rate"] = decoder.policy.acceptanceRate
                        spec["verify_seconds_by_rows"] = decoder.policy.verifySeconds
                        spec["draft_seconds_by_length"] = decoder.policy.draftSeconds
                        record["speculation"] = spec
                    }
                    if name == promptSet[0].0, mode == "plain", stepTokens.isEmpty {
                        stepTokens = result.ids
                        stepPrompt = result.prompt
                    }
                    cases.append(record)
                    unsafe log(String(format: "  %-8@ %-5@ rep %d  %6.2f tok/s  %d tokens", name as NSString,
                               mode as NSString, rep, decodeTPS, result.result.newTokens))
                }
            }
        }

        let steps = try await stepSamples(runner: runner, prompt: stepPrompt, generated: stepTokens)
        let report: [String: Any] = [
            "schema": "ane-fieldfare.bench.v1",
            "label": options.label,
            "environment": environment(),
            "options": ["tokens": options.tokens, "reps": options.reps, "suite": options.suite,
                        "drafter": options.drafter as Any? ?? NSNull(), "max_context": maxContext],
            "cases": cases,
            "steps": steps,
            "wall_seconds": Date().timeIntervalSince(started),
        ]
        let data = try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])
        if let out = options.out {
            try data.write(to: URL(fileURLWithPath: out))
            log(unsafe "[bench \(options.label)] wrote \(out) in \(String(format: "%.0f", Date().timeIntervalSince(started)))s")
        } else {
            FileHandle.standardOutput.write(data)
        }
    }

    struct Generation {
        let result: RawDecodeResult
        let ids: [Int32]
        let prompt: [Int32]
    }

    static func generate(prompt: String, tokens: Int, producer: any LogitProducer, tokenizer: GFTokenizer,
                         context: MetalContext, scratch: RawCompletionScratch) async throws -> Generation {
        let ids = tokenizer.encode(try tokenizer.applyChatTemplate([.init(role: .user, content: prompt)]),
                                   addBOS: false)
        var generated: [Int32] = []
        let result = try await runRawCompletion(
            producer: producer, tokenizer: tokenizer, promptIds: ids,
            config: GenerationConfig(maxNewTokens: tokens, temperature: 0, topK: nil, topP: nil),
            context: context, scratch: scratch,
            prefillConfig: .production(chunkTokens: PrefillRuntimeConfig.autoChunkTokens(promptTokens: ids.count))
        ) { progress in
            if case .token(_, let id, _) = progress { generated.append(id) }
        }
        return Generation(result: result, ids: generated, prompt: ids)
    }

    /// Cumulative runner counters, differenced per case.
    struct Counters {
        var values: [String: Double]

        init(_ runner: RealForwardRunner) {
            func s(_ n: UInt64) -> Double { Double(n) / 1e9 }
            values = [
                "decode_attention_gpu_s": s(runner.decodeAttentionGPUNanos),
                "decode_shared_gpu_s": s(runner.decodeSharedGPUNanos),
                "decode_routed_gpu_s": s(runner.decodeRoutedGPUNanos),
                "decode_io_s": s(runner.totalIoNanos),
                "expert_hits": Double(runner.routedExpertCacheHits),
                "expert_misses": Double(runner.routedExpertCacheMisses),
                "verify_pre_router_gpu_s": s(runner.prefillPreRouterGPUNanos),
                "verify_shared_gpu_s": s(runner.prefillSharedGPUNanos),
                "verify_routed_gpu_s": s(runner.prefillRoutedGPUNanos),
                "verify_fetch_wait_s": s(runner.prefillTileFetchNanos),
                "verify_expert_hits": Double(runner.prefillTileHits),
                "verify_expert_misses": Double(runner.prefillTileMisses),
                "verify_early_reads": Double(runner.prefillEarlyReads),
                "early_reads": Double(runner.earlyExpertReadStats?.reads ?? 0),
                "early_reads_used": Double(runner.earlyExpertReadStats?.used ?? 0),
            ]
        }

        func minus(_ other: Counters) -> Counters {
            var delta = self
            for (key, value) in values { delta.values[key] = value - (other.values[key] ?? 0) }
            return delta
        }

        var json: [String: Double] { values }
    }

    /// Medians need many samples, so steps rewind to the same context and
    /// slide the token window, as generation would vary routes.
    static func stepSamples(runner: RealForwardRunner, prompt: [Int32], generated: [Int32]) async throws
        -> [[String: Any]] {
        guard generated.count >= 96 else { return [] }
        guard let device = MTLCreateSystemDefaultDevice(),
              let logits = device.makeBuffer(length: 262_144 * 2, options: .storageModeShared) else { return [] }
        let warm = 48
        let prefix = prompt + generated[0..<warm]
        runner.reset()
        _ = try await runner.prefillChunked(
            tokens: prefix[...], startPosition: 0, outputMode: .greedyIfAvailable,
            config: .production(chunkTokens: PrefillRuntimeConfig.autoChunkTokens(promptTokens: prefix.count)),
            into: logits, onProgress: { _ in })
        let position = prefix.count
        var out: [[String: Any]] = []
        for rows in [0, 1, 2, 4, 8] {
            var samples: [Double] = []
            for i in 0..<40 {
                let start = warm + (i * 3) % (generated.count - warm - 9)
                let t0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                if rows == 0 {
                    try await runner.produce(token: generated[start], position: position, into: logits)
                } else {
                    _ = try await runner.verify(tokens: Array(generated[start..<(start + rows)]),
                                                startPosition: position)
                }
                samples.append(Double(clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - t0) / 1e6)
                try runner.rewind(to: position)
            }
            out.append(["rows": rows, "ms": Array(samples.dropFirst(6))])
        }
        return out
    }

    static func environment() -> [String: Any] {
        var size = 0
        unsafe sysctlbyname("hw.model", nil, &size, nil, 0)
        var model = [CChar](repeating: 0, count: max(size, 1))
        unsafe sysctlbyname("hw.model", &model, &size, nil, 0)
        let env = ProcessInfo.processInfo.environment
        return [
            "host": ProcessInfo.processInfo.hostName,
            "hardware": String(decoding: model.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self),
            "os": ProcessInfo.processInfo.operatingSystemVersionString,
            "memory_bytes": ProcessInfo.processInfo.physicalMemory,
            "binary": CommandLine.arguments[0],
            "git": env["ANE_FIELDFARE_BENCH_GIT"] ?? "unknown",
            "date": ISO8601DateFormatter().string(from: Date()),
        ]
    }
}
