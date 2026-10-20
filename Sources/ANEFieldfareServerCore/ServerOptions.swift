public import Foundation
public import ANEFieldfareHotpath
import ANEFieldfareHotpathCLICore

public struct ServerOptions: Sendable, Equatable {
    public var model: String
    public var drafter: String?
    public var host: String = "127.0.0.1"
    public var port: UInt16 = 8080
    public var maxContext: Int = 32_768
    public var modelID: String = "gemma-4-26b-a4b-it"
    public var apiKey: String?
    public var maxQueue: Int = 8
    public var defaults = SamplingDefaults()
    public var backend: RuntimeBackend = .hybridANE
    public var faultPolicy: RuntimeFaultPolicy = RuntimeConfiguration().faultPolicy
    public var earlyExpertRead: RuntimeEarlyExpertRead = RuntimeConfiguration().earlyExpertRead

    public init(model: String) {
        self.model = model
    }

    public enum ParseError: Error, Equatable, CustomStringConvertible {
        case helpRequested
        case unknownFlag(String)
        case missingValue(String)
        case invalidValue(flag: String, value: String)
        case requiredMissing(String)

        public var description: String {
            switch self {
            case .helpRequested: return "help requested"
            case .unknownFlag(let flag): return "unknown flag: \(flag)"
            case .missingValue(let flag): return "missing value for \(flag)"
            case .invalidValue(let flag, let value): return "invalid value for \(flag): \(value)"
            case .requiredMissing(let flag): return "required flag missing: \(flag)"
            }
        }
    }

    public static let apiKeyEnvironmentVariable = "ANE_FIELDFARE_API_KEY"

    public static let usage = """
    ANEFieldfareServer — OpenAI-compatible HTTP API for Gemma 4 26B-A4B

    usage: ANEFieldfareServer --model <dir> [options]

    required:
      --model <dir>              Path to a .gturbo model directory.

    options:
      --drafter <dir|off>        DFlash drafter checkpoint: requests decode
                                 speculatively, greedy or sampled (not with a
                                 repetition penalty). Default: found beside the
                                 model or in ../dflash/\(DrafterLocation.directoryName).
      --host <address>           Address to bind (default 127.0.0.1).
      --port <int>               Port to bind (default 8080; 0 = any free port).
      --max-context <int>        Context limit in tokens, prompt plus
                                 completion (default 32768).
      --model-id <id>            Model id the API advertises (default
                                 gemma-4-26b-a4b-it). Requests may name any model.
      --api-key <key>            Require Authorization: Bearer <key> on /v1.
                                 Also read from \(apiKeyEnvironmentVariable).
      --max-queue <int>          Requests that may wait while one generates
                                 (default 8); more are refused with 503.
      --temperature <float>      Default temperature when a request sends none
                                 (default 0.2; 0 = greedy).
      --top-k <int>              Default top-k, 1...256 (default 64; 0 = off).
      --top-p <float>            Default top-p (default 0.95).
      --max-tokens <int>         Default completion limit when a request sends
                                 none (default: the rest of the context).
      --backend <metal|hybrid-ane>
                                 Execution backend (default hybrid-ane).
      --fault-policy <fail-closed|failover-metal>
                                 On an ANE guest fault (default fail-closed).
      --early-expert-read <off|router|fitted>
                                 Decode early expert read (default fitted).
      --help                     Show this message.
    """

    public static func parse(_ argv: [String],
                             environment: [String: String] = ProcessInfo.processInfo.environment) throws
        -> ServerOptions {
        var model: String?
        var options = ServerOptions(model: "")
        options.apiKey = environment[apiKeyEnvironmentVariable].flatMap { $0.isEmpty ? nil : $0 }
        var index = 0
        func value(_ flag: String) throws -> String {
            guard index + 1 < argv.count else { throw ParseError.missingValue(flag) }
            index += 2
            return argv[index - 1]
        }
        while index < argv.count {
            let flag = argv[index]
            switch flag {
            case "--help", "-h":
                throw ParseError.helpRequested
            case "--model":
                model = try value(flag)
            case "--drafter":
                options.drafter = try value(flag)
            case "--host":
                let host = try value(flag)
                guard !host.isEmpty else { throw ParseError.invalidValue(flag: flag, value: host) }
                options.host = host
            case "--port":
                let text = try value(flag)
                guard let port = UInt16(text) else { throw ParseError.invalidValue(flag: flag, value: text) }
                options.port = port
            case "--max-context":
                let text = try value(flag)
                guard let parsed = Int(text), parsed >= 16 else {
                    throw ParseError.invalidValue(flag: flag, value: text)
                }
                options.maxContext = parsed
            case "--model-id":
                let id = try value(flag)
                guard !id.isEmpty else { throw ParseError.invalidValue(flag: flag, value: id) }
                options.modelID = id
            case "--api-key":
                let key = try value(flag)
                guard !key.isEmpty else { throw ParseError.invalidValue(flag: flag, value: key) }
                options.apiKey = key
            case "--max-queue":
                let text = try value(flag)
                guard let parsed = Int(text), parsed >= 0 else {
                    throw ParseError.invalidValue(flag: flag, value: text)
                }
                options.maxQueue = parsed
            case "--temperature":
                let text = try value(flag)
                guard let parsed = Float(text), parsed.isFinite, parsed >= 0 else {
                    throw ParseError.invalidValue(flag: flag, value: text)
                }
                options.defaults.temperature = parsed
            case "--top-k":
                let text = try value(flag)
                guard let parsed = Int(text), (0...256).contains(parsed) else {
                    throw ParseError.invalidValue(flag: flag, value: text)
                }
                options.defaults.topK = parsed == 0 ? nil : parsed
            case "--top-p":
                let text = try value(flag)
                guard let parsed = Float(text), parsed > 0, parsed <= 1 else {
                    throw ParseError.invalidValue(flag: flag, value: text)
                }
                options.defaults.topP = parsed
            case "--max-tokens":
                let text = try value(flag)
                guard let parsed = Int(text), parsed > 0 else {
                    throw ParseError.invalidValue(flag: flag, value: text)
                }
                options.defaults.maxTokens = parsed
            case "--backend":
                let text = try value(flag)
                guard let parsed = RuntimeBackend(rawValue: text) else {
                    throw ParseError.invalidValue(flag: flag, value: text)
                }
                options.backend = parsed
            case "--fault-policy":
                let text = try value(flag)
                guard let parsed = RuntimeFaultPolicy(rawValue: text) else {
                    throw ParseError.invalidValue(flag: flag, value: text)
                }
                options.faultPolicy = parsed
            case "--early-expert-read":
                let text = try value(flag)
                guard let parsed = RuntimeEarlyExpertRead(rawValue: text) else {
                    throw ParseError.invalidValue(flag: flag, value: text)
                }
                options.earlyExpertRead = parsed
            default:
                throw ParseError.unknownFlag(flag)
            }
        }
        guard let model, !model.isEmpty else { throw ParseError.requiredMissing("--model") }
        options.model = model
        let defaults = options.defaults
        if defaults.temperature > 0, defaults.topK == nil, let topP = defaults.topP, topP < 1 {
            throw ParseError.invalidValue(flag: "--top-p", value: "\(topP) requires --top-k between 1 and 256")
        }
        return options
    }
}

/// Loads the model, serves until the task is cancelled, then stops.
/// Returns the process exit code.
public func runServer(options: ServerOptions,
                      log: @escaping @Sendable (String) -> Void) async -> Int32 {
    do {
        let modelURL = URL(fileURLWithPath: options.model)
        log("[server] loading tokenizer and model from \(modelURL.path)")
        let tokenizer = try await GFTokenizer.load(forModelDirectory: modelURL)
        let loadStarted = ContinuousClock.now
        let drafterURL: URL?
        switch options.drafter {
        case "off"?: drafterURL = nil
        case let path?: drafterURL = URL(fileURLWithPath: path)
        case nil:
            drafterURL = DrafterLocation.find(model: options.model)
            if drafterURL == nil {
                log("[server] no drafter at "
                    + DrafterLocation.candidates(model: options.model).map(\.path).joined(separator: " or ")
                    + "; decoding without speculation")
            }
        }
        let engine = try await ModelEngine(
            options: ModelEngine.Options(
                modelURL: modelURL,
                drafterURL: drafterURL,
                maxContext: options.maxContext,
                backend: options.backend,
                faultPolicy: options.faultPolicy,
                earlyExpertRead: options.earlyExpertRead),
            tokenizer: tokenizer)
        let drafterNote = engine.hasDrafter ? ", drafter on" : ""
        let loaded = loadStarted.duration(to: .now).components
        let loadSeconds = (Double(loaded.seconds) + Double(loaded.attoseconds) / 1e18) * 100
        log("[server] model loaded in \(loadSeconds.rounded() / 100) s; context \(options.maxContext) tokens, "
            + "prefill chunks up to \(engine.maxPrefillChunkTokens)\(drafterNote)")
        let backend = SerializedBackend(engine: engine, tokenizer: tokenizer, maxContext: options.maxContext,
                                        maxWaiting: options.maxQueue)
        let service = OpenAIService(
            configuration: OpenAIService.Configuration(
                modelID: options.modelID, apiKey: options.apiKey, defaults: options.defaults, log: log),
            backend: backend)
        let server = try HTTPServer(
            configuration: HTTPServer.Configuration(host: options.host, port: options.port),
            handler: service.handler)
        let port = try await server.start()
        let shownHost = options.host.contains(":") ? "[\(options.host)]" : options.host
        log("[server] listening on http://\(shownHost):\(port)/v1 (model id \(options.modelID)"
            + (options.apiKey == nil ? "" : ", API key required") + ")")
        while !Task.isCancelled {
            try? await Task.sleep(for: .seconds(3600))
        }
        log("[server] stopping")
        server.stop()
        return 0
    } catch {
        log("error: \(error)")
        return 1
    }
}
