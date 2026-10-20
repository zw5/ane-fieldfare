public import Foundation
public import ANEFieldfareHotpath
import ANEFieldfareHotpathCLICore

public struct ServerOptions: Sendable, Equatable {
    public var model: String
    public var drafter: String?
    public var host: String = "127.0.0.1"
    public var port: UInt16 = 8080
    public var maxContext: Int = OptionDefaults.maxContext
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
        var cursor = ArgumentCursor(argv)
        while let flag = cursor.nextFlag() {
            switch flag {
            case "--help", "-h":
                throw ArgsError.helpRequested
            case "--model":
                model = try cursor.value(for: flag)
            case "--drafter":
                options.drafter = try cursor.value(for: flag)
            case "--host":
                options.host = try cursor.value(for: flag, as: String.self) { !$0.isEmpty }
            case "--port":
                options.port = try cursor.value(for: flag, as: UInt16.self)
            case "--max-context":
                options.maxContext = try cursor.maxContext(for: flag)
            case "--model-id":
                options.modelID = try cursor.value(for: flag, as: String.self) { !$0.isEmpty }
            case "--api-key":
                options.apiKey = try cursor.value(for: flag, as: String.self) { !$0.isEmpty }
            case "--max-queue":
                options.maxQueue = try cursor.value(for: flag, as: Int.self) { $0 >= 0 }
            case "--temperature":
                options.defaults.temperature = try cursor.temperature(for: flag)
            case "--top-k":
                options.defaults.topK = try cursor.topK(for: flag)
            case "--top-p":
                options.defaults.topP = try cursor.topP(for: flag)
            case "--max-tokens":
                options.defaults.maxTokens = try cursor.value(for: flag, as: Int.self) { $0 > 0 }
            case "--backend":
                options.backend = try cursor.choice(for: flag, as: RuntimeBackend.self)
            case "--fault-policy":
                options.faultPolicy = try cursor.choice(for: flag, as: RuntimeFaultPolicy.self)
            case "--early-expert-read":
                options.earlyExpertRead = try cursor.choice(for: flag, as: RuntimeEarlyExpertRead.self)
            default:
                throw ArgsError.unknownFlag(flag)
            }
        }
        guard let model, !model.isEmpty else { throw ArgsError.requiredMissing("--model") }
        options.model = model
        try validateTruncation(temperature: options.defaults.temperature, topK: options.defaults.topK,
                               topP: options.defaults.topP)
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
