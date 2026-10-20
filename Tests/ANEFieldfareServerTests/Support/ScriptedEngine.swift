import Foundation
import Metal
import Synchronization
import ANEFieldfareHotpath
import ANEFieldfareServerCore
import ANEFieldfareValidationSupport

/// Counts what the scripted engine did, readable from any task.
final class EngineProbe: Sendable {
    struct Counts: Sendable, Equatable {
        var started = 0
        var cancelled = 0
    }

    private let state = Mutex(Counts())

    var counts: Counts { state.withLock { $0 } }
    func record(_ update: (inout Counts) -> Void) { state.withLock { update(&$0) } }
}

/// Delays each decode step, so a test can disconnect mid-generation.
private final class PacedProducer: LogitProducer {
    let inner: ScriptedLogitProducer
    let delay: Duration
    let firstDecodePosition: Int

    init(inner: ScriptedLogitProducer, delay: Duration, firstDecodePosition: Int) {
        self.inner = inner
        self.delay = delay
        self.firstDecodePosition = firstDecodePosition
    }

    func reset() { inner.reset() }

    func produce(token: Int32, position: Int, into logits: any MTLBuffer) async throws {
        if delay > .zero, position >= firstDecodePosition { try await Task.sleep(for: delay) }
        try await inner.produce(token: token, position: position, into: logits)
    }
}

/// A `CompletionEngine` that runs the real `runRawCompletion` loop (sampler,
/// detokenizer, stop strings, cancellation) over scripted logits: after the
/// prompt it emits `reply`, then end of turn, or `reply` forever.
final class ScriptedEngine: CompletionEngine {
    let context: MetalContext
    let scratch: RawCompletionScratch
    let tokenizer: GFTokenizer
    let reply: [Int32]
    let repeats: Bool
    let delay: Duration
    let probe: EngineProbe

    init(tokenizer: GFTokenizer, reply: [Int32], repeats: Bool = false, delay: Duration = .zero,
         probe: EngineProbe) throws {
        let context = try MetalContext()
        self.context = context
        self.scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)
        self.tokenizer = tokenizer
        self.reply = reply
        self.repeats = repeats
        self.delay = delay
        self.probe = probe
    }

    func complete(promptIDs: [Int32], config: GenerationConfig,
                  onProgress: (RawDecodeProgress) -> Void) async throws -> EngineCompletion {
        probe.record { $0.started += 1 }
        let lastPrompt = promptIDs.count - 1
        let reply = reply
        let repeats = repeats
        let end = tokenizer.endOfTurnID
        let inner = ScriptedLogitProducer(vocabSize: tokenizer.vocabSize) { _, call in
            let index = call - lastPrompt
            guard index >= 0 else { return .argmax(0) }
            if repeats { return .argmax(reply[index % reply.count]) }
            return .argmax(index < reply.count ? reply[index] : end)
        }
        let producer = PacedProducer(inner: inner, delay: delay, firstDecodePosition: promptIDs.count)
        do {
            let result = try await runRawCompletion(
                producer: producer, tokenizer: tokenizer, promptIds: promptIDs, config: config,
                context: context, scratch: scratch, prefillConfig: .off, onProgress: onProgress)
            return EngineCompletion(result: result, decodePath: "scripted")
        } catch {
            if error is CancellationError { probe.record { $0.cancelled += 1 } }
            throw error
        }
    }
}

/// The real HTTP server and OpenAI service over a scripted engine.
struct ScriptedServer {
    let server: HTTPServer
    let port: UInt16
    let probe: EngineProbe
    let tokenizer: GFTokenizer
    let backend: SerializedBackend<ScriptedEngine>

    static let replyText = "Hello world, this is a scripted reply."

    static func start(repeats: Bool = false, delay: Duration = .zero, maxWaiting: Int = 4,
                      apiKey: String? = nil, maxContext: Int = 4096) async throws -> ScriptedServer {
        let tokenizer = try await GFTokenizer.load()
        let probe = EngineProbe()
        let engine = try ScriptedEngine(tokenizer: tokenizer, reply: tokenizer.encode(replyText, addBOS: false),
                                        repeats: repeats, delay: delay, probe: probe)
        let backend = SerializedBackend(engine: engine, tokenizer: tokenizer, maxContext: maxContext,
                                        maxWaiting: maxWaiting)
        let service = OpenAIService(
            configuration: OpenAIService.Configuration(
                modelID: "test-model", apiKey: apiKey,
                defaults: SamplingDefaults(temperature: 0, topK: 64, topP: 0.95)),
            backend: backend)
        let server = try HTTPServer(configuration: .init(host: "127.0.0.1", port: 0, idleTimeout: .seconds(5)),
                                    handler: service.handler)
        let port = try await server.start()
        return ScriptedServer(server: server, port: port, probe: probe, tokenizer: tokenizer, backend: backend)
    }

    var base: String { "http://127.0.0.1:\(port)" }

    /// Waits until the backend is idle and `condition` holds on the probe.
    func waitUntil(timeout: Duration = .seconds(5),
                   _ condition: (EngineProbe.Counts) -> Bool) async -> Bool {
        let deadline = ContinuousClock.now + timeout
        while ContinuousClock.now < deadline {
            if condition(probe.counts), !backend.queueStatus.running { return true }
            try? await Task.sleep(for: .milliseconds(10))
        }
        return false
    }
}

/// A raw TCP client, for wire-level checks URLSession hides.
final class RawClient {
    let task: URLSessionStreamTask
    private var buffer = Data()

    init(port: UInt16) {
        task = URLSession.shared.streamTask(withHostName: "127.0.0.1", port: Int(port))
        task.resume()
    }

    func send(_ text: String) async throws {
        try await task.write(Data(text.utf8), timeout: 5)
    }

    /// Reads until `marker` has arrived; returns everything up to and
    /// including it and keeps the rest.
    func read(through marker: String, timeout: TimeInterval = 10) async throws -> String {
        let needle = Data(marker.utf8)
        while true {
            if let range = buffer.range(of: needle) {
                let head = buffer[..<range.upperBound]
                buffer = Data(buffer[range.upperBound...])
                return String(decoding: head, as: UTF8.self)
            }
            let (data, atEOF) = try await task.readData(ofMinLength: 1, maxLength: 65_536, timeout: timeout)
            if let data { buffer.append(data) }
            if atEOF, data?.isEmpty ?? true {
                throw URLError(.networkConnectionLost)
            }
        }
    }

    /// Reads a response with a `Content-Length` body.
    func readResponse() async throws -> (head: String, body: String) {
        let head = try await read(through: "\r\n\r\n")
        let length = head.split(separator: "\r\n")
            .first { $0.lowercased().hasPrefix("content-length:") }
            .flatMap { Int($0.split(separator: ":")[1].trimmingCharacters(in: .whitespaces)) } ?? 0
        while buffer.count < length {
            let (data, atEOF) = try await task.readData(ofMinLength: 1, maxLength: 65_536, timeout: 10)
            if let data { buffer.append(data) }
            if atEOF { break }
        }
        let body = buffer.prefix(length)
        buffer = Data(buffer.dropFirst(length))
        return (head, String(decoding: body, as: UTF8.self))
    }

    /// Whether the server closed the connection.
    func readsEOF() async throws -> Bool {
        let (data, atEOF) = try await task.readData(ofMinLength: 1, maxLength: 65_536, timeout: 10)
        return atEOF && (data?.isEmpty ?? true)
    }

    func close() {
        task.cancel()
    }
}

/// The `data:` payloads of an SSE body, in order.
func sseEvents(_ text: String) -> [String] {
    text.components(separatedBy: "\n\n").compactMap { block in
        block.hasPrefix("data: ") ? String(block.dropFirst("data: ".count)) : nil
    }
}

func jsonObject(_ text: String) throws -> [String: Any] {
    guard let object = try JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any] else {
        throw URLError(.cannotParseResponse)
    }
    return object
}
