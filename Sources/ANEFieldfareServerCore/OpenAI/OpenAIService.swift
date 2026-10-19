public import Foundation
import Synchronization

/// The OpenAI-compatible API over one `CompletionBackend`.
///
/// Routes `/v1/...` and the same paths under `/api/v1/...` (OpenRouter's base
/// URL shape), answers CORS preflights, checks the bearer key when one is
/// configured, and runs each generation in a task group beside a watcher
/// that cancels it when the client disconnects.
public struct OpenAIService: Sendable {
    public struct Configuration: Sendable {
        public var modelID: String
        public var apiKey: String?
        public var defaults: SamplingDefaults
        /// Unix time the server started; the model's `created`.
        public var created: Int
        /// Called after each generation with a one-line summary.
        public var log: @Sendable (String) -> Void

        public init(modelID: String, apiKey: String? = nil, defaults: SamplingDefaults = SamplingDefaults(),
                    created: Int = Int(Date().timeIntervalSince1970),
                    log: @escaping @Sendable (String) -> Void = { _ in }) {
            self.modelID = modelID
            self.apiKey = apiKey
            self.defaults = defaults
            self.created = created
            self.log = log
        }
    }

    public let configuration: Configuration
    public let backend: any CompletionBackend

    public init(configuration: Configuration, backend: any CompletionBackend) {
        self.configuration = configuration
        self.backend = backend
    }

    public var handler: HTTPHandler {
        { request, connection in await self.handle(request, on: connection) }
    }

    // MARK: - Routing

    enum Route: Equatable {
        case chatPage
        case health
        case models
        case model(String)
        case chatCompletions
        case completions
        case notFound
    }

    /// The route for `path`; `/api/v1/...` is `/v1/...`.
    static func route(_ path: String) -> Route {
        var path = path
        if path.hasPrefix("/api/v1/") || path == "/api/v1" { path.removeFirst(4) }
        while path.count > 1, path.hasSuffix("/") { path.removeLast() }
        switch path {
        case "/", "/chat": return .chatPage
        case "/health", "/v1/health": return .health
        case "/v1/models": return .models
        case "/v1/chat/completions": return .chatCompletions
        case "/v1/completions": return .completions
        default:
            if path.hasPrefix("/v1/models/") {
                let id = String(path.dropFirst("/v1/models/".count))
                return .model(id.removingPercentEncoding ?? id)
            }
            return .notFound
        }
    }

    static func allowedMethods(_ route: Route) -> [String] {
        switch route {
        case .chatPage, .health, .models, .model: return ["GET", "HEAD", "OPTIONS"]
        case .chatCompletions, .completions: return ["POST", "OPTIONS"]
        case .notFound: return []
        }
    }

    /// CORS headers for every response: any origin may call, with the
    /// `Authorization` header (keys are bearer tokens, not cookies).
    static func corsHeaders(for request: HTTPRequest) -> HTTPHeaders {
        var headers = HTTPHeaders()
        headers.add("Access-Control-Allow-Origin", "*")
        headers.add("Access-Control-Expose-Headers", "Content-Type, Retry-After")
        if request.method == "OPTIONS" {
            headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            let requested = request.headers["Access-Control-Request-Headers"]
            headers.add("Access-Control-Allow-Headers",
                        requested.map { $0.isEmpty ? "Authorization, Content-Type" : $0 }
                            ?? "Authorization, Content-Type")
            headers.add("Access-Control-Max-Age", "86400")
        }
        return headers
    }

    /// Constant-time comparison of the bearer token with the configured key.
    static func authorized(_ request: HTTPRequest, apiKey: String?) -> Bool {
        guard let apiKey else { return true }
        guard let header = request.headers["Authorization"] else { return false }
        let parts = header.split(separator: " ", maxSplits: 1)
        guard parts.count == 2, parts[0].lowercased() == "bearer" else { return false }
        let given = Array(parts[1].trimmingCharacters(in: .whitespaces).utf8)
        let expected = Array(apiKey.utf8)
        var difference = given.count ^ expected.count
        for index in 0..<max(given.count, expected.count) {
            let a = index < given.count ? given[index] : 0
            let b = index < expected.count ? expected[index] : 0
            difference |= Int(a ^ b)
        }
        return difference == 0
    }

    // MARK: - Handling

    func handle(_ request: HTTPRequest, on connection: HTTPConnection) async -> HTTPDisposition {
        let cors = Self.corsHeaders(for: request)
        let route = Self.route(request.path)
        let allowed = Self.allowedMethods(route)

        func send(_ response: HTTPResponse) async -> HTTPDisposition {
            var response = response
            for field in cors.fields { response.headers.set(field.name, field.value) }
            let data = response.serialized(keepAlive: request.keepAlive, includeBody: request.method != "HEAD")
            do {
                try await connection.send(data)
                return .keepAlive
            } catch {
                return .close
            }
        }
        func fail(_ error: APIError, headers: HTTPHeaders = HTTPHeaders()) async -> HTTPDisposition {
            await send(OpenAIErrorResponse.response(error, headers: headers))
        }

        if route == .notFound {
            return await fail(APIError(status: 404, code: "not_found",
                                       message: "no route for \(request.method) \(request.path)"))
        }
        if request.method == "OPTIONS" {
            var headers = HTTPHeaders()
            headers.add("Allow", allowed.joined(separator: ", "))
            return await send(HTTPResponse(status: 204, headers: headers))
        }
        guard allowed.contains(request.method) else {
            var headers = HTTPHeaders()
            headers.add("Allow", allowed.joined(separator: ", "))
            return await fail(APIError(status: 405, code: "method_not_allowed",
                                       message: "\(request.method) is not allowed on \(request.path)"),
                              headers: headers)
        }
        // The page holds no data; its requests carry the key the user enters.
        if route != .health, route != .chatPage, !Self.authorized(request, apiKey: configuration.apiKey) {
            var headers = HTTPHeaders()
            headers.add("WWW-Authenticate", "Bearer")
            return await fail(APIError(status: 401, code: "invalid_api_key",
                                       message: "missing or incorrect API key: send Authorization: Bearer <key>"),
                              headers: headers)
        }

        switch route {
        case .chatPage:
            var headers = HTTPHeaders()
            headers.add("Content-Type", "text/html; charset=utf-8")
            headers.add("Cache-Control", "no-cache")
            return await send(HTTPResponse(status: 200, headers: headers, body: Data(ChatPage.html.utf8)))
        case .health:
            let status = backend.queueStatus
            let body: [String: any Sendable] = [
                "status": "ok", "model": configuration.modelID,
                "busy": status.running, "queued": status.waiting,
            ]
            return await send(json(body))
        case .models:
            return await send(json(["object": "list", "data": [modelObject()]]))
        case .model(let id):
            guard id == configuration.modelID else {
                return await fail(APIError(status: 404, code: "model_not_found",
                                           message: "the model '\(id)' does not exist"))
            }
            return await send(json(modelObject()))
        case .chatCompletions, .completions:
            let chat = route == .chatCompletions
            let parsed: CompletionRequest
            let prompt: PreparedPrompt
            do {
                parsed = chat
                    ? try OpenAIRequestDecoder.chat(request.body, defaults: configuration.defaults)
                    : try OpenAIRequestDecoder.completion(request.body, defaults: configuration.defaults)
                prompt = try backend.prepare(parsed.prompt)
            } catch let error as APIError {
                return await fail(error)
            } catch {
                return await fail(APIError(status: 500, type: "server_error", code: "internal_error",
                                           message: "\(error)"))
            }
            let envelope = CompletionEnvelope(
                id: (chat ? "chatcmpl-" : "cmpl-") + Self.randomID(),
                object: chat ? (parsed.stream ? "chat.completion.chunk" : "chat.completion") : "text_completion",
                created: Int(Date().timeIntervalSince1970),
                model: configuration.modelID)
            let generation = Generation(request: request, cors: cors, chat: chat, parsed: parsed, prompt: prompt,
                                        envelope: envelope, backend: backend, connection: connection,
                                        log: configuration.log)
            return await generation.run()
        case .notFound:
            return .close
        }
    }

    private func modelObject() -> [String: any Sendable] {
        [
            "id": configuration.modelID,
            "object": "model",
            "created": configuration.created,
            "owned_by": "ane-fieldfare",
            "context_length": backend.maxContext,
        ]
    }

    private func json(_ object: [String: any Sendable]) -> HTTPResponse {
        let data = (try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes]))
            ?? Data("{}".utf8)
        var headers = HTTPHeaders()
        headers.add("Content-Type", "application/json")
        return HTTPResponse(status: 200, headers: headers, body: data)
    }

    static func randomID() -> String {
        let alphabet = Array("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        var generator = SystemRandomNumberGenerator()
        return String((0..<24).map { _ in alphabet[Int(generator.next(upperBound: UInt64(alphabet.count)))] })
    }
}

/// What the generating task reports to the task writing the response.
enum GenerationEvent: Sendable {
    case started
    case text(String)
    case finished(CompletionOutcome)
    case failed(any Error)
}

/// One completion request from admission to the last byte.
///
/// Three parties: the generating child task, which waits its turn and runs
/// the model; a watcher child, which returns when the client disconnects;
/// and the parent, which writes the response from the generation's events.
/// Whichever way it ends, the parent cancels the group, so a disconnect
/// cancels the generation (or its place in line), and a finished generation
/// stops the watcher.
struct Generation: Sendable {
    let request: HTTPRequest
    let cors: HTTPHeaders
    let chat: Bool
    let parsed: CompletionRequest
    let prompt: PreparedPrompt
    let envelope: CompletionEnvelope
    let backend: any CompletionBackend
    let connection: HTTPConnection
    let log: @Sendable (String) -> Void

    func run() async -> HTTPDisposition {
        let (events, sink) = AsyncStream<GenerationEvent>.makeStream()
        let backend = backend
        let prompt = prompt
        let parameters = parsed.parameters
        let connection = connection
        return await withTaskGroup(of: Void.self) { group in
            group.addTask {
                do {
                    let outcome = try await backend.complete(
                        prompt, parameters: parameters,
                        onStart: { sink.yield(.started) },
                        onText: { sink.yield(.text($0)) })
                    sink.yield(.finished(outcome))
                } catch {
                    sink.yield(.failed(error))
                }
                sink.finish()
            }
            group.addTask {
                await connection.waitUntilClosed()
                // Ends the event stream; the writer sees no outcome and stops.
                sink.finish()
            }
            defer { group.cancelAll() }
            return parsed.stream ? await stream(events) : await respond(events)
        }
    }

    private var keepAlive: Bool { request.keepAlive }

    private func headers(contentType: String) -> HTTPHeaders {
        var headers = cors
        headers.set("Content-Type", contentType)
        return headers
    }

    private func errorResponse(_ error: any Error) -> APIError {
        if let error = error as? APIError { return error }
        return APIError(status: 500, type: "server_error", code: "internal_error", message: "\(error)")
    }

    private func logOutcome(_ outcome: CompletionOutcome) {
        log(unsafe "[\(envelope.id)] path=\(outcome.decodePath) prompt=\(outcome.promptTokens)tok "
            + "new=\(outcome.completionTokens)tok finish=\(outcome.finishReason.rawValue) "
            + "prefill=\(String(format: "%.2f", outcome.prefillSeconds))s "
            + "decode=\(String(format: "%.2f", outcome.decodeSeconds))s "
            + "tok/s=\(String(format: "%.3f", outcome.tokensPerSecond))")
    }

    /// Non-streaming: collect the text, answer once.
    private func respond(_ events: AsyncStream<GenerationEvent>) async -> HTTPDisposition {
        var text = ""
        for await event in events {
            switch event {
            case .started:
                break
            case .text(let delta):
                text += delta
            case .finished(let outcome):
                logOutcome(outcome)
                let body = CompletionBody(
                    envelope: envelope,
                    choice: chat ? .message(text) : .text(text),
                    finishReason: outcome.finishReason,
                    usage: UsageJSON(outcome),
                    encodeNullUsage: false,
                    timings: TimingsJSON(outcome))
                let response = HTTPResponse(status: 200, headers: headers(contentType: "application/json"),
                                            body: OpenAIJSON.encode(body))
                return await send(response.serialized(keepAlive: keepAlive)) ? .keepAlive : .close
            case .failed(let error):
                if error is CancellationError || connection.isClosed { return .close }
                let failure = errorResponse(error)
                var extra = cors
                if failure.code == APIError.busy.code { extra.set("Retry-After", "1") }
                let response = OpenAIErrorResponse.response(failure, headers: extra)
                return await send(response.serialized(keepAlive: keepAlive)) ? .keepAlive : .close
            }
        }
        // The client left before an outcome.
        log("[\(envelope.id)] client disconnected; generation cancelled")
        return .close
    }

    /// Streaming: SSE chunks as the text arrives.
    private func stream(_ events: AsyncStream<GenerationEvent>) async -> HTTPDisposition {
        // HTTP/1.0 has no chunked coding; the stream ends with the connection.
        let chunked = request.version == .http11
        let persistent = chunked && keepAlive
        var started = false
        func frame(_ data: Data) -> Data { chunked ? HTTPChunkedEncoding.chunk(data) : data }
        func chunk(_ choice: CompletionBody.Choice, finish: FinishReason? = nil,
                   usage: UsageJSON? = nil, timings: TimingsJSON? = nil) -> Data {
            frame(ServerSentEvents.event(OpenAIJSON.encode(CompletionBody(
                envelope: envelope, choice: choice, finishReason: finish, usage: usage,
                encodeNullUsage: parsed.includeUsage, timings: timings))))
        }
        func start() async -> Bool {
            started = true
            var headers = headers(contentType: "text/event-stream; charset=utf-8")
            headers.set("Cache-Control", "no-cache")
            headers.set("X-Accel-Buffering", "no")
            var data = HTTPResponse.streamingHead(status: 200, headers: headers, version: request.version,
                                                  keepAlive: persistent)
            // The role arrives first, before the prompt is processed.
            data.append(chat ? chunk(.delta(role: "assistant", content: "")) : Data())
            return await send(data)
        }

        for await event in events {
            switch event {
            case .started:
                guard await start() else { return .close }
            case .text(let delta):
                if !started, !(await start()) { return .close }
                guard await send(chunk(chat ? .delta(role: nil, content: delta) : .text(delta))) else {
                    return .close
                }
            case .finished(let outcome):
                logOutcome(outcome)
                if !started, !(await start()) { return .close }
                var tail = chunk(chat ? .delta(role: nil, content: nil) : .text(""), finish: outcome.finishReason,
                                 timings: TimingsJSON(outcome))
                if parsed.includeUsage { tail.append(chunk(.none, usage: UsageJSON(outcome))) }
                tail.append(frame(ServerSentEvents.done))
                if chunked { tail.append(HTTPChunkedEncoding.end) }
                guard await send(tail) else { return .close }
                return persistent ? .keepAlive : .close
            case .failed(let error):
                if error is CancellationError || connection.isClosed { return .close }
                let failure = errorResponse(error)
                guard started else {
                    // Nothing sent yet: an ordinary error response.
                    var extra = cors
                    if failure.code == APIError.busy.code { extra.set("Retry-After", "1") }
                    let response = OpenAIErrorResponse.response(failure, headers: extra)
                    return await send(response.serialized(keepAlive: keepAlive)) ? .keepAlive : .close
                }
                // Headers are out: report the error in the stream and end it.
                log("[\(envelope.id)] failed mid-stream: \(failure.message)")
                var tail = frame(ServerSentEvents.event(OpenAIErrorResponse.body(failure)))
                tail.append(frame(ServerSentEvents.done))
                if chunked { tail.append(HTTPChunkedEncoding.end) }
                _ = await send(tail)
                return .close
            }
        }
        log("[\(envelope.id)] client disconnected; generation cancelled")
        return .close
    }

    private func send(_ data: Data) async -> Bool {
        do {
            try await connection.send(data)
            return true
        } catch {
            return false
        }
    }
}
