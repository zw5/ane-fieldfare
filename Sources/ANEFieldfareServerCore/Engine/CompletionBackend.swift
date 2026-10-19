import Foundation
public import ANEFieldfareHotpath

/// A prompt as the API received it, before templating and tokenizing.
public enum PromptSource: Sendable, Equatable {
    /// Chat turns, rendered with the Gemma chat template.
    case chat([GFTokenizer.Message])
    /// Raw completion text; `<bos>` is prepended, as the CLI's `--prompt`.
    case text(String)
    /// Token IDs, used verbatim.
    case tokens([Int32])
}

public struct PreparedPrompt: Sendable, Equatable {
    public let tokenIDs: [Int32]

    public init(tokenIDs: [Int32]) {
        self.tokenIDs = tokenIDs
    }
}

public struct SamplingParameters: Sendable, Equatable {
    public var temperature: Float
    public var topK: Int?
    public var topP: Float?
    public var repetitionPenalty: Float
    public var seed: UInt64?

    public init(temperature: Float, topK: Int?, topP: Float?, repetitionPenalty: Float = 1, seed: UInt64? = nil) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
    }
}

public struct CompletionParameters: Sendable, Equatable {
    /// nil: generate until a stop or the end of the context.
    public var maxTokens: Int?
    public var sampling: SamplingParameters
    public var stop: [String]

    public init(maxTokens: Int?, sampling: SamplingParameters, stop: [String]) {
        self.maxTokens = maxTokens
        self.sampling = sampling
        self.stop = stop
    }

    /// The loop's configuration for `promptTokens` prompt tokens in a
    /// `maxContext` window. `max_tokens` past the window is clamped to it, as
    /// the CLI clamps `--max-new`.
    public func generationConfig(promptTokens: Int, maxContext: Int) -> GenerationConfig {
        let room = maxContext - promptTokens
        return GenerationConfig(
            maxNewTokens: min(maxTokens ?? room, room),
            temperature: sampling.temperature,
            topK: sampling.topK,
            topP: sampling.topP,
            repetitionPenalty: sampling.repetitionPenalty,
            seed: sampling.seed,
            stopStrings: stop,
            extraStopTokens: [])
    }
}

public enum FinishReason: String, Sendable, Equatable {
    case stop
    case length
}

extension FinishReason {
    /// End of turn, EOS and stop strings are `stop`; the token limit is
    /// `length`. The loop's `.cancelled` never reaches a client.
    public init(_ reason: StopReason) {
        switch reason {
        case .maxTokens: self = .length
        case .eos, .endOfTurn, .stopString, .toolCalls, .cancelled: self = .stop
        }
    }
}

public struct CompletionOutcome: Sendable, Equatable {
    public let promptTokens: Int
    public let completionTokens: Int
    public let finishReason: FinishReason
    public let prefillSeconds: Double
    public let decodeSeconds: Double
    /// Which producer decoded: `speculative`, `hot-head`, `sampled` or `plain`.
    public let decodePath: String
    /// Prompt tokens the engine kept cached from the previous request instead
    /// of prefilling.
    public var cachedTokens = 0

    public init(promptTokens: Int, completionTokens: Int, finishReason: FinishReason,
                prefillSeconds: Double, decodeSeconds: Double, decodePath: String) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.finishReason = finishReason
        self.prefillSeconds = prefillSeconds
        self.decodeSeconds = decodeSeconds
        self.decodePath = decodePath
    }

    public var tokensPerSecond: Double {
        decodeSeconds > 0 ? Double(completionTokens) / decodeSeconds : 0
    }
}

/// A refusal the API reports with its own status, type and code.
public struct APIError: Error, Sendable, Equatable, CustomStringConvertible {
    public let status: Int
    public let type: String
    public let code: String?
    public let param: String?
    public let message: String

    public init(status: Int, type: String = "invalid_request_error", code: String? = nil,
                param: String? = nil, message: String) {
        self.status = status
        self.type = type
        self.code = code
        self.param = param
        self.message = message
    }

    public var description: String { message }

    public static func badRequest(_ message: String, param: String? = nil,
                                  code: String? = "invalid_request") -> APIError {
        APIError(status: 400, code: code, param: param, message: message)
    }

    public static let busy = APIError(
        status: 503, type: "server_error", code: "server_busy",
        message: "the server is busy: too many requests are waiting for the model; retry later")
}

/// What the HTTP layer generates through. `SerializedBackend` is the one
/// implementation; tests drive it with a scripted engine.
public protocol CompletionBackend: Sendable {
    var maxContext: Int { get }
    var queueStatus: GenerationGate.Status { get }
    /// Templates and tokenizes; refuses an empty prompt (400) and one that
    /// leaves no room to generate (413).
    func prepare(_ prompt: PromptSource) throws -> PreparedPrompt
    /// Waits its turn, then generates. `onStart` runs once the model is this
    /// request's; `onText` receives the visible text in order, with stop
    /// strings already applied. Throws `APIError.busy` when the line is full
    /// and `CancellationError` when the task is cancelled.
    func complete(_ prompt: PreparedPrompt, parameters: CompletionParameters,
                  onStart: @Sendable () -> Void,
                  onText: @Sendable (String) -> Void) async throws -> CompletionOutcome
}

/// One loaded model: owns the GPU state of one generation at a time.
///
/// Not `Sendable` — it holds Metal buffers. `SerializedBackend` lets one
/// task at a time use it, behind its `GenerationGate`.
public protocol CompletionEngine: AnyObject {
    func complete(promptIDs: [Int32], config: GenerationConfig,
                  onProgress: (RawDecodeProgress) -> Void) async throws -> EngineCompletion
}

public struct EngineCompletion: Sendable {
    public let result: RawDecodeResult
    public let decodePath: String

    public init(result: RawDecodeResult, decodePath: String) {
        self.result = result
        self.decodePath = decodePath
    }
}

/// The engine, shared by reference between request tasks.
///
/// Region isolation cannot express lending one non-`Sendable` object to a
/// task and taking it back: a `Mutex` refuses a captured value, and a value
/// sent into a task does not come back. A generation has to run in its
/// request's task, so that the client's disconnect cancels exactly it through
/// the runtime's `Task.checkCancellation()` polls in prefill and decode;
/// a long-lived worker task owning the engine could not be cancelled per
/// request. So the engine is held here and the gate provides the exclusion.
// SAFETY: `engine` is read only by `SerializedBackend.complete`, between a
// successful `gate.acquire()` and the matching `gate.release()`. The gate
// admits one task at a time and hands over through its `Mutex`, so each
// holder's use of the engine happens after the previous holder's ended.
// Nothing else keeps a reference to the engine.
private final class EngineCell<Engine: CompletionEngine>: @unchecked Sendable {
    let engine: Engine

    init(_ engine: sending Engine) {
        self.engine = engine
    }
}

/// Serializes generations over one engine: one generates, up to
/// `maxWaiting` wait in line, first come first served, and more are refused
/// with `APIError.busy`. Each generation runs in its caller's task, so
/// cancelling that task (the client disconnected) cancels the generation, or
/// gives up its place in line.
public final class SerializedBackend<Engine: CompletionEngine>: CompletionBackend {
    public let tokenizer: GFTokenizer
    public let maxContext: Int
    private let gate: GenerationGate
    private let cell: EngineCell<Engine>

    public init(engine: sending Engine, tokenizer: GFTokenizer, maxContext: Int, maxWaiting: Int) {
        self.tokenizer = tokenizer
        self.maxContext = maxContext
        self.gate = GenerationGate(maxWaiting: maxWaiting)
        self.cell = EngineCell(engine)
    }

    public var queueStatus: GenerationGate.Status { gate.status }

    public func prepare(_ prompt: PromptSource) throws -> PreparedPrompt {
        let ids: [Int32]
        switch prompt {
        case .chat(let messages):
            guard !messages.isEmpty else { throw APIError.badRequest("messages must not be empty", param: "messages") }
            let rendered: String
            do {
                rendered = try tokenizer.applyChatTemplate(messages)
            } catch {
                throw APIError.badRequest("\(error)", param: "messages")
            }
            ids = tokenizer.encode(rendered, addBOS: false)
        case .text(let text):
            guard !text.isEmpty else { throw APIError.badRequest("prompt must not be empty", param: "prompt") }
            ids = tokenizer.encode(text, addBOS: true)
        case .tokens(let tokens):
            guard !tokens.isEmpty else { throw APIError.badRequest("prompt must not be empty", param: "prompt") }
            guard tokens.allSatisfy({ $0 >= 0 && Int($0) < tokenizer.vocabSize }) else {
                throw APIError.badRequest("prompt token IDs must be in 0..<\(tokenizer.vocabSize)", param: "prompt")
            }
            ids = tokens
        }
        guard ids.count < maxContext else {
            throw APIError(status: 413, code: "context_length_exceeded", param: "messages",
                           message: "the prompt is \(ids.count) tokens, which reaches this server's "
                               + "context of \(maxContext) tokens")
        }
        return PreparedPrompt(tokenIDs: ids)
    }

    public func complete(_ prompt: PreparedPrompt, parameters: CompletionParameters,
                         onStart: @Sendable () -> Void,
                         onText: @Sendable (String) -> Void) async throws -> CompletionOutcome {
        let config = parameters.generationConfig(promptTokens: prompt.tokenIDs.count, maxContext: maxContext)
        do {
            try config.validate()
        } catch {
            throw APIError.badRequest("\(error)")
        }
        do {
            try await gate.acquire()
        } catch is GenerationGate.Busy {
            throw APIError.busy
        }
        defer { gate.release() }
        // Granted after its client left: give the turn up before any GPU work.
        try Task.checkCancellation()
        let engine = cell.engine
        onStart()
        let completion = try await engine.complete(promptIDs: prompt.tokenIDs, config: config) { progress in
            switch progress {
            case .token(_, _, let delta) where !delta.isEmpty: onText(delta)
            case .tail(let tail) where !tail.isEmpty: onText(tail)
            default: break
            }
        }
        let result = completion.result
        var outcome = CompletionOutcome(
            promptTokens: prompt.tokenIDs.count,
            completionTokens: result.newTokens,
            finishReason: FinishReason(result.reason),
            prefillSeconds: result.prefillSeconds,
            decodeSeconds: result.decodeSeconds,
            decodePath: completion.decodePath)
        outcome.cachedTokens = result.reusedPrefixTokens
        return outcome
    }
}
