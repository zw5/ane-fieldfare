public import Foundation
import ANEFieldfareHotpath
public import ANEFieldfareHotpathCLICore

/// Sampling defaults for requests that leave a field out. The server's flags
/// set them; the built-in values are the CLI's (`OptionDefaults`).
public struct SamplingDefaults: Sendable, Equatable {
    public var temperature: Float
    public var topK: Int?
    public var topP: Float?
    /// nil: until a stop or the end of the context.
    public var maxTokens: Int?

    public init(temperature: Float = OptionDefaults.temperature, topK: Int? = OptionDefaults.topK,
                topP: Float? = OptionDefaults.topP, maxTokens: Int? = nil) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.maxTokens = maxTokens
    }
}

/// A validated chat or text completion request.
public struct CompletionRequest: Sendable, Equatable {
    public var prompt: PromptSource
    public var parameters: CompletionParameters
    public var stream: Bool
    public var includeUsage: Bool
}

// MARK: - Wire shapes

/// `"stop": "x"` or `"stop": ["x", "y"]`.
private enum StopField: Decodable {
    case one(String)
    case many([String])

    init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let one = try? container.decode(String.self) {
            self = .one(one)
        } else {
            self = .many(try container.decode([String].self))
        }
    }

    var values: [String] {
        switch self {
        case .one(let value): return [value]
        case .many(let values): return values
        }
    }
}

private struct StreamOptions: Decodable {
    let includeUsage: Bool?
}

/// The request fields both endpoints share. `model` names nothing (the
/// server serves one model) and `user` is advisory, so both are ignored with
/// every other unknown field; fields that would change the output are
/// refused by `refuseUnsupported` instead.
private struct CommonFields: Decodable {
    let stream: Bool?
    let streamOptions: StreamOptions?
    let maxTokens: Int?
    let maxCompletionTokens: Int?
    let temperature: Double?
    let topP: Double?
    let topK: Int?
    let repetitionPenalty: Double?
    let stop: StopField?
    let seed: Int64?
    let n: Int?
    let logprobs: LogprobsField?
}

/// `logprobs` is a boolean on chat and an integer on completions; any
/// request for them is refused.
private enum LogprobsField: Decodable {
    case requested(Bool)

    init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let flag = try? container.decode(Bool.self) {
            self = .requested(flag)
        } else {
            self = .requested(try container.decode(Int.self) > 0)
        }
    }
}

private struct ChatFields: Decodable {
    let messages: [ChatMessageJSON]
}

private struct CompletionFields: Decodable {
    let prompt: PromptField
    let echo: Bool?
    let bestOf: Int?
    let suffix: String?
}

private enum PromptField: Decodable {
    case text(String)
    case texts([String])
    case tokens([Int32])

    init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let text = try? container.decode(String.self) {
            self = .text(text)
        } else if let texts = try? container.decode([String].self) {
            self = .texts(texts)
        } else if let tokens = try? container.decode([Int32].self) {
            self = .tokens(tokens)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container, debugDescription: "prompt must be a string, an array of one string, or token IDs")
        }
    }
}

struct ChatMessageJSON: Decodable {
    let role: String
    let content: ChatContentJSON?
}

enum ChatContentJSON: Decodable {
    case text(String)
    case parts([ChatPartJSON])

    init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let text = try? container.decode(String.self) {
            self = .text(text)
        } else {
            self = .parts(try container.decode([ChatPartJSON].self))
        }
    }
}

struct ChatPartJSON: Decodable {
    let type: String
    let text: String?
}

// MARK: - Decoding and validation

public enum OpenAIRequestDecoder {
    private static func decoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }

    private static func decode<T: Decodable>(_ type: T.Type, from body: Data) throws -> T {
        do {
            return try decoder().decode(type, from: body)
        } catch let error as DecodingError {
            throw APIError.badRequest(describe(error), code: "invalid_json")
        }
    }

    public static func chat(_ body: Data, defaults: SamplingDefaults) throws -> CompletionRequest {
        try refuseUnsupported(requireObject(body))
        let common = try decode(CommonFields.self, from: body)
        let chat = try decode(ChatFields.self, from: body)
        let messages = try ChatPrompt.messages(from: chat.messages)
        return try request(prompt: .chat(messages), common: common, defaults: defaults)
    }

    public static func completion(_ body: Data, defaults: SamplingDefaults) throws -> CompletionRequest {
        try refuseUnsupported(requireObject(body))
        let common = try decode(CommonFields.self, from: body)
        let fields = try decode(CompletionFields.self, from: body)
        if fields.echo == true {
            throw APIError.badRequest("echo is not supported", param: "echo")
        }
        if let bestOf = fields.bestOf, bestOf != 1 {
            throw APIError.badRequest("best_of must be 1", param: "best_of")
        }
        if let suffix = fields.suffix, !suffix.isEmpty {
            throw APIError.badRequest("suffix is not supported", param: "suffix")
        }
        let prompt: PromptSource
        switch fields.prompt {
        case .text(let text):
            prompt = .text(text)
        case .texts(let texts):
            guard texts.count == 1 else {
                throw APIError.badRequest("prompt must hold exactly one string (batches are not supported)",
                                          param: "prompt")
            }
            prompt = .text(texts[0])
        case .tokens(let tokens):
            prompt = .tokens(tokens)
        }
        return try request(prompt: prompt, common: common, defaults: defaults)
    }

    private static func requireObject(_ body: Data) throws -> [String: Any] {
        guard !body.isEmpty else {
            throw APIError.badRequest("the request body must be a JSON object", code: "invalid_json")
        }
        let parsed: Any
        do {
            parsed = try JSONSerialization.jsonObject(with: body, options: [.fragmentsAllowed])
        } catch {
            throw APIError.badRequest("the request body is not valid JSON", code: "invalid_json")
        }
        guard let object = parsed as? [String: Any] else {
            throw APIError.badRequest("the request body must be a JSON object", code: "invalid_json")
        }
        return object
    }

    /// Fields this server cannot honour, refused unless they hold the value
    /// that asks for nothing: tools, a structured response format, and
    /// presence or frequency penalties.
    private static func refuseUnsupported(_ object: [String: Any]) throws {
        if let tools = object["tools"] as? [Any], !tools.isEmpty {
            throw APIError.badRequest("tools are not supported", param: "tools")
        }
        if let format = object["response_format"] as? [String: Any], format["type"] as? String != "text" {
            throw APIError.badRequest("response_format other than text is not supported", param: "response_format")
        }
        for penalty in ["presence_penalty", "frequency_penalty"] {
            if let value = object[penalty] as? NSNumber, value.doubleValue != 0 {
                throw APIError.badRequest("\(penalty) is not supported", param: penalty)
            }
        }
    }

    private static func request(prompt: PromptSource, common: CommonFields,
                                defaults: SamplingDefaults) throws -> CompletionRequest {
        if let n = common.n, n != 1 {
            throw APIError.badRequest("n must be 1", param: "n")
        }
        if case .requested(true) = common.logprobs {
            throw APIError.badRequest("logprobs are not supported", param: "logprobs")
        }
        let maxTokens = common.maxCompletionTokens ?? common.maxTokens ?? defaults.maxTokens
        if let maxTokens, maxTokens < 1 {
            throw APIError.badRequest("max_tokens must be at least 1", param: "max_tokens")
        }
        let temperature = try common.temperature.map { value -> Float in
            guard value.isFinite, value >= 0 else {
                throw APIError.badRequest("temperature must be a nonnegative number", param: "temperature")
            }
            return Float(value)
        } ?? defaults.temperature
        let topP = try common.topP.map { value -> Float in
            guard value.isFinite, value > 0, value <= 1 else {
                throw APIError.badRequest("top_p must be greater than 0 and at most 1", param: "top_p")
            }
            return Float(value)
        } ?? defaults.topP
        var topK = defaults.topK
        if let value = common.topK {
            // 0 and -1 are the usual spellings of "no top-k".
            guard value <= 256, value >= -1 else {
                throw APIError.badRequest("top_k must be between 1 and 256 (0 or -1: off)", param: "top_k")
            }
            topK = value >= 1 ? value : nil
        }
        let repetitionPenalty = try common.repetitionPenalty.map { value -> Float in
            guard value.isFinite, value > 0 else {
                throw APIError.badRequest("repetition_penalty must be greater than 0", param: "repetition_penalty")
            }
            return Float(value)
        } ?? 1
        if temperature > 0, topK == nil, let topP, topP < 1 {
            throw APIError.badRequest(
                "top_p below 1 requires top_k between 1 and 256: full-vocabulary nucleus sampling "
                    + "is not implemented", param: "top_p")
        }
        let stop = (common.stop?.values ?? []).filter { !$0.isEmpty }
        guard stop.count <= 16 else {
            throw APIError.badRequest("at most 16 stop sequences are supported", param: "stop")
        }
        let stream = common.stream ?? false
        return CompletionRequest(
            prompt: prompt,
            parameters: CompletionParameters(
                maxTokens: maxTokens,
                sampling: SamplingParameters(
                    temperature: temperature,
                    topK: topK,
                    topP: topP,
                    repetitionPenalty: repetitionPenalty,
                    seed: common.seed.map { UInt64(bitPattern: $0) }),
                stop: stop),
            stream: stream,
            includeUsage: stream && (common.streamOptions?.includeUsage ?? false))
    }

    /// The wire spelling of a decoded key: `maxTokens` is `max_tokens`.
    private static func snakeCase(_ key: String) -> String {
        key.reduce(into: "") { result, character in
            if character.isUppercase { result += "_" + character.lowercased() } else { result.append(character) }
        }
    }

    private static func describe(_ error: DecodingError) -> String {
        func path(_ context: DecodingError.Context) -> String {
            let keys = context.codingPath.map { key in
                key.intValue.map { "[\($0)]" } ?? snakeCase(key.stringValue)
            }
            return keys.isEmpty ? "the request" : keys.joined(separator: ".")
                .replacingOccurrences(of: ".[", with: "[")
        }
        switch error {
        case .keyNotFound(let key, let context):
            let parent = context.codingPath.isEmpty ? "" : " in \(path(context))"
            return "missing required field '\(snakeCase(key.stringValue))'\(parent)"
        case .typeMismatch(_, let context), .valueNotFound(_, let context):
            return "invalid type at \(path(context)): \(context.debugDescription)"
        case .dataCorrupted(let context):
            return "invalid value at \(path(context)): \(context.debugDescription)"
        @unknown default:
            return "invalid request body"
        }
    }
}

/// OpenAI chat messages to the runtime's chat turns.
public enum ChatPrompt {
    /// Text parts are joined without a separator, as the CLI's
    /// `--messages-file` joins them. Leading `system`/`developer` messages
    /// merge into one system turn, separated by a blank line; the Gemma
    /// template has one system turn, first. Image, audio and tool content is
    /// refused: this server serves text.
    static func messages(from rows: [ChatMessageJSON]) throws -> [GFTokenizer.Message] {
        guard !rows.isEmpty else {
            throw APIError.badRequest("messages must not be empty", param: "messages")
        }
        var system: [String] = []
        var turns: [GFTokenizer.Message] = []
        for (index, row) in rows.enumerated() {
            let param = "messages[\(index)]"
            let text: String
            switch row.content {
            case nil:
                throw APIError.badRequest("\(param) has no content (tool calls are not supported)", param: param)
            case .text(let value)?:
                text = value
            case .parts(let parts)?:
                text = try parts.enumerated().map { partIndex, part -> String in
                    switch part.type {
                    case "text", "input_text":
                        guard let value = part.text else {
                            throw APIError.badRequest("\(param).content[\(partIndex)] has no text", param: param)
                        }
                        return value
                    case "image_url", "input_image", "image":
                        throw APIError(status: 400, code: "unsupported_content", param: param,
                                       message: "image input is not supported by this server")
                    default:
                        throw APIError(status: 400, code: "unsupported_content", param: param,
                                       message: "unsupported content part type '\(part.type)'")
                    }
                }.joined()
            }
            switch row.role {
            case "system", "developer":
                guard turns.isEmpty else {
                    throw APIError.badRequest(
                        "\(param): system messages must come before every user and assistant message",
                        param: param)
                }
                system.append(text)
            case "user":
                turns.append(GFTokenizer.Message(role: .user, content: text))
            case "assistant":
                turns.append(GFTokenizer.Message(role: .assistant, content: text))
            case "tool", "function":
                throw APIError(status: 400, code: "unsupported_content", param: param,
                               message: "tool messages are not supported by this server")
            default:
                throw APIError.badRequest("\(param) has unknown role '\(row.role)'", param: param)
            }
        }
        guard !turns.isEmpty else {
            throw APIError.badRequest("messages must contain a user or assistant message", param: "messages")
        }
        if !system.isEmpty {
            turns.insert(GFTokenizer.Message(role: .system, content: system.joined(separator: "\n\n")), at: 0)
        }
        return turns
    }
}
