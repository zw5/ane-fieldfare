public import Foundation

/// JSON bodies in the OpenAI wire format. Fields OpenAI sends as `null`
/// (`finish_reason` mid-stream, `logprobs`) are encoded as explicit nulls,
/// since some clients validate the chunk shape.
enum OpenAIJSON {
    static func encode(_ value: some Encodable) -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        // Every value here is plain strings and numbers; encoding cannot fail.
        return (try? encoder.encode(value)) ?? Data("{}".utf8)
    }
}

struct UsageJSON: Encodable, Equatable {
    let promptTokens: Int
    let completionTokens: Int
    let totalTokens: Int

    init(_ outcome: CompletionOutcome) {
        promptTokens = outcome.promptTokens
        completionTokens = outcome.completionTokens
        totalTokens = outcome.promptTokens + outcome.completionTokens
    }

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
    }
}

/// Server-side timings, in llama.cpp's `timings` shape. Extra top-level
/// fields are ignored by OpenAI clients.
struct TimingsJSON: Encodable {
    let cacheN: Int
    let promptN: Int
    let promptMS: Double
    let predictedN: Int
    let predictedMS: Double
    let predictedPerSecond: Double
    let decodePath: String

    init(_ outcome: CompletionOutcome) {
        cacheN = outcome.cachedTokens
        promptN = outcome.promptTokens - outcome.cachedTokens
        promptMS = outcome.prefillSeconds * 1000
        predictedN = outcome.completionTokens
        predictedMS = outcome.decodeSeconds * 1000
        predictedPerSecond = outcome.tokensPerSecond
        decodePath = outcome.decodePath
    }

    enum CodingKeys: String, CodingKey {
        case cacheN = "cache_n"
        case promptN = "prompt_n"
        case promptMS = "prompt_ms"
        case predictedN = "predicted_n"
        case predictedMS = "predicted_ms"
        case predictedPerSecond = "predicted_per_second"
        case decodePath = "decode_path"
    }
}

/// The fields every completion object and chunk starts with.
struct CompletionEnvelope: Sendable {
    let id: String
    let object: String
    let created: Int
    let model: String
}

private enum EnvelopeKeys: String, CodingKey {
    case id, object, created, model, choices, usage, timings
    case systemFingerprint = "system_fingerprint"
}

private enum ChoiceKeys: String, CodingKey {
    case index, message, delta, text, logprobs
    case finishReason = "finish_reason"
}

private struct MessageJSON: Encodable {
    let role: String?
    let content: String?

    func encode(to encoder: any Encoder) throws {
        enum Keys: String, CodingKey { case role, content }
        var container = encoder.container(keyedBy: Keys.self)
        try container.encodeIfPresent(role, forKey: .role)
        try container.encodeIfPresent(content, forKey: .content)
    }
}

/// One completion object (`chat.completion`, `text_completion`) or stream
/// chunk (`chat.completion.chunk`, `text_completion` streamed).
struct CompletionBody: Encodable {
    enum Choice {
        /// `chat.completion`: the whole message.
        case message(String)
        /// `chat.completion.chunk`: a delta, optionally with the role.
        case delta(role: String?, content: String?)
        /// `text_completion`, whole or streamed.
        case text(String)
        /// A usage-only chunk: `choices` is empty.
        case none
    }

    let envelope: CompletionEnvelope
    let choice: Choice
    let finishReason: FinishReason?
    let usage: UsageJSON?
    /// Streams with `include_usage` carry `"usage": null` on every chunk but
    /// the last.
    let encodeNullUsage: Bool
    let timings: TimingsJSON?

    func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: EnvelopeKeys.self)
        try container.encode(envelope.id, forKey: .id)
        try container.encode(envelope.object, forKey: .object)
        try container.encode(envelope.created, forKey: .created)
        try container.encode(envelope.model, forKey: .model)
        try container.encode("fp_ane_fieldfare", forKey: .systemFingerprint)
        var choices = container.nestedUnkeyedContainer(forKey: .choices)
        if case .none = choice {} else {
            var entry = choices.nestedContainer(keyedBy: ChoiceKeys.self)
            try entry.encode(0, forKey: .index)
            switch choice {
            case .message(let content):
                try entry.encode(MessageJSON(role: "assistant", content: content), forKey: .message)
            case .delta(let role, let content):
                try entry.encode(MessageJSON(role: role, content: content), forKey: .delta)
            case .text(let text):
                try entry.encode(text, forKey: .text)
            case .none:
                break
            }
            try entry.encodeNil(forKey: .logprobs)
            if let finishReason {
                try entry.encode(finishReason.rawValue, forKey: .finishReason)
            } else {
                try entry.encodeNil(forKey: .finishReason)
            }
        }
        if let usage {
            try container.encode(usage, forKey: .usage)
        } else if encodeNullUsage {
            try container.encodeNil(forKey: .usage)
        }
        try container.encodeIfPresent(timings, forKey: .timings)
    }
}

/// `{"error": {"message", "type", "param", "code"}}`.
public enum OpenAIErrorResponse {
    struct Body: Encodable {
        struct Detail: Encodable {
            let message: String
            let type: String
            let param: String?
            let code: String?

            func encode(to encoder: any Encoder) throws {
                enum Keys: String, CodingKey { case message, type, param, code }
                var container = encoder.container(keyedBy: Keys.self)
                try container.encode(message, forKey: .message)
                try container.encode(type, forKey: .type)
                if let param { try container.encode(param, forKey: .param) } else { try container.encodeNil(forKey: .param) }
                if let code { try container.encode(code, forKey: .code) } else { try container.encodeNil(forKey: .code) }
            }
        }

        let error: Detail
    }

    public static func body(_ error: APIError) -> Data {
        OpenAIJSON.encode(Body(error: .init(message: error.message, type: error.type,
                                            param: error.param, code: error.code)))
    }

    public static func response(_ error: APIError, headers extra: HTTPHeaders = HTTPHeaders()) -> HTTPResponse {
        var headers = extra
        headers.set("Content-Type", "application/json")
        return HTTPResponse(status: error.status, headers: headers, body: body(error))
    }
}

/// Server-sent events framing: one `data:` line per event, a blank line
/// after each. JSON never contains a raw newline, so one line suffices.
public enum ServerSentEvents {
    public static func event(_ json: Data) -> Data {
        var data = Data("data: ".utf8)
        data.append(json)
        data.append(contentsOf: Array("\n\n".utf8))
        return data
    }

    public static let done = Data("data: [DONE]\n\n".utf8)
}
