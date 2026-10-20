import Foundation
import Testing
import ANEFieldfareHotpath
@testable import ANEFieldfareServerCore

/// Request decoding and validation and bearer-key auth, without a model.
@Suite struct OpenAIRequestTests {
    private let defaults = SamplingDefaults(temperature: 0.2, topK: 64, topP: 0.95, maxTokens: nil)

    private func chat(_ json: String) throws -> CompletionRequest {
        try OpenAIRequestDecoder.chat(Data(json.utf8), defaults: defaults)
    }

    private func refusal(_ body: () throws -> Any) -> APIError? {
        do {
            _ = try body()
            return nil
        } catch let error as APIError {
            return error
        } catch {
            return APIError(status: -1, message: "\(error)")
        }
    }

    @Test func everySupportedFieldIsRead() throws {
        let request = try chat("""
        {"messages":[{"role":"user","content":"Hi"}],"stream":true,"stream_options":{"include_usage":true},
         "max_tokens":10,"max_completion_tokens":20,"temperature":0,"top_p":0.5,"top_k":7,
         "repetition_penalty":1.1,"stop":"END","seed":-1,"n":1,"user":"u","logprobs":false,
         "frequency_penalty":0,"presence_penalty":0,"tools":[],"response_format":{"type":"text"},
         "unknown_field":{"nested":true}}
        """)
        #expect(request.stream && request.includeUsage)
        #expect(request.parameters.maxTokens == 20, "max_completion_tokens wins over max_tokens")
        #expect(request.parameters.sampling == SamplingParameters(
            temperature: 0, topK: 7, topP: 0.5, repetitionPenalty: 1.1, seed: UInt64.max))
        #expect(request.parameters.stop == ["END"])
        #expect(try chat(#"{"messages":[{"role":"user","content":"Hi"}],"stop":["a","","b"]}"#).parameters.stop
                == ["a", "b"])
        #expect(try chat(#"{"messages":[{"role":"user","content":"Hi"}],"stop":null,"temperature":null}"#)
            .parameters.sampling.temperature == 0.2)
    }

    @Test func messagesBecomeTemplateTurns() throws {
        let request = try chat("""
        {"messages":[
          {"role":"system","content":"Be brief."},
          {"role":"developer","content":[{"type":"text","text":"Answer in "},{"type":"text","text":"English."}]},
          {"role":"user","content":[{"type":"text","text":"What is 2+2?"}]},
          {"role":"assistant","content":"4"},
          {"role":"user","content":"And 3+3?","name":"x"}]}
        """)
        #expect(request.prompt == .chat([
            GFTokenizer.Message(role: .system, content: "Be brief.\n\nAnswer in English."),
            GFTokenizer.Message(role: .user, content: "What is 2+2?"),
            GFTokenizer.Message(role: .assistant, content: "4"),
            GFTokenizer.Message(role: .user, content: "And 3+3?"),
        ]))
    }

    @Test func unsupportedChatContentIsRefusedClearly() {
        // Image content is refused end to end in ServerIntegrationTests.
        #expect(refusal { try chat(#"{"messages":[{"role":"tool","content":"x","tool_call_id":"1"}]}"#) }?.status == 400)
        #expect(refusal { try chat(#"{"messages":[{"role":"assistant","content":null,"tool_calls":[]}]}"#) }?.status == 400)
        #expect(refusal { try chat(#"{"messages":[{"role":"wizard","content":"x"}]}"#) }?.status == 400)
        #expect(refusal { try chat(#"{"messages":[]}"#) }?.status == 400)
        #expect(refusal { try chat(#"{"messages":[{"role":"system","content":"only"}]}"#) }?.status == 400)
        let late = refusal { try chat("""
        {"messages":[{"role":"user","content":"a"},{"role":"system","content":"late"}]}
        """) }
        #expect(late?.param == "messages[1]")
    }

    @Test func invalidRequestsAreRefusedWithTheirParameter() {
        let user = #""messages":[{"role":"user","content":"Hi"}]"#
        let cases: [(String, String?)] = [
            ("{\(user),\"n\":2}", "n"),
            ("{\(user),\"logprobs\":true}", "logprobs"),
            ("{\(user),\"max_tokens\":0}", "max_tokens"),
            ("{\(user),\"temperature\":-1}", "temperature"),
            ("{\(user),\"top_p\":0}", "top_p"),
            ("{\(user),\"top_k\":1000}", "top_k"),
            ("{\(user),\"top_k\":0,\"top_p\":0.5,\"temperature\":1}", "top_p"),
            ("{\(user),\"repetition_penalty\":0}", "repetition_penalty"),
            ("{\(user),\"tools\":[{\"type\":\"function\"}]}", "tools"),
            ("{\(user),\"response_format\":{\"type\":\"json_object\"}}", "response_format"),
            ("{\(user),\"presence_penalty\":0.5}", "presence_penalty"),
            ("{\(user),\"frequency_penalty\":1}", "frequency_penalty"),
        ]
        for (json, param) in cases {
            let error = refusal { try chat(json) }
            #expect(error?.status == 400, "\(json)")
            #expect(error?.param == param, "\(json)")
        }
        #expect(refusal { try chat("not json") }?.code == "invalid_json")
        #expect(refusal { try chat("[1,2]") }?.code == "invalid_json")
        #expect(refusal { try chat(#"{"model":"m"}"#) }?.message == "missing required field 'messages'")
        #expect(refusal { try chat(#"{"messages":[{"role":"user","content":"Hi"}],"max_tokens":"ten"}"#) }?
            .message.contains("max_tokens") == true)
        // Greedy ignores top-k, so "off" with top_p is fine there.
        #expect(refusal { try chat("{\(user),\"top_k\":0,\"top_p\":0.5,\"temperature\":0}") } == nil)
    }

    @Test func completionPromptsAndTheirLimits() throws {
        func completion(_ json: String) throws -> CompletionRequest {
            try OpenAIRequestDecoder.completion(Data(json.utf8), defaults: defaults)
        }
        #expect(try completion(#"{"prompt":"Once"}"#).prompt == .text("Once"))
        #expect(try completion(#"{"prompt":["Once"]}"#).prompt == .text("Once"))
        #expect(try completion(#"{"prompt":[2,100,200]}"#).prompt == .tokens([2, 100, 200]))
        #expect(try completion(#"{"prompt":"x","stream":true}"#).stream)
        #expect(refusal { try completion(#"{"prompt":["a","b"]}"#) }?.param == "prompt")
        #expect(refusal { try completion(#"{"prompt":"x","echo":true}"#) }?.param == "echo")
        #expect(refusal { try completion(#"{"prompt":"x","best_of":2}"#) }?.param == "best_of")
        #expect(refusal { try completion(#"{"prompt":"x","logprobs":3}"#) }?.param == "logprobs")
        #expect(refusal { try completion(#"{"prompt":{"a":1}}"#) }?.status == 400)
    }

    @Test func maxTokensIsClampedToTheContext() {
        let parameters = CompletionParameters(
            maxTokens: 10_000, sampling: SamplingParameters(temperature: 0, topK: nil, topP: nil), stop: ["x"])
        let config = parameters.generationConfig(promptTokens: 100, maxContext: 1024)
        #expect(config.maxNewTokens == 924)
        #expect(config.isPureGreedy)
        #expect(config.stopStrings == ["x"])
        let open = CompletionParameters(maxTokens: nil, sampling: parameters.sampling, stop: [])
        #expect(open.generationConfig(promptTokens: 1000, maxContext: 1024).maxNewTokens == 24)
    }

    @Test func bearerKeysCompareExactly() {
        func request(_ authorization: String?) -> HTTPRequest {
            var headers = HTTPHeaders()
            if let authorization { headers.add("Authorization", authorization) }
            return HTTPRequest(method: "GET", target: "/v1/models", version: .http11, headers: headers, body: Data())
        }
        #expect(OpenAIService.authorized(request(nil), apiKey: nil))
        #expect(OpenAIService.authorized(request("Bearer sk-1"), apiKey: "sk-1"))
        #expect(OpenAIService.authorized(request("bearer sk-1"), apiKey: "sk-1"))
        #expect(!OpenAIService.authorized(request(nil), apiKey: "sk-1"))
        #expect(!OpenAIService.authorized(request("Bearer sk-12"), apiKey: "sk-1"))
        #expect(!OpenAIService.authorized(request("Bearer sk-"), apiKey: "sk-1"))
        #expect(!OpenAIService.authorized(request("Basic sk-1"), apiKey: "sk-1"))
    }
}
