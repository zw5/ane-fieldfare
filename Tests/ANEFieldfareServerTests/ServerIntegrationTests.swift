import Foundation
import Testing
import ANEFieldfareHotpath
@testable import ANEFieldfareServerCore

/// The whole server over real sockets, with a scripted engine in place of the
/// model: wire framing, the OpenAI shapes, stop strings applied while
/// streaming, the bounded line, and cancellation on disconnect.
@Suite struct ServerIntegrationTests {
    private func post(_ server: ScriptedServer, _ path: String, _ json: String,
                      headers: [String: String] = [:]) async throws -> (Int, [AnyHashable: Any], String) {
        var request = URLRequest(url: try #require(URL(string: server.base + path)))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        for (name, value) in headers { request.setValue(value, forHTTPHeaderField: name) }
        request.httpBody = Data(json.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        let http = try #require(response as? HTTPURLResponse)
        return (http.statusCode, http.allHeaderFields, String(decoding: data, as: UTF8.self))
    }

    private func get(_ server: ScriptedServer, _ path: String,
                     headers: [String: String] = [:]) async throws -> (Int, String) {
        var request = URLRequest(url: try #require(URL(string: server.base + path)))
        for (name, value) in headers { request.setValue(value, forHTTPHeaderField: name) }
        let (data, response) = try await URLSession.shared.data(for: request)
        return (try #require(response as? HTTPURLResponse).statusCode, String(decoding: data, as: UTF8.self))
    }

    private func chatBody(_ extra: String = "") -> String {
        #"{"model":"whatever","messages":[{"role":"user","content":"Say hello."}]"# + extra + "}"
    }

    @Test func modelsAndHealth() async throws {
        let server = try await ScriptedServer.start()
        defer { server.server.stop() }
        for path in ["/v1/models", "/api/v1/models"] {
            let (status, body) = try await get(server, path)
            #expect(status == 200)
            let object = try jsonObject(body)
            let data = try #require(object["data"] as? [[String: Any]])
            #expect(object["object"] as? String == "list")
            #expect(data.first?["id"] as? String == "test-model")
            #expect(data.first?["context_length"] as? Int == 4096)
        }
        #expect(try await get(server, "/v1/models/test-model").0 == 200)
        #expect(try await get(server, "/v1/models/other").0 == 404)
        let (status, body) = try await get(server, "/health")
        #expect(status == 200)
        #expect(try jsonObject(body)["status"] as? String == "ok")
    }

    @Test func nonStreamingChatCompletion() async throws {
        let server = try await ScriptedServer.start()
        defer { server.server.stop() }
        let (status, headers, body) = try await post(server, "/v1/chat/completions", chatBody())
        #expect(status == 200)
        #expect(headers["Access-Control-Allow-Origin"] as? String == "*")
        let object = try jsonObject(body)
        #expect(object["object"] as? String == "chat.completion")
        #expect((object["id"] as? String)?.hasPrefix("chatcmpl-") == true)
        #expect(object["model"] as? String == "test-model")
        #expect(object["created"] is Int)
        let choice = try #require((object["choices"] as? [[String: Any]])?.first)
        let message = try #require(choice["message"] as? [String: Any])
        #expect(message["role"] as? String == "assistant")
        #expect(message["content"] as? String == ScriptedServer.replyText)
        #expect(choice["finish_reason"] as? String == "stop")
        let usage = try #require(object["usage"] as? [String: Int])
        let prompt = try server.backend.prepare(.chat([GFTokenizer.Message(role: .user, content: "Say hello.")]))
        #expect(usage["prompt_tokens"] == prompt.tokenIDs.count)
        // The reply's tokens plus the end-of-turn token.
        let replyTokens = server.tokenizer.encode(ScriptedServer.replyText, addBOS: false).count
        #expect(usage["completion_tokens"] == replyTokens + 1)
        #expect(usage["total_tokens"] == prompt.tokenIDs.count + replyTokens + 1)
    }

    @Test func maxTokensEndsWithLength() async throws {
        let server = try await ScriptedServer.start()
        defer { server.server.stop() }
        let (_, _, body) = try await post(server, "/api/v1/chat/completions", chatBody(#","max_tokens":3"#))
        let object = try jsonObject(body)
        let choice = try #require((object["choices"] as? [[String: Any]])?.first)
        #expect(choice["finish_reason"] as? String == "length")
        #expect((object["usage"] as? [String: Int])?["completion_tokens"] == 3)
        let reply = server.tokenizer.encode(ScriptedServer.replyText, addBOS: false)
        #expect((choice["message"] as? [String: Any])?["content"] as? String
                == server.tokenizer.decode(Array(reply.prefix(3))))
    }

    @Test func streamingChatCompletionFraming() async throws {
        let server = try await ScriptedServer.start()
        defer { server.server.stop() }
        var request = URLRequest(url: try #require(URL(string: server.base + "/v1/chat/completions")))
        request.httpMethod = "POST"
        request.httpBody = Data(chatBody(#","stream":true,"stream_options":{"include_usage":true}"#).utf8)
        let (bytes, response) = try await URLSession.shared.bytes(for: request)
        let http = try #require(response as? HTTPURLResponse)
        #expect(http.statusCode == 200)
        #expect(http.value(forHTTPHeaderField: "Content-Type")?.hasPrefix("text/event-stream") == true)
        var raw = Data()
        for try await byte in bytes { raw.append(byte) }
        let text = String(decoding: raw, as: UTF8.self)
        let events = sseEvents(text)
        #expect(text.hasSuffix("data: [DONE]\n\n"))
        #expect(events.last == "[DONE]")
        let chunks = try events.dropLast().map(jsonObject)
        #expect(chunks.allSatisfy { $0["object"] as? String == "chat.completion.chunk" })
        #expect(Set(chunks.compactMap { $0["id"] as? String }).count == 1, "one id for the whole stream")
        let first = try #require((chunks.first?["choices"] as? [[String: Any]])?.first)
        #expect((first["delta"] as? [String: Any])?["role"] as? String == "assistant")
        #expect(first["finish_reason"] is NSNull)
        let choices = chunks.compactMap { ($0["choices"] as? [[String: Any]])?.first }
        let content = choices.compactMap { ($0["delta"] as? [String: Any])?["content"] as? String }.joined()
        #expect(content == ScriptedServer.replyText)
        #expect(choices.count > 3, "the text arrives in several chunks")
        #expect(choices.compactMap { $0["finish_reason"] as? String } == ["stop"])
        // include_usage: a final chunk with empty choices and the usage;
        // every other chunk says "usage": null.
        let last = try #require(chunks.last)
        #expect((last["choices"] as? [Any])?.isEmpty == true)
        #expect((last["usage"] as? [String: Int])?["completion_tokens"] ?? 0 > 0)
        #expect(chunks.dropLast().allSatisfy { $0["usage"] is NSNull })
    }

    /// A stop string that spans several tokens never leaks a prefix of
    /// itself into the stream: the matcher holds back a possible start until
    /// it is ruled in or out.
    @Test func streamingStopStringSplitAcrossTokens() async throws {
        let server = try await ScriptedServer.start()
        defer { server.server.stop() }
        let reply = server.tokenizer.encode(ScriptedServer.replyText, addBOS: false)
        let pieces = reply.map { server.tokenizer.decode([$0]) }
        // The stop starts inside one token and ends inside a later one.
        let stop = "is a scr"
        #expect(!pieces.contains { $0.contains(stop) }, "the stop must span tokens for this test: \(pieces)")
        let expected = "Hello world, this "
        for path in ["/v1/chat/completions", "/v1/completions"] {
            let body = path.hasSuffix("chat/completions")
                ? chatBody(#","stream":true,"stop":["\#(stop)","never"]"#)
                : #"{"prompt":"Say hello.","stream":true,"stop":"\#(stop)"}"#
            var request = URLRequest(url: try #require(URL(string: server.base + path)))
            request.httpMethod = "POST"
            request.httpBody = Data(body.utf8)
            let (data, _) = try await URLSession.shared.data(for: request)
            let events = sseEvents(String(decoding: data, as: UTF8.self))
            let choices = try events.dropLast().map(jsonObject).compactMap { ($0["choices"] as? [[String: Any]])?.first }
            let deltas = choices.compactMap { choice -> String? in
                (choice["delta"] as? [String: Any])?["content"] as? String ?? choice["text"] as? String
            }
            #expect(deltas.joined() == expected, "\(path)")
            #expect(!deltas.joined().contains("is a"), "\(path)")
            #expect(choices.compactMap { $0["finish_reason"] as? String } == ["stop"], "\(path)")
        }
    }

    @Test func rawCompletionsStream() async throws {
        let server = try await ScriptedServer.start()
        defer { server.server.stop() }
        let (status, _, body) = try await post(server, "/v1/completions", #"{"prompt":"Say hello.","stream":true}"#)
        #expect(status == 200)
        let chunks = try sseEvents(body).dropLast().map(jsonObject)
        #expect(chunks.allSatisfy { $0["object"] as? String == "text_completion" })
        #expect((chunks.first?["id"] as? String)?.hasPrefix("cmpl-") == true)
        let text = chunks.compactMap { ($0["choices"] as? [[String: Any]])?.first?["text"] as? String }.joined()
        #expect(text == ScriptedServer.replyText)
        let (_, _, whole) = try await post(server, "/v1/completions", #"{"prompt":[2,1000,2000]}"#)
        let choice = try #require((try jsonObject(whole)["choices"] as? [[String: Any]])?.first)
        #expect(choice["text"] as? String == ScriptedServer.replyText)
        #expect(choice["logprobs"] is NSNull)
    }

    @Test func errorsHaveStatusesAndShapes() async throws {
        let server = try await ScriptedServer.start(apiKey: "sk-test", maxContext: 32)
        defer { server.server.stop() }
        let auth = ["Authorization": "Bearer sk-test"]
        func error(_ body: String) throws -> [String: Any] {
            try #require(try jsonObject(body)["error"] as? [String: Any])
        }
        // Health and the chat page need no key; the API does.
        #expect(try await get(server, "/health").0 == 200)
        let (page, pageBody) = try await get(server, "/")
        #expect(page == 200 && pageBody.contains("<title>Fieldfare Chat</title>"))
        #expect(pageBody.contains("/v1/chat/completions"))
        let (unauthorized, unauthorizedBody) = try await get(server, "/v1/models")
        #expect(unauthorized == 401)
        #expect(try error(unauthorizedBody)["code"] as? String == "invalid_api_key")
        #expect(try await get(server, "/v1/models", headers: ["Authorization": "Bearer wrong"]).0 == 401)
        #expect(try await get(server, "/v1/models", headers: auth).0 == 200)

        let (missing, missingBody) = try await get(server, "/v1/nothing", headers: auth)
        #expect(missing == 404)
        #expect(try error(missingBody)["type"] as? String == "invalid_request_error")
        #expect(try await get(server, "/v1/chat/completions", headers: auth).0 == 405)

        let (bad, _, badBody) = try await post(server, "/v1/chat/completions", "{not json", headers: auth)
        #expect(bad == 400)
        #expect(try error(badBody)["code"] as? String == "invalid_json")
        let (tooLong, _, tooLongBody) = try await post(
            server, "/v1/chat/completions",
            #"{"messages":[{"role":"user","content":"\#(String(repeating: "word ", count: 40))"}]}"#, headers: auth)
        #expect(tooLong == 413)
        #expect(try error(tooLongBody)["code"] as? String == "context_length_exceeded")
        let (image, _, imageBody) = try await post(server, "/v1/chat/completions", """
            {"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA"}}]}]}
            """, headers: auth)
        #expect(image == 400)
        #expect(try error(imageBody)["code"] as? String == "unsupported_content")
    }

    @Test func corsPreflight() async throws {
        let server = try await ScriptedServer.start(apiKey: "sk-test")
        defer { server.server.stop() }
        let client = RawClient(port: server.port)
        defer { client.close() }
        try await client.send("OPTIONS /api/v1/chat/completions HTTP/1.1\r\nHost: x\r\nOrigin: http://localhost:3000\r\n"
                              + "Access-Control-Request-Method: POST\r\n"
                              + "Access-Control-Request-Headers: authorization, content-type, x-title\r\n\r\n")
        let (head, body) = try await client.readResponse()
        #expect(head.hasPrefix("HTTP/1.1 204 No Content\r\n"))
        #expect(head.contains("Access-Control-Allow-Origin: *\r\n"))
        #expect(head.contains("Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"))
        #expect(head.contains("Access-Control-Allow-Headers: authorization, content-type, x-title\r\n"))
        #expect(body.isEmpty)
    }

    @Test func keepAliveChunkedBodiesAndContinue() async throws {
        let server = try await ScriptedServer.start()
        defer { server.server.stop() }
        let client = RawClient(port: server.port)
        defer { client.close() }
        let json = chatBody()
        // 1: Content-Length body.
        try await client.send("POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                              + "Content-Length: \(json.utf8.count)\r\n\r\n\(json)")
        let first = try await client.readResponse()
        #expect(first.head.hasPrefix("HTTP/1.1 200 OK"))
        #expect(first.head.contains("Connection: keep-alive"))
        #expect(first.body.contains(ScriptedServer.replyText))
        // 2: chunked request body on the same connection.
        let half = json.utf8.count / 2
        let a = String(json.prefix(half))
        let b = String(json.dropFirst(half))
        try await client.send("POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n"
                              + "\(String(a.utf8.count, radix: 16))\r\n\(a)\r\n"
                              + "\(String(b.utf8.count, radix: 16))\r\n\(b)\r\n0\r\n\r\n")
        let second = try await client.readResponse()
        #expect(second.head.hasPrefix("HTTP/1.1 200 OK"))
        #expect(second.body.contains(ScriptedServer.replyText))
        // 3: Expect: 100-continue, then Connection: close.
        try await client.send("POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nExpect: 100-continue\r\n"
                              + "Connection: close\r\nContent-Length: \(json.utf8.count)\r\n\r\n")
        #expect(try await client.read(through: "\r\n\r\n") == "HTTP/1.1 100 Continue\r\n\r\n")
        try await client.send(json)
        let third = try await client.readResponse()
        #expect(third.head.contains("Connection: close"))
        #expect(third.body.contains(ScriptedServer.replyText))
        #expect(try await client.readsEOF())
    }

    @Test func streamingOverHTTP10IsCloseDelimited() async throws {
        let server = try await ScriptedServer.start()
        defer { server.server.stop() }
        let client = RawClient(port: server.port)
        defer { client.close() }
        let json = chatBody(#","stream":true"#)
        try await client.send("POST /v1/chat/completions HTTP/1.0\r\nContent-Length: \(json.utf8.count)\r\n\r\n\(json)")
        let head = try await client.read(through: "\r\n\r\n")
        #expect(!head.contains("Transfer-Encoding"))
        #expect(head.contains("Connection: close"))
        let stream = try await client.read(through: "data: [DONE]\n\n")
        #expect(stream.hasPrefix("data: {"))
        #expect(try await client.readsEOF())
    }

    @Test func malformedRequestGets400AndClose() async throws {
        let server = try await ScriptedServer.start()
        defer { server.server.stop() }
        let client = RawClient(port: server.port)
        defer { client.close() }
        try await client.send("POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 5\r\nTransfer-Encoding: chunked\r\n\r\n")
        let (head, body) = try await client.readResponse()
        #expect(head.hasPrefix("HTTP/1.1 400 Bad Request"))
        #expect(body.contains(#""error""#))
        #expect(try await client.readsEOF())
    }

    /// A client that disconnects mid-stream cancels its generation; the
    /// engine is reset and serves the next request in full.
    @Test func disconnectCancelsTheGeneration() async throws {
        let server = try await ScriptedServer.start(repeats: true, delay: .milliseconds(20))
        defer { server.server.stop() }
        let client = RawClient(port: server.port)
        let json = chatBody(#","stream":true"#)
        try await client.send("POST /v1/chat/completions HTTP/1.1\r\nContent-Length: \(json.utf8.count)\r\n\r\n\(json)")
        _ = try await client.read(through: #""content":"Hello""#)
        client.close()
        #expect(await server.waitUntil { $0.cancelled == 1 }, "the generation was not cancelled: \(server.probe.counts)")
        #expect(server.probe.counts.finished == 0)
    }

    /// One generates, `maxWaiting` wait, the rest get 503; a waiting client
    /// that leaves gives its place up without ever starting.
    @Test func boundedLineAndQueuedCancellation() async throws {
        let server = try await ScriptedServer.start(repeats: true, delay: .milliseconds(20), maxWaiting: 1)
        defer { server.server.stop() }
        let json = chatBody(#","stream":true"#)
        let request = "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: \(json.utf8.count)\r\n\r\n\(json)"
        let running = RawClient(port: server.port)
        try await running.send(request)
        _ = try await running.read(through: #""content":"Hello""#)
        let waiting = RawClient(port: server.port)
        try await waiting.send(request)
        let deadline = ContinuousClock.now + .seconds(5)
        while server.backend.queueStatus.waiting < 1, ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(server.backend.queueStatus == GenerationGate.Status(running: true, waiting: 1))

        let (status, headers, body) = try await post(server, "/v1/chat/completions", chatBody())
        #expect(status == 503)
        #expect(headers["Retry-After"] as? String == "1")
        #expect(try (jsonObject(body)["error"] as? [String: Any])?["code"] as? String == "server_busy")

        waiting.close()
        while server.backend.queueStatus.waiting > 0, ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(server.backend.queueStatus.waiting == 0)
        running.close()
        #expect(await server.waitUntil { $0.cancelled == 1 })
        #expect(server.probe.counts.started == 1, "the request that left the line never started")
    }
}
