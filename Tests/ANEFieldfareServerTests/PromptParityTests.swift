import Foundation
import Testing
import ANEFieldfareHotpath
@testable import ANEFieldfareHotpathCLICore
@testable import ANEFieldfareServerCore

/// An engine that is never asked to generate: `prepare` alone is under test.
private final class IdleEngine: CompletionEngine {
    struct Unused: Error {}
    func complete(promptIDs: [Int32], config: GenerationConfig,
                  onProgress: (RawDecodeProgress) -> Void) async throws -> EngineCompletion {
        throw Unused()
    }
}

/// The server's prompts are the CLI's prompts: the same chat template over
/// the same turns, the same raw-prompt encoding, token for token.
@Suite struct PromptParityTests {
    private func backend(_ tokenizer: GFTokenizer, maxContext: Int = 32_768) -> SerializedBackend<IdleEngine> {
        SerializedBackend(engine: IdleEngine(), tokenizer: tokenizer, maxContext: maxContext, maxWaiting: 0)
    }

    /// Multi-turn chat through the API equals the CLI's `--messages-file`
    /// rendering of the same conversation.
    @Test func multiTurnChatMatchesTheCLIMessagesFile() async throws {
        let tokenizer = try await GFTokenizer.load()
        let conversation = """
        [{"role":"system","content":"You are terse."},
         {"role":"user","content":"What is the capital of France?"},
         {"role":"assistant","content":"Paris."},
         {"role":"user","content":[{"type":"text","text":"And of "},{"type":"text","text":"Italy?"}]}]
        """
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("server-parity-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("messages.json")
        try Data(conversation.utf8).write(to: file)

        let args = try Args.parse(["--model", "m.gturbo", "--messages-file", file.path])
        guard case .messages(let cliMessages) = try parseInput(args: args) else {
            Issue.record("the CLI did not read a text conversation")
            return
        }
        let cliIDs = tokenizer.encode(try tokenizer.applyChatTemplate(cliMessages), addBOS: false)

        let request = try OpenAIRequestDecoder.chat(
            Data(#"{"messages":\#(conversation)}"#.utf8), defaults: SamplingDefaults())
        guard case .chat(let serverMessages) = request.prompt else {
            Issue.record("the server did not read a chat")
            return
        }
        #expect(serverMessages == cliMessages)
        #expect(try backend(tokenizer).prepare(request.prompt).tokenIDs == cliIDs)
    }

    /// `/v1/completions` encodes like `--prompt`: `<bos>` and the text.
    @Test func rawPromptMatchesTheCLIPrompt() async throws {
        let tokenizer = try await GFTokenizer.load()
        let text = "The three largest planets are"
        #expect(try backend(tokenizer).prepare(.text(text)).tokenIDs == tokenizer.encode(text, addBOS: true))
        #expect(try backend(tokenizer).prepare(.tokens([5, 6])).tokenIDs == [5, 6])
        #expect(throws: APIError.self) { try backend(tokenizer).prepare(.tokens([-1])) }
        #expect(throws: APIError.self) { try backend(tokenizer).prepare(.text("")) }
        let error = #expect(throws: APIError.self) { try backend(tokenizer, maxContext: 4).prepare(.text(text)) }
        #expect(error?.status == 413)
    }
}
