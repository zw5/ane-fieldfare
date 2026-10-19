import Foundation
import Synchronization
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
        let cliText = try tokenizer.applyChatTemplate(cliMessages)
        let cliIDs = tokenizer.encode(cliText, addBOS: false)

        let request = try OpenAIRequestDecoder.chat(
            Data(#"{"messages":\#(conversation)}"#.utf8), defaults: SamplingDefaults())
        guard case .chat(let serverMessages) = request.prompt else {
            Issue.record("the server did not read a chat")
            return
        }
        #expect(serverMessages == cliMessages)
        #expect(try tokenizer.applyChatTemplate(serverMessages) == cliText)
        #expect(try backend(tokenizer).prepare(request.prompt).tokenIDs == cliIDs)
        #expect(cliText.hasSuffix("<|turn>user\nAnd of Italy?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"))
        #expect(cliText.contains("<|turn>model\n<|channel>thought\n<channel|>Paris.<turn|>\n"))
    }

    /// A single user message equals `--chat-prompt`.
    @Test func singleTurnMatchesTheCLIChatPrompt() async throws {
        let tokenizer = try await GFTokenizer.load()
        let args = try Args.parse(["--model", "m.gturbo", "--chat-prompt", "Name three colors."])
        guard case .messages(let cliMessages) = try parseInput(args: args) else {
            Issue.record("the CLI did not read a text prompt")
            return
        }
        let request = try OpenAIRequestDecoder.chat(
            Data(#"{"messages":[{"role":"user","content":"Name three colors."}]}"#.utf8), defaults: SamplingDefaults())
        let expected = tokenizer.encode(try tokenizer.applyChatTemplate(cliMessages), addBOS: false)
        #expect(try backend(tokenizer).prepare(request.prompt).tokenIDs == expected)
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

@Suite struct GenerationGateTests {
    @Test func grantsInOrderAndRefusesWhenTheLineIsFull() async throws {
        let gate = GenerationGate(maxWaiting: 2)
        try await gate.acquire()
        #expect(gate.status == GenerationGate.Status(running: true, waiting: 0))
        let order = OrderLog()
        let first = Task { try await gate.acquire(); order.append(1); gate.release() }
        while gate.status.waiting < 1 { await Task.yield() }
        let second = Task { try await gate.acquire(); order.append(2); gate.release() }
        while gate.status.waiting < 2 { await Task.yield() }
        await #expect(throws: GenerationGate.Busy.self) { try await gate.acquire() }
        gate.release()
        try await first.value
        try await second.value
        #expect(order.values == [1, 2])
        #expect(gate.status == GenerationGate.Status(running: false, waiting: 0))
    }

    @Test func aCancelledWaiterLeavesTheLine() async throws {
        let gate = GenerationGate(maxWaiting: 1)
        try await gate.acquire()
        let waiter = Task { try await gate.acquire() }
        while gate.status.waiting < 1 { await Task.yield() }
        waiter.cancel()
        await #expect(throws: CancellationError.self) { try await waiter.value }
        #expect(gate.status == GenerationGate.Status(running: true, waiting: 0))
        gate.release()
        #expect(gate.status.running == false)
        // A task cancelled before it asks is refused without queueing.
        let late = Task {
            unsafe withUnsafeCurrentTask { unsafe $0?.cancel() }
            try await gate.acquire()
        }
        await #expect(throws: CancellationError.self) { try await late.value }
        #expect(gate.status == GenerationGate.Status(running: false, waiting: 0))
    }
}

private final class OrderLog: Sendable {
    private let state = Mutex<[Int]>([])
    var values: [Int] { state.withLock { $0 } }
    func append(_ value: Int) { state.withLock { $0.append(value) } }
}
