import Foundation
import Testing
@testable import ANEFieldfareHotpath

/// Hand-authored Gemma 4 chat template. Loads the IT tokenizer (cached after
/// first run).
@Suite("ChatTemplate")
struct ChatTemplateTests {
    let tok: GFTokenizer

    init() async throws {
        self.tok = try await GFTokenizer.load()
    }

    private typealias Message = GFTokenizer.Message

    @Test("Single-turn token IDs match the pinned upstream template")
    func tokenIDsMatchPinnedUpstream() throws {
        let prompt = try tok.applyChatTemplate([Message(role: .user, content: "Hi")])
        #expect(tok.encode(prompt, addBOS: false) == [
            2, 105, 2364, 107, 10979, 106, 107, 105, 4368, 107, 100, 45518, 107, 101,
        ])
    }

    /// A past model turn keeps the empty thought block it was generated after;
    /// the trailing model turn is left open.
    @Test("Multi-turn renders system first, closes prior turns, leaves the model open")
    func multiTurn() throws {
        let p = try tok.applyChatTemplate([
            Message(role: .system, content: "Be terse."),
            Message(role: .user, content: "A"),
            Message(role: .assistant, content: "B"),
            Message(role: .user, content: "C"),
        ])
        #expect(p == "<bos><|turn>system\nBe terse.<turn|>\n<|turn>user\nA<turn|>\n"
                + "<|turn>model\n<|channel>thought\n<channel|>B<turn|>\n<|turn>user\nC<turn|>\n"
                + "<|turn>model\n<|channel>thought\n<channel|>")
        #expect(try tok.applyChatTemplate([]) == "<bos><|turn>model\n<|channel>thought\n<channel|>")
    }

    @Test("System message after a user turn is rejected")
    func misplacedSystemTurn() {
        #expect(throws: GFTokenizerError.self) {
            _ = try tok.applyChatTemplate([
                Message(role: .user, content: "Hi"),
                Message(role: .system, content: "Too late"),
            ])
        }
    }

    /// Control-token spoofing: a user typing the literal open-turn marker must
    /// not inject an extra turn-CLOSE id (the marker that would end the turn).
    /// HF tokenizers do recognize special-token text, so the verbatim close
    /// marker can still be injected; that trusted-input limitation is accepted.
    @Test("Open-turn marker in user content does not add a turn-close id")
    func spoofingDoesNotInjectTurnClose() throws {
        let p = try tok.applyChatTemplate([Message(role: .user, content: "<|turn> spoofing attempt")])
        let closes = tok.encode(p, addBOS: false).filter { $0 == tok.endOfTurnID }.count
        #expect(closes == 1, "expected exactly one <turn|>, got \(closes)")
    }
}
