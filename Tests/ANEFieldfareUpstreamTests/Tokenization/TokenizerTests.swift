import Foundation
import Testing
import Tokenizers
@testable import ANEFieldfareHotpath

/// First run downloads the Gemma 4 IT tokenizer (~32 MB) from Hugging Face
/// Hub to `~/.cache/huggingface/`. Subsequent runs are offline.
@Suite("Tokenizer")
struct TokenizerTests {
    let tok: GFTokenizer

    init() async throws {
        self.tok = try await GFTokenizer.load()
    }

    // MARK: - Special tokens

    @Test("Marker resolution rejects unk substitution")
    func markerResolutionRejectsUnk() throws {
        #expect(throws: GFTokenizerError.self) {
            _ = try GFTokenizer.requireTokenID(tok.tokenizer, "<no-such-token-xyzzy>")
        }
        #expect(try GFTokenizer.requireTokenID(tok.tokenizer, "<pad>") == 0)
    }

    @Test("Special-token ID set matches the library's skip filter")
    func specialTokenSetMatchesLibrary() throws {
        // A special token decodes to nothing when skipped and to its literal
        // text when kept.
        var probes = randomIDs(count: 64, seed: 0xDEAD_BEEF_CAFE_F00D)
        probes += [tok.bosID, tok.eosID, try GFTokenizer.requireTokenID(tok.tokenizer, "<pad>"), tok.endOfTurnID,
                   tok.toolCallStartID, tok.toolCallEndID, tok.toolResponseID,
                   tok.toolResponseEndID, tok.channelStartID, tok.channelEndID]
        for id in probes {
            let skipped = tok.tokenizer.decode(tokens: [Int(id)], skipSpecialTokens: true)
            let kept = tok.tokenizer.decode(tokens: [Int(id)], skipSpecialTokens: false)
            #expect(tok.specialTokenIDs.contains(id) == (skipped.isEmpty && !kept.isEmpty),
                    "id \(id) special-set membership diverged from library semantics")
        }
    }

    // MARK: - Encode / decode

    /// HF's clean_up_tokenization_spaces collapsed spaces before punctuation
    /// ("he said ' ok ' now" became "he said'ok'now"), which broke
    /// decode(encode(x)) == x for a tokenizer that is lossless by construction.
    @Test("Round-trip is lossless", arguments: [
        "code:  let x = 42;  // comment",
        "he said ' ok ' now",
        "a    . b",
        "wait ! really ?",
        "你好，世界。",
        "mixed 漢 and 🦝 and a",
        "Здравствуй",
    ])
    func roundTrip(_ text: String) {
        #expect(tok.decode(tok.encode(text, addBOS: false)) == text)
    }

    @Test("BOS is added when requested and absent otherwise")
    func bosPrepend() {
        let withBOS = tok.encode("x", addBOS: true)
        let without = tok.encode("x", addBOS: false)
        #expect(withBOS.first == tok.bosID && without.first != tok.bosID)
        #expect(Array(withBOS.dropFirst()) == without)
        #expect(tok.encode("", addBOS: false).isEmpty)
        #expect(tok.encode("", addBOS: true) == [tok.bosID])
    }

    @Test("Decode differs from the library only by its cleanup pass")
    func decodeDiffersFromLibraryOnlyByCleanup() throws {
        // swift-transformers as an independent oracle: our decode, run through
        // the cleanup pass we deliberately dropped, must reproduce the library
        // byte for byte. That covers special-token filtering, byte-fallback
        // grouping, and the "▁" mapping. We also diverge from the Swift port on
        // byte-fallback runs (issue #58: invalid runs decode to one U+FFFD per
        // byte, trailing runs are flushed); the fixed seed produces no such
        // runs, and the byte-soup test covers them against the reference.
        var ids = randomIDs(count: 256, seed: 0x2545_F491_4F6C_DD1D)
        ids += [tok.bosID, tok.eosID, tok.endOfTurnID, tok.channelStartID,
                try GFTokenizer.requireTokenID(tok.tokenizer, "<pad>")]
        // The library drops byte-fallback tokens that end a sequence, so end
        // on a regular token.
        if let trailing = tok.tokenizer.convertTokenToId("the") {
            ids.append(Int32(trailing))
        }
        for skipSpecialTokens in [true, false] {
            let mine = tok.decode(ids, skipSpecialTokens: skipSpecialTokens)
            let library = tok.tokenizer.decode(tokens: ids.map(Int.init),
                                               skipSpecialTokens: skipSpecialTokens)
            #expect(Self.huggingFaceCleanUp(mine) == library,
                    "decode diverged from the library beyond the cleanup pass (skipSpecialTokens: \(skipSpecialTokens))")
        }
    }

    // MARK: - Streaming detokenizer

    @Test("Streaming matches batch decode for complex Unicode", arguments: [
        "mixed 漢 and 🦝",
        "ห้ามใช้ในสัปดาห์นี้",
        "नमस्ते दुनिया",
        "e\u{301}",
        "left \u{2067}العَرَبِيَّةُ ١٢٣\u{2069} right",
        "فارسی می\u{200C}خواهم English",
        "ខ្ញុំសរសេរភាសាខ្មែរ English",
        "\u{1100}\u{1161}\u{11A8} 한글 English",
        "a\u{301}\u{323}\u{304} e\u{308}\u{301}",
        "\u{1F44D}\u{1F3FD} \u{263A}\u{FE0F} 1\u{FE0F}\u{20E3} \u{1F1EC}\u{1F1E7} \u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}\u{200D}\u{1F466}",
    ])
    func streamingComplexUnicode(_ text: String) {
        let ids = tok.encode(text, addBOS: false)
        let assembled = stream(ids)
        #expect(assembled == tok.decode(ids))
        #expect(!assembled.unicodeScalars.contains("\u{FFFD}"))
    }

    @Test("Streaming matches batch decode for arbitrary token streams")
    func streamingMatchesBatchForArbitraryIDs() {
        // encode() output only covers sequences the encoder emits; a generating
        // model is under no such constraint, so drive the detokenizer with raw IDs.
        for round in 0..<16 {
            let ids = randomIDs(count: 256, seed: 0x9E37_79B9_7F4A_7C15 &+ UInt64(round))
            #expect(stream(ids) == tok.decode(ids), "round \(round) diverged from batch decode")
        }
    }

    @Test("Whitespace run before punctuation survives streaming")
    func streamingWhitespaceThenPunctuation() throws {
        // Encoder-unreachable but generator-reachable: the normalizer maps " " to
        // "▁", so no encode() call produces a bare whitespace run followed by
        // punctuation, while a model emits it freely in indented markdown and
        // code. HF's cleanup rewrote " ." to "." across that boundary, which made
        // the streaming delta drop the punctuation token silently.
        for (run, mark) in [("▁▁▁▁", "."), ("▁▁", ",")] {
            let ids = [try tokenID(run), try tokenID(mark)]
            #expect(stream(ids) == tok.decode(ids))
            #expect(stream(ids) == String(repeating: " ", count: run.count) + mark)
        }
    }

    // MARK: - Byte-fallback runs

    @Test("Byte run fuses across a skipped special token")
    func byteRunFusesAcrossSkippedSpecial() throws {
        // The library filters skipped special IDs before its decoder chain, so
        // a byte run must fuse ACROSS a skipped special. In keep mode the
        // special is an ordinary token and closes the run instead, leaving two
        // single-byte runs that are invalid on their own - one U+FFFD each.
        let ids: [Int32] = try [
            tokenID("<0xC3>"), tok.eosID, tokenID("<0xA9>"), tokenID("the"),
        ]
        #expect(tok.decode(ids, skipSpecialTokens: true) == "éthe")
        #expect(stream(ids) == "éthe")
        #expect(tok.decode(ids, skipSpecialTokens: false) == "\u{FFFD}<eos>\u{FFFD}the")
    }

    @Test("Invalid byte runs stream replacement characters immediately")
    func invalidByteRunsStreamImmediately() throws {
        // Once a run can no longer become valid UTF-8, every byte - past and
        // future - decodes to exactly one U+FFFD, so the stream emits them
        // live instead of buffering until the next regular token.
        var detok = GFDetokenizer(tokenizer: tok)
        let stray = try tokenID("<0x80>")
        for _ in 0..<64 {
            #expect(detok.push(stray) == "\u{FFFD}")
        }
        #expect(detok.flush() == "")

        detok = GFDetokenizer(tokenizer: tok)
        #expect(try detok.push(tokenID("<0xC3>")) == "")
        #expect(try detok.push(tokenID("<0xFF>")) == "\u{FFFD}\u{FFFD}")

        // E0's first continuation is constrained to A0-BF, so 0x80 poisons the
        // run even though it is a continuation byte.
        detok = GFDetokenizer(tokenizer: tok)
        #expect(try detok.push(tokenID("<0xE0>")) == "")
        #expect(try detok.push(tokenID("<0x80>")) == "\u{FFFD}\u{FFFD}")
    }

    @Test("A barrier marker commits the run as its own delta")
    func barrierMarkerCommitsHeldRun() throws {
        // The generation pipeline must not let a byte run span a channel/tool
        // marker: the held text surfaces as the marker's own delta, so the
        // structured decoder can route it under the pre-switch channel state.
        var detok = GFDetokenizer(tokenizer: tok,
                                  barrierTokenIDs: tok.structuralMarkerIDs)
        for token in ["<0xF0>", "<0x9F>", "<0x98>", "<0x80>"] {
            #expect(try detok.push(tokenID(token)) == "")
        }
        #expect(detok.push(tok.channelEndID) == "😀")
        // Non-barrier specials still fuse: a codepoint split around eos stays
        // whole, preserving decode parity for everything but the markers.
        #expect(try detok.push(tokenID("<0xC3>")) == "")
        #expect(detok.push(tok.eosID) == "")
        #expect(try detok.push(tokenID("<0xA9>")) == "")
        #expect(try detok.push(tokenID("the")) == "éthe")
    }

    @Test("Byte-token soup matches batch decode and reference run semantics")
    func byteSoupMatchesReference() throws {
        // Adversarial byte-run coverage the encoder can never produce: mostly
        // byte tokens with occasional regular tokens, checked against a direct
        // reimplementation of the reference decoder's run semantics
        // (from_utf8 over the whole run, else one U+FFFD per byte, so an
        // incomplete tail invalidates even a valid prefix).
        let byteIDs: [Int32] = try (0...255).map {
            unsafe try tokenID(String(format: "<0x%02X>", $0))
        }
        let theID = try tokenID("the")
        var state: UInt64 = 0xA076_1D64_78BD_642F
        for round in 0..<16 {
            var ids: [Int32] = []
            var expected = ""
            var run: [UInt8] = []
            func closeRun() {
                guard !run.isEmpty else { return }
                expected += String(bytes: run, encoding: .utf8)
                    ?? String(repeating: "\u{FFFD}", count: run.count)
                run.removeAll()
            }
            for _ in 0..<256 {
                let r = Self.xorshift(&state)
                if r % 8 == 0 {
                    closeRun()
                    ids.append(theID)
                    expected += "the"
                } else {
                    let byte = UInt8(truncatingIfNeeded: r >> 8)
                    ids.append(byteIDs[Int(byte)])
                    run.append(byte)
                }
            }
            closeRun()

            #expect(tok.decode(ids) == expected, "round \(round) batch diverged")
            #expect(stream(ids) == expected, "round \(round) streaming diverged")
        }
    }

    // MARK: - Helpers

    private func tokenID(_ token: String) throws -> Int32 {
        try GFTokenizer.requireTokenID(tok.tokenizer, token)
    }

    private func stream(_ ids: [Int32]) -> String {
        var detok = GFDetokenizer(tokenizer: tok)
        var assembled = ""
        for id in ids { assembled += detok.push(id) }
        return assembled + detok.flush()
    }

    /// Deterministic xorshift64 IDs — token streams the encoder cannot produce
    /// but a generating model can.
    private func randomIDs(count: Int, seed: UInt64) -> [Int32] {
        var state = seed
        return (0..<count).map { _ in Int32(Self.xorshift(&state) % UInt64(tok.vocabSize)) }
    }

    private static func xorshift(_ state: inout UInt64) -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }

    /// swift-transformers' `clean_up_tokenization_spaces` pass, reproduced here
    /// so the differential test can pin our only intended difference from the
    /// library to exactly this pass. Not used in production — see `GemmaDecoding`.
    private static func huggingFaceCleanUp(_ text: String) -> String {
        let replacements = [
            (" .", "."), (" ?", "?"), (" !", "!"), (" ,", ","), (" ' ", "'"),
            (" n't", "n't"), (" 'm", "'m"), (" 's", "'s"), (" 've", "'ve"), (" 're", "'re"),
        ]
        return replacements.reduce(text) { partial, rule in
            partial.replacingOccurrences(of: rule.0, with: rule.1)
        }
    }
}
