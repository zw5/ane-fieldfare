import Hub
import Testing
@testable import ANEFieldfareHotpath

/// Structural verification of `tokenizer.json` decoder declarations, driven by
/// `Config` literals — no network and no tokenizer fixture required.
@Suite("Decoder configuration")
struct DecoderConfigTests {
    @Test("Only the pinned Gemma decoder sequence passes")
    func onlyThePinnedSequencePasses() throws {
        let replace: Config = ["type": "Replace", "pattern": ["String": "▁"], "content": " "]
        try GFTokenizer.verifyDecoderConfiguration(["decoder": [
            "type": "Sequence", "decoders": [replace, ["type": "ByteFallback"], ["type": "Fuse"]],
        ]])
        let rejected: [Config] = [
            ["decoder": ["type": "Metaspace", "replacement": "▁", "prependScheme": "first"]],
            ["decoder": ["type": "Sequence", "decoders": [replace, ["type": "ByteFallback"]]]],
            ["decoder": ["type": "Sequence", "decoders": [
                ["type": "Replace", "pattern": ["String": "_"], "content": " "],
                ["type": "ByteFallback"], ["type": "Fuse"],
            ]]],
            ["model": ["type": "BPE"]],
        ]
        for data in rejected {
            #expect(throws: GFTokenizerError.self) { try GFTokenizer.verifyDecoderConfiguration(data) }
        }
    }
}
