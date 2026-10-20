import Foundation
import Testing
@testable import ANEFieldfareHotpath

@Suite("Tokenizer load coordinator")
struct TokenizerLoadCoordinatorTests {
    @Test("Concurrent loads return equivalent tokenizer metadata")
    func concurrentLoadsShareLoadedTokenizer() async throws {
        let tokenizers = try await withThrowingTaskGroup(of: GFTokenizer.self) { group in
            for _ in 0..<8 {
                group.addTask { try await GFTokenizer.load() }
            }
            var loaded: [GFTokenizer] = []
            for try await tokenizer in group { loaded.append(tokenizer) }
            return loaded
        }

        #expect(tokenizers.count == 8)
        let first = try #require(tokenizers.first)
        for tokenizer in tokenizers {
            #expect(tokenizer.bosID == first.bosID)
            #expect(tokenizer.endOfTurnID == first.endOfTurnID)
            #expect(tokenizer.encode("The capital of France is", addBOS: true)
                    == first.encode("The capital of France is", addBOS: true))
        }
    }

    @Test("Model tokenizer sidecar is discovered only when present")
    func modelTokenizerSidecarIsDiscovered() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("gf-tokenizer-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("model.fieldfare", isDirectory: true)
        let modelTokenizer = model.appendingPathComponent("tokenizer", isDirectory: true)
        try FileManager.default.createDirectory(at: modelTokenizer, withIntermediateDirectories: true)
        #expect(GFTokenizer.tokenizerFolder(forModelDirectory: model, environment: [:]) == nil)

        try Data("{}".utf8).write(to: modelTokenizer.appendingPathComponent("tokenizer.json"))
        #expect(GFTokenizer.tokenizerFolder(forModelDirectory: model) == modelTokenizer.standardizedFileURL)
    }
}
