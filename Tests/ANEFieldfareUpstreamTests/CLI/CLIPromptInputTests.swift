import Testing
import Foundation
import ANEFieldfareHotpath
@testable import ANEFieldfareHotpathCLICore

/// How the CLI turns image flags and messages files into a prompt.
@Suite struct CLIPromptInputTests {
    /// A relative `image_file` belongs to the messages file's directory. The old
    /// code branched on `URL(fileURLWithPath:).path`, which resolves against the
    /// *process* directory and is therefore always absolute — so the relative
    /// branch never ran and every relative path silently resolved against
    /// wherever the CLI happened to be started. `./a.png` is the case a
    /// `hasPrefix` on a `URL.path` could never distinguish.
    @Test func relativeImagePathsResolveAgainstTheMessagesFile() throws {
        let manager = FileManager.default
        let root = manager.temporaryDirectory
            .appendingPathComponent("cli-messages-\(UUID().uuidString)", isDirectory: true)
        try manager.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? manager.removeItem(at: root) }

        let absolute = root.appendingPathComponent("absolute.png")
        let messages = root.appendingPathComponent("messages.json")
        try Data("""
        [{"role":"user","content":[
          {"type":"image_file","path":"pictures/relative.png"},
          {"type":"image_file","path":"./a.png"},
          {"type":"image_file","path":"\(absolute.path)"},
          {"type":"text","text":"what is this?"}]}]
        """.utf8).write(to: messages)

        let args = try Args.parse(["--model", "m.fieldfare", "--messages-file", messages.path])
        guard case .multimodal(_, let images) = try parseInput(args: args) else {
            Issue.record("expected a multimodal input")
            return
        }
        let paths = Set(images.values.map(\.standardizedFileURL.path))
        let expected = ["pictures/relative.png", "a.png", "absolute.png"]
            .map { root.appendingPathComponent($0).standardizedFileURL.path }
        #expect(paths == Set(expected), "image paths did not resolve against the messages file")
    }

    /// `--image` with no prompt flag used to be reported as a conflict with
    /// `--messages-file`, which the user had not passed. The missing flag is
    /// `--chat-prompt`, and `--help` already says so.
    @Test func aBareImageNamesTheMissingChatPromptRatherThanAConflict() {
        #expect(throws: ArgsError.requiredMissing("--chat-prompt")) {
            _ = try Args.parse(["--model", "m.fieldfare", "--image", "/tmp/a.png"])
        }
        // The conflict message is still right when the conflict is real.
        #expect(throws: ArgsError.mutuallyExclusive("--image", "--messages-file")) {
            _ = try Args.parse([
                "--model", "m.fieldfare", "--image", "/tmp/a.png", "--messages-file", "/tmp/m.json",
            ])
        }
    }

    /// `--prefill off` with an image is silently coerced back to chunked deep in
    /// the runtime, while the docs say off disables that path. The run is
    /// correct; saying nothing about it is what made the flag look honoured.
    @Test func anImagePromptSaysWhenItOverridesTheRequestedPrefill() {
        #expect(prefillCoercionNotice(hasImages: true, config: .off)
                == "[prefill coerced to chunked: image prompts require it]")
        #expect(prefillCoercionNotice(
            hasImages: true, config: .production(chunkTokens: 128)) == nil)
        #expect(prefillCoercionNotice(hasImages: false, config: .off) == nil)
    }

    /// Images are planned in prompt order, not dictionary order: Swift derives
    /// that from a per-process hash seed, so the same command line did the work
    /// in a different order on every run and a failure named a different image
    /// each time.
    @Test func imagesAreProcessedInPromptOrder() {
        let ids = (0..<6).map { _ in UUID() }
        var images: [UUID: URL] = [:]
        for (index, id) in ids.enumerated() {
            images[id] = URL(fileURLWithPath: "/tmp/image-\(index).png")
        }
        let messages = [MultimodalMessage(role: .user, content: ids.map { .image(id: $0) })]
        #expect(orderedImageIDs(messages: messages, images: images) == ids)

        // An image the parts never mention is still processed, not dropped.
        let orphan = UUID()
        images[orphan] = URL(fileURLWithPath: "/tmp/orphan.png")
        #expect(orderedImageIDs(messages: messages, images: images) == ids + [orphan])
    }
}
