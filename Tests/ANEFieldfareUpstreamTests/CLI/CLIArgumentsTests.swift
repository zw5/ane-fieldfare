import Foundation
import Testing
import ANEFieldfareHotpath
@testable import ANEFieldfareHotpathCLICore

@Suite struct CLIArgumentsTests {

    /// `auto` only resolves a size where a chunked prefill will run at it:
    /// under `--prefill off` the size is inert, so a resolved number would be
    /// announced on stderr and then used by nothing. A repeated
    /// `--prefill-chunk-tokens` is last-one-wins in both orderings; without
    /// the clear on the integer branch, `auto` survived a later explicit size.
    @Test(arguments: [
        (["--prefill", "on", "--prefill-chunk-tokens", "auto"], true),
        (["--prefill", "off", "--prefill-chunk-tokens", "auto"], false),
        (["--prefill", "on", "--prefill-chunk-tokens", "64"], false),
        (["--prefill", "on", "--prefill-chunk-tokens", "auto", "--prefill-chunk-tokens", "64"], false),
        (["--prefill", "on", "--prefill-chunk-tokens", "64", "--prefill-chunk-tokens", "auto"], true),
    ])
    func autoResolvesOnlyWhenChunkedPrefillWillRun(flags: [String], resolves: Bool) throws {
        let arguments = try Args.parse(["--model", "m.gturbo", "--prompt", "hi"] + flags)
        #expect(arguments.resolvesPrefillChunkAuto == resolves)
        if flags.last == "64" { #expect(arguments.prefillChunkTokens == 64) }
    }

    /// Sampler limits the kernels cannot honour are refused at parse time.
    @Test func samplerLimitsAreEnforced() throws {
        let disabled = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi", "--top-k", "0", "--top-p", "1",
        ])
        #expect(disabled.topK == nil)
        #expect(throws: ArgsError.self) {
            _ = try Args.parse(["--model", "m.gturbo", "--prompt", "hi", "--top-k", "0"])
        }
        #expect(throws: ArgsError.invalidValue(flag: "--top-k", value: "257")) {
            _ = try Args.parse(["--model", "m.gturbo", "--prompt", "hi", "--top-k", "257"])
        }
    }

    @Test func drafterIsFoundBesideTheModelByDefault() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("drafter-location-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("gturbo/gemma4.gturbo").path
        let drafter = root.appendingPathComponent("dflash/\(DrafterLocation.directoryName)")
        let missing = try Args.parse(["--model", model, "--chat-prompt", "hi"])
        #expect(missing.drafter == nil && missing.drafterOrigin == Args.DrafterOrigin.missing)

        try FileManager.default.createDirectory(at: drafter, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: drafter.appendingPathComponent("config.json"))
        let found = try Args.parse(["--model", model, "--chat-prompt", "hi"])
        #expect(found.drafter == drafter.standardizedFileURL.path && found.drafterOrigin == Args.DrafterOrigin.found)
        let off = try Args.parse(["--model", model, "--chat-prompt", "hi", "--drafter", "off"])
        #expect(off.drafter == nil && off.drafterOrigin == Args.DrafterOrigin.off)
        // Requests speculation cannot serve decode plainly instead of failing.
        for extra in [["--repetition-penalty", "1.1"], ["--image", "x.png"], ["--prefill", "off"]] {
            let plain = try Args.parse(["--model", model, "--chat-prompt", "hi"] + extra)
            #expect(plain.drafter == nil && plain.drafterOrigin == Args.DrafterOrigin.off)
        }
    }

    @Test func drafterIsTextOnlyAndTakesNoRepetitionPenalty() throws {
        // Sampled speculative decoding draws with the target's noise.
        let sampled = try Args.parse(["--model", "m.gturbo", "--chat-prompt", "hi", "--drafter", "d"])
        #expect(sampled.drafter == "d" && sampled.temperature > 0)
        #expect(throws: ArgsError.self) {
            _ = try Args.parse(["--model", "m.gturbo", "--chat-prompt", "hi", "--drafter", "d",
                                "--repetition-penalty", "1.1"])
        }
        #expect(throws: ArgsError.mutuallyExclusive("--drafter", "--image")) {
            _ = try Args.parse(["--model", "m.gturbo", "--chat-prompt", "hi", "--drafter", "d",
                                "--image", "x.png"])
        }
        #expect(throws: ArgsError.mutuallyExclusive("--drafter", "--prefill off")) {
            _ = try Args.parse(["--model", "m.gturbo", "--chat-prompt", "hi", "--drafter", "d",
                                "--prefill", "off"])
        }
    }

    @Test func unsupportedRuntimeOptionValuesAreRejected() {
        for (flag, value) in [("--prefill", "yes"), ("--prefill-chunk-tokens", "8192")] {
            #expect(throws: ArgsError.invalidValue(flag: flag, value: value)) {
                _ = try Args.parse(["--model", "m.gturbo", "--prompt", "hi", flag, value])
            }
        }
        // Programmatic arguments cannot reach the runtime's preconditions either.
        var arguments = Args(model: "m.gturbo", prompt: "hi")
        arguments.prefillChunkTokens = 8192
        #expect(throws: ArgsError.invalidValue(flag: "--prefill-chunk-tokens", value: "8192")) {
            _ = try arguments.resolvedRuntimeConfiguration(forceLogitsHead: false)
        }
    }

    /// The flags the CLI shares with the server follow the server's rules too.
    @Test func sharedFlagsFollowOneRule() throws {
        let base = ["--model", "m.gturbo", "--prompt", "hi"]
        #expect(throws: ArgsError.helpRequested) { _ = try Args.parse(base + ["-h"]) }
        #expect(try Args.parse(base + ["--max-context", "16"]).maxContext == 16)
        for (flag, value) in [("--max-context", "15"), ("--temperature", "inf"), ("--temperature", "nan"),
                              ("--repetition-penalty", "inf")] {
            #expect(throws: ArgsError.invalidValue(flag: flag, value: value)) {
                _ = try Args.parse(base + [flag, value])
            }
        }
        #expect(throws: ArgsError.unknownFlag("--prefill-chunk")) {
            _ = try Args.parse(base + ["--prefill-chunk", "256"])
        }
    }

    @Test(arguments: [
        (["--prompt", "X", "--messages-file", "Y.json"], ArgsError.mutuallyExclusive("--prompt", "--messages-file")),
        (["--chat-prompt", "X", "--messages-file", "Y.json"],
         ArgsError.mutuallyExclusive("--chat-prompt", "--messages-file")),
        (["--prompt", "X", "--chat-prompt", "Y"], ArgsError.mutuallyExclusive("--prompt", "--chat-prompt")),
        (["--prompt", "X", "--image", "x.png"], ArgsError.mutuallyExclusive("--prompt", "--image")),
    ])
    func promptModesAreMutuallyExclusive(flags: [String], error: ArgsError) {
        #expect(throws: error) {
            _ = try Args.parse(["--model", "m.gturbo"] + flags)
        }
    }
}
