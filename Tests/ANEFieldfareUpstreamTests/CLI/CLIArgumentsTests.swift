import Testing
import ANEFieldfareHotpath
@testable import ANEFieldfareHotpathCLICore

@Suite struct CLIArgumentsTests {
    @Test func defaultsUseProductionGenerationValues() throws {
        let arguments = try Args.parse(["--model", "m.gturbo", "--prompt", "hi"])
        #expect(arguments.model == "m.gturbo")
        #expect(arguments.prompt == "hi")
        #expect(arguments.messagesFile == nil)
        #expect(arguments.maxNew == 1_024)
        // ANE Fieldfare retains its 32K context default.
        #expect(arguments.maxContext == 32_768)
        #expect(arguments.temperature == 0.2)
        #expect(arguments.topK == 64)
        #expect(arguments.topP == 0.95)
        #expect(arguments.repetitionPenalty == 1)
        #expect(arguments.seed == nil)
        #expect(arguments.stops.isEmpty)
        #expect(!arguments.quiet)

        let runtime = try arguments.resolvedRuntimeConfiguration(forceLogitsHead: false)
        #expect(runtime == RuntimeConfiguration.production)
    }

    /// `auto` only resolves a size where a chunked prefill will run at it.
    /// Under `--prefill off` the config is `.off` and the size is inert, and an
    /// image turn there is coerced to the runtime's own chunked default, so the
    /// resolved number would be announced on stderr and then used by nothing.
    @Test(arguments: [
        (prefill: "on", auto: true, resolves: true),
        (prefill: "off", auto: true, resolves: false),
        (prefill: "on", auto: false, resolves: false),
        (prefill: "off", auto: false, resolves: false),
    ])
    func autoResolvesOnlyWhenChunkedPrefillWillRun(
        testCase: (prefill: String, auto: Bool, resolves: Bool)
    ) throws {
        let arguments = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--prefill", testCase.prefill,
            "--prefill-chunk-tokens", testCase.auto ? "auto" : "64",
        ])
        #expect(arguments.prefillChunkTokensAuto == testCase.auto)
        #expect(arguments.resolvesPrefillChunkAuto == testCase.resolves)
    }

    /// A repeated `--prefill-chunk-tokens` is last-one-wins in both orderings.
    /// Without the clear on the integer branch, `auto` survives a later
    /// explicit size and the run resolves its own size over the one asked for.
    @Test func lastPrefillChunkTokensFlagWinsInBothOrderings() throws {
        let explicitLast = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--prefill-chunk-tokens", "auto",
            "--prefill-chunk-tokens", "64",
        ])
        #expect(explicitLast.prefillChunkTokens == 64)
        #expect(explicitLast.prefillChunkTokensAuto == false)
        #expect(explicitLast.resolvesPrefillChunkAuto == false)

        let autoLast = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--prefill-chunk-tokens", "64",
            "--prefill-chunk-tokens", "auto",
        ])
        #expect(autoLast.prefillChunkTokensAuto)
        #expect(autoLast.resolvesPrefillChunkAuto)
    }

    @Test func generationOptionsParseAndStopsRepeat() throws {
        let arguments = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--max-new", "32", "--max-context", "512",
            "--temperature", "0", "--top-k", "40", "--top-p", "0.95",
            "--repetition-penalty", "1.1", "--seed", "42",
            "--stop", "A", "--stop", "B", "--quiet",
        ])
        #expect(arguments.maxNew == 32)
        #expect(arguments.maxContext == 512)
        #expect(arguments.temperature == 0)
        #expect(arguments.topK == 40)
        #expect(arguments.topP == 0.95)
        #expect(arguments.repetitionPenalty == 1.1)
        #expect(arguments.seed == 42)
        #expect(arguments.stops == ["A", "B"])
        #expect(arguments.quiet)
    }

    @Test func topKZeroRequiresTopPToBeDisabled() throws {
        let disabled = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--top-k", "0", "--top-p", "1",
        ])
        #expect(disabled.topK == nil)
        #expect(disabled.topP == 1)

        #expect(throws: ArgsError.self) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi", "--top-k", "0",
            ])
        }
    }

    @Test func topKAboveKernelLimitRejected() {
        #expect(throws: ArgsError.invalidValue(flag: "--top-k", value: "257")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi", "--top-k", "257",
            ])
        }
    }

    @Test func drafterIsGreedyTextOnly() throws {
        let parsed = try Args.parse([
            "--model", "m.gturbo", "--chat-prompt", "hi", "--temperature", "0", "--drafter", "d",
        ])
        #expect(parsed.drafter == "d")
        #expect(throws: ArgsError.self) {
            _ = try Args.parse(["--model", "m.gturbo", "--chat-prompt", "hi", "--drafter", "d"])
        }
        #expect(throws: ArgsError.mutuallyExclusive("--drafter", "--image")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--chat-prompt", "hi", "--temperature", "0",
                "--drafter", "d", "--image", "x.png",
            ])
        }
        #expect(throws: ArgsError.mutuallyExclusive("--drafter", "--prefill off")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--chat-prompt", "hi", "--temperature", "0",
                "--drafter", "d", "--prefill", "off",
            ])
        }
    }

    @Test func helpListsExactlyThePublicOptions() {
        let expected: Set<String> = [
            "--model", "--prompt", "--messages-file", "--max-new", "--max-context",
            "--temperature", "--top-k", "--top-p", "--repetition-penalty",
            "--seed", "--stop", "--quiet",
            "--prefill", "--prefill-chunk-tokens",
            "--help", "--backend", "--fault-policy", "--prompt-file",
            "--count-tokens", "--verify-weights", "--expert-top-k", "--prefill-expert-cap",
            "--chat-prompt", "--image", "--vision-pack", "--vision-residency",
            "--early-expert-read", "--drafter",
        ]
        let words = Args.usage.split { $0.isWhitespace || $0 == "(" || $0 == ")" }
        let options = Set(words.map(String.init).filter { $0.hasPrefix("--") })
        #expect(options == expected)
    }

    @Test func runtimeOptionsReachTypedConfiguration() throws {
        let arguments = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--prefill", "off",
            "--prefill-chunk-tokens", "64",
        ])

        #expect(arguments.prefillPolicy == .off)
        #expect(arguments.prefillChunkTokens == 64)

        let runtime = try arguments.resolvedRuntimeConfiguration(forceLogitsHead: true)
        #expect(runtime.prefillPolicy == .off)
        #expect(runtime.prefillChunkTokens == 64)
        #expect(runtime.headPath == .logits)
    }

    @Test func everySupportedRuntimeOptionParses() throws {
        for value in RuntimeConfiguration.allowedPrefillChunkTokens {
            let arguments = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi",
                "--prefill-chunk-tokens", "\(value)",
            ])
            #expect(arguments.prefillChunkTokens == value)
        }
        #expect(try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi", "--prefill", "on",
        ]).prefillPolicy == .chunked)
        #expect(try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi", "--prefill", "off",
        ]).prefillPolicy == .off)
    }

    @Test func unsupportedRuntimeOptionValuesAreRejected() {
        let invalidValues = [
            ("--prefill", "yes"),
            ("--prefill-chunk-tokens", "8192"),
        ]
        for (flag, value) in invalidValues {
            #expect(throws: ArgsError.invalidValue(flag: flag, value: value)) {
                _ = try Args.parse([
                    "--model", "m.gturbo", "--prompt", "hi", flag, value,
                ])
            }
        }

    }

    /// The page cache replaced the slot cache, so its knobs are gone.
    @Test func removedExpertCacheFlagsAreUnknown() {
        for (flag, value) in [("--expert-cache-slots", "48"), ("--expert-cache-policy", "lru")] {
            #expect(throws: ArgsError.unknownFlag(flag)) {
                _ = try Args.parse(["--model", "m.gturbo", "--prompt", "hi", flag, value])
            }
        }
    }

    @Test func programmaticArgumentsCannotReachRuntimePreconditions() {
        var arguments = Args(model: "m.gturbo", prompt: "hi")
        arguments.prefillChunkTokens = 8192
        #expect(throws: ArgsError.invalidValue(
            flag: "--prefill-chunk-tokens", value: "8192")) {
            _ = try arguments.resolvedRuntimeConfiguration(forceLogitsHead: false)
        }
    }

    @Test func unsupportedSelectorsAreRejected() {
        for flag in ["--runtime-profile", "--experiment-id", "-h"] {
            #expect(throws: ArgsError.unknownFlag(flag)) {
                _ = try Args.parse(["--model", "m.gturbo", "--prompt", "hi", flag])
            }
        }
    }

    @Test func modelAndPromptAreRequired() {
        #expect(throws: ArgsError.requiredMissing("--model")) {
            _ = try Args.parse(["--prompt", "hi"])
        }
        #expect(throws: ArgsError.modeMissing) {
            _ = try Args.parse(["--model", "m.gturbo"])
        }
    }

    @Test func messagesFileSelectsChatMode() throws {
        let arguments = try Args.parse([
            "--model", "m.gturbo", "--messages-file", "chat.json",
        ])
        #expect(arguments.prompt == nil)
        #expect(arguments.messagesFile == "chat.json")
    }

    @Test func promptAndMessagesFileAreMutuallyExclusive() {
        #expect(throws: ArgsError.mutuallyExclusive("--prompt", "--messages-file")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi",
                "--messages-file", "chat.json",
            ])
        }
    }
    @Test func imageChatOptionsPreserveOrderAndResidency() throws {
        let a = try Args.parse([
            "--model", "m.gturbo",
            "--image", "first.png",
            "--chat-prompt", "compare",
            "--image", "second.jpg",
            "--vision-pack", "vision.gturbo",
            "--vision-residency", "keep-ready",
        ])
        #expect(a.chatPrompt == "compare")
        #expect(a.images == ["first.png", "second.jpg"])
        #expect(a.visionPack == "vision.gturbo")
        #expect(a.visionResidency == .keepReady)
    }

    @Test func imagesRejectRawAndMessagesModesButNotACount() {
        #expect(throws: ArgsError.self) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "raw", "--image", "x.png",
            ])
        }
        #expect(throws: ArgsError.self) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--messages-file", "m.json",
                "--image", "x.png",
            ])
        }
        // No fixed image count: what bounds a request is the context, checked
        // against the rendered prompt once token costs are known. The server
        // moved to a context-derived budget and the CLI follows.
        let many = try? Args.parse([
            "--model", "m.gturbo", "--chat-prompt", "x",
            "--image", "1", "--image", "2", "--image", "3",
            "--image", "4", "--image", "5", "--image", "6",
        ])
        #expect(many?.images.count == 6)
    }

    @Test func mutual_exclusion_chatPromptVsMessagesFile() {
        do {
            _ = try Args.parse([
                "--model", "m.gturbo",
                "--chat-prompt", "X",
                "--messages-file", "Y.json",
            ])
            Issue.record("expected ArgsError.mutuallyExclusive")
        } catch let e as ArgsError {
            #expect(e == .mutuallyExclusive("--chat-prompt", "--messages-file"))
        } catch {
            Issue.record("unexpected error: \(error)")
        }
    }

    @Test func mutual_exclusion_promptVsChatPrompt() {
        do {
            _ = try Args.parse([
                "--model", "m.gturbo",
                "--prompt", "X",
                "--chat-prompt", "Y",
            ])
            Issue.record("expected ArgsError.mutuallyExclusive")
        } catch let e as ArgsError {
            #expect(e == .mutuallyExclusive("--prompt", "--chat-prompt"))
        } catch {
            Issue.record("unexpected error: \(error)")
        }
    }

}
