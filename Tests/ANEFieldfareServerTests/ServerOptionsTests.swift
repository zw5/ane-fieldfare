import Testing
import ANEFieldfareHotpathCLICore
import ANEFieldfareServerCore

/// The server's flags, through the parser it shares with the CLI.
@Suite struct ServerOptionsTests {
    @Test func flagsAndTheirLimits() throws {
        let options = try ServerOptions.parse(
            ["--model", "m", "--port", "0", "--max-context", "16", "--temperature", "0", "--top-k", "0",
             "--top-p", "0.5", "--max-tokens", "9"], environment: [:])
        #expect(options.port == 0 && options.maxContext == 16 && options.defaults.maxTokens == 9)
        #expect(options.defaults.topK == nil && options.defaults.topP == 0.5)
        #expect(try ServerOptions.parse(["--model", "m"], environment: [:]).defaults == SamplingDefaults())
        #expect(throws: ArgsError.helpRequested) { _ = try ServerOptions.parse(["-h"], environment: [:]) }
        #expect(throws: ArgsError.requiredMissing("--model")) { _ = try ServerOptions.parse([], environment: [:]) }
        for (flag, value) in [("--max-context", "15"), ("--temperature", "inf"), ("--top-k", "257"),
                              ("--host", ""), ("--port", "70000")] {
            #expect(throws: ArgsError.invalidValue(flag: flag, value: value)) {
                _ = try ServerOptions.parse(["--model", "m", flag, value], environment: [:])
            }
        }
        #expect(throws: ArgsError.invalidValue(flag: "--top-p", value: "0.5 requires --top-k between 1 and 256")) {
            _ = try ServerOptions.parse(["--model", "m", "--top-k", "0", "--top-p", "0.5"], environment: [:])
        }
    }
}
