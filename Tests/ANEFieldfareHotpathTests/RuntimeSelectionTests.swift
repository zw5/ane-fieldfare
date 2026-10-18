import XCTest
import ANEFieldfareHotpath
import ANEFieldfareHotpathCLICore

final class RuntimeSelectionTests: XCTestCase {
    func testBackendAndPrefillSelection() throws {
        let required = ["--model", "/model", "--prompt", "hello"]
        let defaults = try Args.parse(required)
        XCTAssertEqual(defaults.backend, .metal)
        XCTAssertEqual(defaults.prefillChunkTokens, 1024)
        let hybrid = try Args.parse(required + ["--backend", "hybrid-ane", "--prefill-chunk", "512"])
        XCTAssertEqual(hybrid.backend, .hybridANE)
        XCTAssertEqual(hybrid.prefillChunkTokens, 512)
        XCTAssertThrowsError(try Args.parse(required + ["--backend", "ane-magic"]))
        XCTAssertEqual(defaults.faultPolicy, .failClosed)
        let failover = try Args.parse(required + ["--backend", "hybrid-ane", "--fault-policy", "failover-metal"])
        XCTAssertEqual(try failover.resolvedRuntimeConfiguration(forceLogitsHead: false).faultPolicy, .failoverMetal)
        XCTAssertThrowsError(try Args.parse(required + ["--fault-policy", "retry"]))
        XCTAssertThrowsError(try Args.parse(required + ["--prefill-chunk", "4096"]))
    }

    func testUpstreamRuntimeControlsPreserveANEContracts() throws {
        let required = ["--model", "/model", "--prompt", "hello"]
        let args = try Args.parse(required + ["--backend", "hybrid-ane",
            "--expert-cache-slots", "24", "--expert-cache-policy", "lru",
            "--prefill-chunk-tokens", "256"])
        let runtime = try args.resolvedRuntimeConfiguration(forceLogitsHead: true)
        XCTAssertEqual(runtime.backend, .hybridANE)
        XCTAssertEqual(runtime.expertCacheSlots, 24)
        XCTAssertEqual(runtime.expertCachePolicy, .lru)
        XCTAssertEqual(runtime.prefillChunkTokens, 256)
        XCTAssertEqual(runtime.activeRoutedExperts, 8)
        XCTAssertEqual(runtime.prefillAttentionPath, .causalSIMD)
        XCTAssertThrowsError(try Args.parse(required + ["--backend", "hybrid-ane", "--expert-top-k", "6"]))
        XCTAssertThrowsError(try Args.parse(required + ["--backend", "hybrid-ane", "--prefill-expert-cap", "8"]))
        XCTAssertThrowsError(try Args.parse(required + ["--prefill", "on", "--expert-cache-slots", "8"]))
        let file = try Args.parse(["--model", "/model", "--prompt-file", "/prompt.txt", "--count-tokens"])
        XCTAssertEqual(file.promptFile, "/prompt.txt")
        XCTAssertTrue(file.countTokens)
    }
}
