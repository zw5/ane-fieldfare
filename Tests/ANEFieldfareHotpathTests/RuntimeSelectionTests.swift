import XCTest
import ANEFieldfareHotpath
import ANEFieldfareHotpathCLICore

final class RuntimeSelectionTests: XCTestCase {
    /// The hybrid ANE rail runs all eight routed experts through the causal
    /// SIMD prefill path; flags that would break that contract are refused.
    func testHybridANESelectionPreservesANEContracts() throws {
        let required = ["--model", "/model", "--prompt", "hello"]
        let args = try Args.parse(required + ["--backend", "hybrid-ane",
            "--prefill-chunk-tokens", "256", "--fault-policy", "failover-metal"])
        let runtime = try args.resolvedRuntimeConfiguration(forceLogitsHead: true)
        XCTAssertEqual(runtime.backend, .hybridANE)
        XCTAssertEqual(runtime.prefillChunkTokens, 256)
        XCTAssertEqual(runtime.faultPolicy, .failoverMetal)
        XCTAssertEqual(runtime.activeRoutedExperts, 8)
        XCTAssertEqual(runtime.prefillAttentionPath, .causalSIMD)
        XCTAssertEqual(try Args.parse(required).backend, .hybridANE)
        for flags in [["--expert-top-k", "6"], ["--prefill-expert-cap", "8"], ["--prefill", "off"]] {
            XCTAssertEqual(try Args.parse(required + flags).backend, .metal)
        }
        XCTAssertThrowsError(try Args.parse(required + ["--backend", "ane-magic"]))
        XCTAssertThrowsError(try Args.parse(required + ["--backend", "hybrid-ane", "--expert-top-k", "6"]))
        XCTAssertThrowsError(try Args.parse(required + ["--backend", "hybrid-ane", "--prefill-expert-cap", "8"]))
    }
}
