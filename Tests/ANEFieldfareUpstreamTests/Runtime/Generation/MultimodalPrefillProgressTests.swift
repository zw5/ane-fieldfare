import Metal
import Testing
@testable import ANEFieldfareHotpath

/// Defaults that travel with multimodal prefill.
@Suite struct MultimodalPrefillProgressTests {
    /// Every entry point must default to the same residency. The conversation's
    /// default had drifted to `keepReady`, which pins ~1.1 GB of tower mappings
    /// for the life of the process while the CLI released them.
    @Test func everyEntryPointDefaultsToTheSameResidency() {
        #expect(VisionResidencyPolicy.defaultPolicy == .onDemand)
        #expect(VisionResidencyPolicy(rawValue: "keep-ready") == .keepReady,
                "the opt-in policy is still spelled keep-ready")
    }
}
