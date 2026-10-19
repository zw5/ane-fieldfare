import Testing
@testable import ANEFieldfareHotpath

@Suite struct RuntimeConfigurationTests {
    @Test func productionDefaultsAreStable() {
        let runtime = RuntimeConfiguration.production
        #expect(runtime.fp16RingEnabled)
        #expect(runtime.prefillPolicy == .chunked)
        #expect(runtime.prefillChunkTokens == 1024)
        #expect(runtime.prefillAttentionPath == .causalSIMD)
        #expect(runtime.headPath == .fusedRows)
    }

    @Test func retainedControlsReachTypedRuntime() {
        let runtime = RuntimeConfiguration(
            prefillEnabled: false,
            prefillChunkTokens: 64,
            prefillAttentionPath: .causalTiled,
            forceLogitsHead: true)
        #expect(runtime.prefillConfig == .off)
        #expect(runtime.prefillAttentionPath == .causalTiled)
        #expect(runtime.headPath == .logits)
    }

    @Test(arguments: [32, 64, 128, 256, 512, 1024])
    func productionPrefillSupportsPublicChunkSizes(_ chunkTokens: Int) {
        let runtime = RuntimeConfiguration(prefillChunkTokens: chunkTokens)
        #expect(runtime.prefillConfig.mode == .chunked)
        #expect(runtime.prefillConfig.chunkTokens == chunkTokens)
    }
}
