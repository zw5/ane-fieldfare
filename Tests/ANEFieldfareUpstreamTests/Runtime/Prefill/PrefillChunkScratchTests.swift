import Testing
@testable import ANEFieldfareHotpath

@Suite struct PrefillChunkScratchTests {
    /// The scratch holds a text chunk, or the largest image block as one unit.
    @Test func scratchHoldsTextChunksAndWholeImageBlocks() {
        let text = PrefillChunkScratchLayout(config: .gemma4_26B_A4B, chunkTokens: 32)
        #expect(text.chunkTokens == 32)

        let tokens = VisionConfig().maximumPooledTokens
        let image = PrefillChunkScratchLayout(
            config: .gemma4_26B_A4B,
            chunkTokens: tokens,
            chunkTokenLimit: tokens)
        #expect(image.chunkTokens == 280)
    }

    @Test func layoutClampsChunkSizeToRuntimeBounds() {
        #expect(PrefillChunkScratchLayout(config: .gemma4_26B_A4B, chunkTokens: 0).chunkTokens == 1)
        // Chunks up to 4,096 tokens keep a long prompt layer-major.
        #expect(PrefillChunkScratchLayout(config: .gemma4_26B_A4B, chunkTokens: 8192).chunkTokens
                == PrefillRuntimeConfig.maxChunkTokens)
        #expect(PrefillRuntimeConfig.maxChunkTokens == 4096)
        #expect(PrefillChunkScratchLayout(config: .gemma4_26B_A4B, chunkTokens: 256).chunkTokens == 256)
    }

    /// `auto` asks for the smallest allowed chunk that covers the prompt,
    /// because one chunk means each layer's experts are read once - the floor.
    @Test func autoPicksTheSmallestChunkThatCoversThePrompt() {
        #expect(PrefillRuntimeConfig.autoChunkTokens(promptTokens: 20) == 32)
        #expect(PrefillRuntimeConfig.autoChunkTokens(promptTokens: 32) == 32)
        #expect(PrefillRuntimeConfig.autoChunkTokens(promptTokens: 33) == 64)
        #expect(PrefillRuntimeConfig.autoChunkTokens(promptTokens: 200) == 256)
        // Beyond the ceiling it saturates rather than inventing a size.
        #expect(PrefillRuntimeConfig.autoChunkTokens(promptTokens: 7_019) == 4096)
        // A cap below the ceiling is honoured, so a caller can stay smaller.
        #expect(PrefillRuntimeConfig.autoChunkTokens(promptTokens: 7_019, cap: 64) == 64)
    }

    @Test func plannerUsesConfiguredChunkSize() {
        let spans = PrefillChunkPlanner.spans(
            tokenCount: 130,
            startPosition: 7,
            config: .production(chunkTokens: 64))
        #expect(spans.map(\.tokenCount) == [64, 64, 2])
        #expect(spans.map(\.startPosition) == [7, 71, 135])
    }
}
