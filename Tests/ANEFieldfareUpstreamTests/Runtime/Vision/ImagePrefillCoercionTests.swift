import Testing

@testable import ANEFieldfareHotpath

/// Image spans are only served by the chunked multimodal path. Prefill mode is
/// a performance choice; whether images work at all is not, so an image turn is
/// coerced rather than refused after every image has already been encoded.
@Suite struct ImagePrefillCoercionTests {
    @Test func anImagePromptCoercesPrefillOffToChunked() throws {
        let off = PrefillRuntimeConfig.off
        #expect(!off.servesImagePrompt)

        let coerced = try #require(off.coercedForImagePrompt())
        #expect(coerced.mode == .chunked)
        #expect(coerced.servesImagePrompt)
        // nil means "already runs", not "cannot run".
        #expect(coerced.coercedForImagePrompt() == nil)
    }

    /// The planner cuts text at the same clamp the scratch layout applies, so a
    /// caller that asks for more never hands the buffer a larger chunk than it
    /// was sized for. An image span is one indivisible block: its features are
    /// projected as a unit, so it passes through whole even past the clamp.
    @Test func textChunksRespectTheClampAndImageSpansStayWhole() {
        let span = 400..<(400 + PrefillRuntimeConfig.maxChunkTokens + 1)
        let work = PrefillChunkPlanner.multimodalWork(
            tokenCount: span.upperBound + 5_000, imageRanges: [span], chunkTokens: 8_192)
        let textItems = work.filter { !$0.isImage }
        #expect(!textItems.isEmpty)
        for item in textItems {
            #expect(item.range.count <= PrefillRuntimeConfig.maxChunkTokens)
        }
        #expect(work.filter(\.isImage).map(\.range) == [span])
    }

    @Test func workItemsCoverTheWholePromptInOrder() {
        let work = PrefillChunkPlanner.multimodalWork(
            tokenCount: 500, imageRanges: [10..<40, 200..<260], chunkTokens: 64)
        #expect(work.first?.range.lowerBound == 0)
        #expect(work.last?.range.upperBound == 500)
        for (previous, next) in zip(work, work.dropFirst()) {
            #expect(previous.range.upperBound == next.range.lowerBound)
        }
        #expect(work.filter(\.isImage).map(\.range) == [10..<40, 200..<260])
    }
}
