import Metal
import Testing
@testable import ANEFieldfareHotpath

/// Projected features of `rows` text-hidden rows, contents unspecified.
func visionTestFeatures(rows: Int, context: MetalContext) throws -> VisionFeatures {
    let bytes = rows * VisionConfig.textHiddenSize * MemoryLayout<Float16>.stride
    let buffer = try #require(context.device.makeBuffer(
        length: bytes, options: .storageModePrivate))
    return VisionFeatures(
        buffer: buffer, tokenCount: rows,
        hiddenSize: VisionConfig.textHiddenSize)
}

@Suite struct MultimodalPrefillInputTests {
    @Test func rejectsFeatureTokenCountMismatch() throws {
        let features = try visionTestFeatures(rows: 3, context: MetalContext())
        #expect(throws: MultimodalPrefillInputError.featureShapeMismatch) {
            try MultimodalPrefillInput(
                effectiveTokenIDs: [10, 20, 20, 30],
                embeddingTokenIDs: [10, 0, 0, 30],
                imageSpans: [MultimodalImageSpan(tokenRange: 1..<3, features: features)])
        }
    }

    @Test func acceptsOrderedNonOverlappingImageSpans() throws {
        let context = try MetalContext()
        let input = try MultimodalPrefillInput(
            effectiveTokenIDs: [1, 2, 2, 3, 4, 4, 4, 5],
            embeddingTokenIDs: [1, 0, 0, 3, 0, 0, 0, 5],
            imageSpans: [
                MultimodalImageSpan(tokenRange: 1..<3, features: try visionTestFeatures(rows: 2, context: context)),
                MultimodalImageSpan(tokenRange: 4..<7, features: try visionTestFeatures(rows: 3, context: context)),
            ])
        #expect(input.imageSpans.map(\.tokenRange) == [1..<3, 4..<7])
        #expect(input.effectiveTokenIDs[1] == 2)
        #expect(input.embeddingTokenIDs[1] == 0)
    }
}
