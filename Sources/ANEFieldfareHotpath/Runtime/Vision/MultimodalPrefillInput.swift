import Metal

public enum MultimodalPrefillInputError: Error, Equatable {
    case tokenCountMismatch
    case invalidImageTokenRange
    case featureShapeMismatch
}

public struct MultimodalImageSpan: Sendable {
    public let tokenRange: Range<Int>
    public let features: VisionFeatures

    public init(tokenRange: Range<Int>, features: VisionFeatures) {
        self.tokenRange = tokenRange
        self.features = features
    }
}

public struct MultimodalPrefillInput: Sendable {
    public let effectiveTokenIDs: [Int32]
    public let embeddingTokenIDs: [Int32]
    public let imageSpans: [MultimodalImageSpan]

    public init(effectiveTokenIDs: [Int32],
                embeddingTokenIDs: [Int32],
                imageSpans: [MultimodalImageSpan]) throws {
        guard effectiveTokenIDs.count == embeddingTokenIDs.count else {
            throw MultimodalPrefillInputError.tokenCountMismatch
        }
        // The span count is self-limiting: spans are ordered, non-overlapping and
        // bounded by the prompt, so the prompt length caps how many can exist.
        guard !imageSpans.isEmpty else {
            throw MultimodalPrefillInputError.invalidImageTokenRange
        }
        var previousUpperBound = 0
        for span in imageSpans {
            let range = span.tokenRange
            guard !range.isEmpty,
                  range.lowerBound >= previousUpperBound,
                  range.upperBound <= embeddingTokenIDs.count,
                  range.count <= VisionConfig().maximumPooledTokens else {
                throw MultimodalPrefillInputError.invalidImageTokenRange
            }
            let featureBytes = range.count
                * VisionConfig().textHiddenSize
                * MemoryLayout<Float16>.stride
            guard span.features.tokenCount == range.count,
                  span.features.hiddenSize == VisionConfig().textHiddenSize,
                  span.features.buffer.length >= featureBytes else {
                throw MultimodalPrefillInputError.featureShapeMismatch
            }
            previousUpperBound = range.upperBound
        }
        self.effectiveTokenIDs = effectiveTokenIDs
        self.embeddingTokenIDs = embeddingTokenIDs
        self.imageSpans = imageSpans
    }
}
