import Foundation

public struct Gemma4ImageGeometry: Sendable, Equatable {
    /// This initialiser is public and validates its inputs, so a degenerate
    /// aspect ratio must throw rather than trap on overflow.
    private static func checked(_ lhs: Int, _ rhs: Int) throws -> Int {
        let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else {
            throw VisionImageError.invalidMetadata("source dimensions overflow")
        }
        return value
    }

    public let processedWidth: Int
    public let processedHeight: Int
    public let patchGridWidth: Int
    public let patchGridHeight: Int
    public let patchCount: Int
    public let softTokenCount: Int

    public init(
        sourceWidth: Int,
        sourceHeight: Int
    ) throws {
        guard sourceWidth > 0, sourceHeight > 0 else {
            throw VisionImageError.invalidMetadata("source dimensions must be positive")
        }
        let (sourcePixels, sourceOverflow) = sourceWidth.multipliedReportingOverflow(
            by: sourceHeight)
        guard !sourceOverflow else {
            throw VisionImageError.invalidMetadata("source dimensions overflow")
        }
        let patchPixels = VisionConfig.patchSize * VisionConfig.patchSize
        let targetPixels = VisionConfig.maximumPatches * patchPixels
        let factor = sqrt(Double(targetPixels) / Double(sourcePixels))
        let sideMultiple = VisionConfig.patchSize * VisionConfig.poolingKernel
        var height = Int(floor(factor * Double(sourceHeight) / Double(sideMultiple)))
            * sideMultiple
        var width = Int(floor(factor * Double(sourceWidth) / Double(sideMultiple)))
            * sideMultiple
        let maximumSide = (VisionConfig.maximumPatches
            / (VisionConfig.poolingKernel * VisionConfig.poolingKernel)) * sideMultiple
        if height == 0 && width == 0 {
            throw VisionImageError.invalidMetadata("aspect ratio produces zero target dimensions")
        } else if height == 0 {
            height = sideMultiple
            width = min(try Self.checked(sourceWidth / sourceHeight, sideMultiple), maximumSide)
        } else if width == 0 {
            width = sideMultiple
            height = min(try Self.checked(sourceHeight / sourceWidth, sideMultiple), maximumSide)
        }
        let (processedPixels, processedOverflow) = width.multipliedReportingOverflow(by: height)
        guard !processedOverflow, processedPixels <= targetPixels else {
            throw VisionImageError.invalidMetadata("target geometry exceeds patch budget")
        }

        let gridWidth = width / VisionConfig.patchSize
        let gridHeight = height / VisionConfig.patchSize
        let patches = gridWidth * gridHeight
        guard gridWidth > 0, gridHeight > 0,
              gridWidth.isMultiple(of: VisionConfig.poolingKernel),
              gridHeight.isMultiple(of: VisionConfig.poolingKernel),
              patches <= VisionConfig.maximumPatches else {
            throw VisionImageError.invalidMetadata("target patch grid is invalid")
        }
        processedWidth = width
        processedHeight = height
        patchGridWidth = gridWidth
        patchGridHeight = gridHeight
        patchCount = patches
        softTokenCount = patches / (VisionConfig.poolingKernel * VisionConfig.poolingKernel)
    }
}
