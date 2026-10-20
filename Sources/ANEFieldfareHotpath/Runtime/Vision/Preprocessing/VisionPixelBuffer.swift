public import Metal

public struct VisionImagePlan {
    public let metadata: VisionImageMetadata
    public let geometry: Gemma4ImageGeometry
    package let opened: OpenedVisionImage
}

public struct VisionPixelBuffer {
    public let patchesBF16: any MTLBuffer
    public let positionsInt32x2: any MTLBuffer
    public let geometry: Gemma4ImageGeometry
}
