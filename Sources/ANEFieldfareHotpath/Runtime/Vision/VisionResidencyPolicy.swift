public enum VisionResidencyPolicy: String, Sendable {
    case onDemand = "on-demand"
    case keepReady = "keep-ready"

    /// Keeping the tower mapped costs ~5 MB of footprint and saves ~105 ms of
    /// remapping per image, not worth pinning by default.
    public static let defaultPolicy: VisionResidencyPolicy = .onDemand
}

public struct VisionExpertResidencyTransition: Sendable, Equatable {
    public let policy: VisionResidencyPolicy
    public let gpuDrainNanoseconds: UInt64
    public let releasedLayerCount: Int
    public let remainingOpenLayerCount: Int
    public let wallNanoseconds: UInt64
}
