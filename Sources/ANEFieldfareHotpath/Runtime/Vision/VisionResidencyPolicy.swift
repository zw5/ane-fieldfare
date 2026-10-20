public enum VisionResidencyPolicy: String, Sendable {
    case onDemand = "on-demand"
    case keepReady = "keep-ready"

    /// Remapping the tower per image costs less than keeping it pinned.
    public static let defaultPolicy: VisionResidencyPolicy = .onDemand
}
