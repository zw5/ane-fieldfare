/// The Gemma 4 vision tower's fixed shape.
public enum VisionConfig {
    public static let hiddenSize = 1_152
    public static let intermediateSize = 4_304
    public static let numLayers = 27
    public static let numHeads = 16
    public static let patchSize = 16
    public static let patchDimension = 768
    public static let maximumPatches = 2_520
    public static let poolingKernel = 3
    public static let textHiddenSize = 2_816

    public static var maximumPooledTokens: Int {
        maximumPatches / (poolingKernel * poolingKernel)
    }
}
