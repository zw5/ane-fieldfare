import Metal

/// One prefill chunk's captured state, handed to `PrefillCapture.onChunk`
/// after the chunk's last layer and before its KV cursor advances.
public struct PrefillCaptureChunk {
    public let startPosition: Int
    public let tokenCount: Int
    /// `[numLayers][tokenCount * topK]` routed expert IDs, as read back for routing.
    public let routes: [[UInt32]]
    /// The capture buffer; the next chunk overwrites it.
    private let storage: any MTLBuffer
    private let hiddenCount: Int

    init(startPosition: Int, tokenCount: Int, routes: [[UInt32]],
         storage: any MTLBuffer, hiddenCount: Int) {
        precondition(hiddenCount * MemoryLayout<Float16>.stride <= storage.length,
                     "capture chunk exceeds its buffer")
        self.startPosition = startPosition
        self.tokenCount = tokenCount
        self.routes = routes
        self.storage = storage
        self.hiddenCount = hiddenCount
    }

    /// Lends `[hiddenLayers.count][tokenCount][hiddenSize]` FP16, the residual
    /// stream after each captured layer's tail, layer scalar included, for the
    /// duration of `body`. Read it inside `onChunk`: the next chunk reuses the
    /// buffer.
    public func withHidden<Result>(_ body: (Span<Float16>) throws -> Result) rethrows -> Result {
        // The precondition in `init` bounds the view by the buffer's length.
        let base = unsafe storage.contents().assumingMemoryBound(to: Float16.self)
        let elements = unsafe UnsafeBufferPointer(start: base, count: hiddenCount)
        return try body(unsafe elements.span)
    }

}

/// Opt-in capture of layer outputs and routes during chunked prefill. Off by
/// default; the forward path only pays a blit per captured layer when set.
public final class PrefillCapture {
    public let hiddenLayers: [Int]
    let onChunk: (PrefillCaptureChunk) throws -> Void
    private(set) var buffer: any MTLBuffer?
    var routes: [[UInt32]] = []

    public init(hiddenLayers: [Int], onChunk: @escaping (PrefillCaptureChunk) throws -> Void) {
        self.hiddenLayers = hiddenLayers
        self.onChunk = onChunk
    }

    func prepare(device: any MTLDevice, chunkTokens: Int, hiddenSize: Int, numLayers: Int) throws {
        let length = max(1, hiddenLayers.count * chunkTokens * hiddenSize * MemoryLayout<Float16>.stride)
        if (buffer?.length ?? 0) < length {
            guard let made = device.makeBuffer(length: length, options: .storageModeShared) else {
                throw PrefillError.chunkedUnsupported("prefill capture allocation failed")
            }
            buffer = made
        }
        routes = Array(repeating: [], count: numLayers)
    }
}
