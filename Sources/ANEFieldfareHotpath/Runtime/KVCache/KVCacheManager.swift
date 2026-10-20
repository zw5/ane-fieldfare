import Foundation
import Darwin
public import Metal

/// Per-layer FP16 K/V storage for the decode loop.
///
/// One K buffer and one V buffer per layer, allocated once in `init`; the
/// decode hot path never allocates. Sliding-window layers are rings sized to
/// the window plus one prefill chunk; full-attention layers are linear over
/// `maxContext`.
///
/// Full-attention layers share `k_proj` for K and V, but K then runs `k_norm`
/// and RoPE while V runs a no-scale `v_norm` without RoPE, so K and V always
/// need separate slots; aliasing them would overwrite V with K's rotated bytes.
///
/// `reset()` returns physical pages to the OS via `MADV_DONTNEED` so a
/// finished generation does not keep its KV resident into the next turn.
public final class KVCacheManager {
    public let maxContext: Int

    private let numLayers: Int
    private let kBuffers: [any MTLBuffer]
    private let vBuffers: [any MTLBuffer]
    private let strides: [Int]          // bytes per token, per layer
    private let isSWA: [Bool]
    private let capacityTokens: [Int]
    private let slidingWindowTokens: Int

    public private(set) var position: Int = 0
    public private(set) var highWaterPosition: Int = 0

    public init(device: any MTLDevice,
                config: ArchConfig,
                maxContext: Int,
                maxPrefillChunkTokens: Int) throws {
        precondition(maxContext > 0, "maxContext must be positive")
        precondition(maxPrefillChunkTokens > 0, "maxPrefillChunkTokens must be positive")
        self.maxContext = maxContext
        self.numLayers = config.numLayers
        self.slidingWindowTokens = config.slidingWindow

        let fp16Size = MemoryLayout<Float16>.size
        let swaStride  = config.numKVHeads     * config.headDim     * fp16Size
        let fullStride = config.numFullKVHeads * config.fullHeadDim * fp16Size
        let swaCapacity = min(maxContext, config.slidingWindow + maxPrefillChunkTokens)

        var ks: [any MTLBuffer] = []
        var vs: [any MTLBuffer] = []
        var st: [Int] = []
        var swa: [Bool] = []
        var caps: [Int] = []
        for layer in 0..<config.numLayers {
            let isFull = config.fullAttentionLayerMask[layer] != 0
            let stride = isFull ? fullStride : swaStride
            let capacity = isFull ? maxContext : swaCapacity
            let length = capacity * stride
            guard let kBuf = device.makeBuffer(length: length, options: .storageModeShared),
                  let vBuf = device.makeBuffer(length: length, options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            kBuf.label = "kv.K.layer\(layer)"
            vBuf.label = "kv.V.layer\(layer)"
            ks.append(kBuf)
            vs.append(vBuf)
            st.append(stride)
            swa.append(!isFull)
            caps.append(capacity)
        }
        self.kBuffers = ks
        self.vBuffers = vs
        self.strides = st
        self.isSWA = swa
        self.capacityTokens = caps
    }

    /// Physical token capacity for `layer`: the ring size for a sliding-window
    /// layer, `maxContext` for a full layer.
    public func capacity(layer: Int) -> Int { capacityTokens[layer] }

    /// The ring capacity attention maps positions with; 0 for a full layer.
    public func ringCapacity(layer: Int) -> Int {
        isSWA[layer] ? capacityTokens[layer] : 0
    }

    /// Write target for this layer's K projection at `position`.
    public func kSlot(layer: Int, position: Int) -> (buffer: any MTLBuffer, offset: Int) {
        validateRange(start: position, count: 1)
        return (kBuffers[layer], physicalSlot(layer: layer, position: position) * strides[layer])
    }

    /// Write target for this layer's V projection at `position`; never
    /// aliases `kSlot`.
    public func vSlot(layer: Int, position: Int) -> (buffer: any MTLBuffer, offset: Int) {
        validateRange(start: position, count: 1)
        return (vBuffers[layer], physicalSlot(layer: layer, position: position) * strides[layer])
    }

    public func kRange(layer: Int, start: Int, count: Int) -> (buffer: any MTLBuffer, offset: Int, stride: Int) {
        validateRange(start: start, count: count)
        validateContiguousPhysicalRange(layer: layer, start: start, count: count)
        return (kBuffers[layer], physicalSlot(layer: layer, position: start) * strides[layer], strides[layer])
    }

    public func vRange(layer: Int, start: Int, count: Int) -> (buffer: any MTLBuffer, offset: Int, stride: Int) {
        validateRange(start: start, count: count)
        validateContiguousPhysicalRange(layer: layer, start: start, count: count)
        return (vBuffers[layer], physicalSlot(layer: layer, position: start) * strides[layer], strides[layer])
    }

    func keyBuffer(layer: Int) -> any MTLBuffer { kBuffers[layer] }

    func valueBuffer(layer: Int) -> any MTLBuffer { vBuffers[layer] }

    /// Advance the position cursor once the current token's K/V are written
    /// across all layers.
    public func advance() { advance(by: 1) }

    public func advance(by count: Int) {
        precondition(count >= 0, "advance count must be non-negative")
        precondition(position + count <= maxContext, "advance would exceed maxContext")
        position += count
        highWaterPosition = max(highWaterPosition, position)
    }

    /// Maximum safe rewind before a reused SWA ring slot could become visible.
    public var maxRewindTokens: Int {
        var slack = maxContext
        for layer in 0..<numLayers where isSWA[layer] && capacityTokens[layer] < maxContext {
            slack = min(slack, max(0, capacityTokens[layer] - slidingWindowTokens))
        }
        return slack
    }

    /// Abandon the newest rows while keeping the remaining KV lineage intact.
    public func rewind(to newPosition: Int) {
        precondition(newPosition >= 0, "rewind target must be non-negative")
        precondition(newPosition <= position,
                     "rewind target \(newPosition) is beyond position \(position)")
        precondition(highWaterPosition - newPosition <= maxRewindTokens,
                     "rewind to \(newPosition) exceeds ring slack \(maxRewindTokens)")
        position = newPosition
    }

    /// Drop all cached positions and return physical pages to the OS. No
    /// zeroing: attention reads only rows below the cursor.
    public func reset() {
        position = 0
        highWaterPosition = 0
        let pageSize = Int(getpagesize())
        for buffer in kBuffers + vBuffers {
            // MTLBuffer allocations are page-aligned; never advise a partial
            // tail page.
            let length = (buffer.length / pageSize) * pageSize
            if length > 0 {
                _ = unsafe posix_madvise(buffer.contents(), length, POSIX_MADV_DONTNEED)
            }
        }
    }

    private func validateRange(start: Int, count: Int) {
        precondition(count >= 0, "count must be non-negative")
        precondition(start >= 0, "start must be non-negative")
        precondition(start + count <= maxContext,
                     "range \(start)..<\(start + count) exceeds maxContext \(maxContext)")
    }

    private func physicalSlot(layer: Int, position: Int) -> Int {
        position % capacityTokens[layer]
    }

    private func validateContiguousPhysicalRange(layer: Int, start: Int, count: Int) {
        guard count > 0, isSWA[layer] else { return }
        let capacity = capacityTokens[layer]
        precondition(start % capacity + count <= capacity,
                     "range \(start)..<\(start + count) wraps FP16 KV ring capacity \(capacity)")
    }
}
