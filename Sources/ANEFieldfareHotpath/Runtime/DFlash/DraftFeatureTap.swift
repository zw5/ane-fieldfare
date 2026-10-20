import Metal

/// Where the target leaves the drafter's context features: for each captured
/// target layer, the residual stream after that layer's tail (layer scalar
/// included), one FP16 row per position at ring slot `position % capacity`.
///
/// Prefill chunks, verify blocks and decode steps copy their rows here on the
/// GPU, in the command buffer of the layer's tail; the drafter reads them on
/// the same queue, so features never pass through the CPU.
struct DraftFeatureTap {
    /// `[layers.count][capacity][hiddenSize]` FP16.
    let buffer: any MTLBuffer
    let layers: [Int]
    let capacity: Int
    let hiddenSize: Int

    var rowBytes: Int { hiddenSize * MemoryLayout<Float16>.stride }

    /// Copies `rows` rows of `source` (FP16, `hiddenSize` apart, from its
    /// start) as target layer `layer`'s rows for positions
    /// `position..<position + rows`. A layer the drafter does not read is
    /// skipped.
    func encodeCopy(commandBuffer: any MTLCommandBuffer, layer: Int, source: any MTLBuffer,
                    position: Int, rows: Int) throws {
        guard let slot = layers.firstIndex(of: layer), rows > 0 else { return }
        guard rows <= capacity else {
            throw DFlashError.invalidState("\(rows) feature rows exceed the \(capacity)-row ring")
        }
        let blit = try commandBuffer.makeTracedBlitEncoder()
        let first = position % capacity
        let head = min(rows, capacity - first)
        let base = slot * capacity * rowBytes
        blit.copy(from: source, sourceOffset: 0, to: buffer, destinationOffset: base + first * rowBytes,
                  size: head * rowBytes)
        if head < rows {
            blit.copy(from: source, sourceOffset: head * rowBytes, to: buffer, destinationOffset: base,
                      size: (rows - head) * rowBytes)
        }
        blit.endEncoding()
    }
}
