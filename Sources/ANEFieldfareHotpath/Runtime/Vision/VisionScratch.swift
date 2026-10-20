package import Metal

package final class VisionScratch {
    package let paddedRows: Int
    /// Live rows, so the clear can skip everything a dispatch will overwrite.
    private let rows: Int
    private let hiddenSize: Int
    package let normalizedPatches: any MTLBuffer
    package let hiddenA: any MTLBuffer
    package let hiddenB: any MTLBuffer
    /// Q/K/V live in attention's padded buffers; `q` is general scratch.
    package let q: any MTLBuffer
    package let attention: any MTLBuffer
    /// Gate projection output; the up projection folds GeGLU into it.
    package let gate: any MTLBuffer

    package init(device: any MTLDevice, rows: Int) throws {
        paddedRows = VisionAttention.paddedRowCount(rows)
        self.rows = rows
        self.hiddenSize = VisionConfig.hiddenSize
        func buffer(_ elements: Int) throws -> any MTLBuffer {
            guard let value = device.makeBuffer(
                length: elements * MemoryLayout<UInt16>.stride,
                options: .storageModePrivate) else {
                throw MetalError.allocationFailed("vision scratch")
            }
            return value
        }
        normalizedPatches = try buffer(paddedRows * VisionConfig.patchDimension)
        hiddenA = try buffer(paddedRows * VisionConfig.hiddenSize)
        hiddenB = try buffer(paddedRows * VisionConfig.hiddenSize)
        q = try buffer(paddedRows * VisionConfig.hiddenSize)
        attention = try buffer(paddedRows * VisionConfig.hiddenSize)
        gate = try buffer(paddedRows * VisionConfig.intermediateSize + 64)
    }

    /// Zeroes the padding rows of `hiddenB` and `attention`, the only buffers
    /// whose tails are read back; every other buffer is fully overwritten,
    /// pad included, by the dispatch that fills it.
    package func encodeClear(commandBuffer: any MTLCommandBuffer) throws {
        guard let blit = commandBuffer.makeTracedBlitEncoder() else {
            throw MetalError.encoderUnavailable
        }
        let liveBytes = rows * hiddenSize * MemoryLayout<UInt16>.stride
        for buffer in [hiddenB, attention] {
            guard buffer.length > liveBytes else { continue }
            blit.fill(buffer: buffer, range: liveBytes..<buffer.length, value: 0)
        }
        blit.endEncoding()
    }
}
