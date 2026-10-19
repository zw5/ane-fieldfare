package import Metal

package final class VisionScratch {
    package let paddedRows: Int
    /// Live rows, so the clear can skip everything a dispatch will overwrite.
    package let rows: Int
    private let hiddenSize: Int
    package let normalizedPatches: any MTLBuffer
    package let hiddenA: any MTLBuffer
    package let hiddenB: any MTLBuffer
    /// Q/K/V live in attention's padded buffers; `q` is general scratch.
    package let q: any MTLBuffer
    package let attention: any MTLBuffer
    /// Gate projection output; the up projection folds GeGLU into it.
    package let gate: any MTLBuffer

    package init(device: any MTLDevice, rows: Int, config: VisionConfig) throws {
        paddedRows = ((rows + 63) / 64) * 64
        self.rows = rows
        self.hiddenSize = config.hiddenSize
        func buffer(_ elements: Int) throws -> any MTLBuffer {
            guard let value = device.makeBuffer(
                length: elements * MemoryLayout<UInt16>.stride,
                options: .storageModePrivate) else {
                throw MetalError.noDevice
            }
            return value
        }
        normalizedPatches = try buffer(paddedRows * config.patchDimension)
        hiddenA = try buffer(paddedRows * config.hiddenSize)
        hiddenB = try buffer(paddedRows * config.hiddenSize)
        q = try buffer(paddedRows * config.hiddenSize)
        attention = try buffer(paddedRows * config.hiddenSize)
        gate = try buffer(paddedRows * config.intermediateSize + 64)
    }

    private var buffers: [any MTLBuffer] {
        [normalizedPatches, hiddenA, hiddenB, q, attention, gate]
    }

    /// Zeroes only what a later read can actually see.
    ///
    /// Clearing every buffer meant about 47 MiB of blit fills per image at the
    /// 2,520-patch maximum, nearly all of it dead: `normalizedPatches`, `q` and
    /// `gate` are fully overwritten by the dispatch that follows, which zeroes
    /// its own pad region. What genuinely has to start clean is the padding
    /// rows of the two buffers whose tails are read back — `hiddenB` and
    /// `attention` — which is under 300 KB.
    package func encodeClear(commandBuffer: any MTLCommandBuffer) throws {
        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw MetalError.noDevice
        }
        let liveBytes = rows * hiddenSize * MemoryLayout<UInt16>.stride
        for buffer in [hiddenB, attention] {
            guard buffer.length > liveBytes else { continue }
            blit.fill(buffer: buffer, range: liveBytes..<buffer.length, value: 0)
        }
        blit.endEncoding()
    }

    package var allocatedBytes: Int {
        buffers.reduce(0) { $0 + $1.length }
    }
}
