public import Metal

/// Make-/read- helpers for FP16 MTLBuffers. Replaces the byte-identical
/// inline copies that lived in 12+ test files.
public enum Fp16Buffer {
    public static func make(_ device: any MTLDevice, values: [Float]) -> any MTLBuffer? {
        let halves = values.map { Float16($0) }
        return halves.withUnsafeBufferPointer { ptr in
            unsafe device.makeBuffer(
                bytes: ptr.baseAddress!,
                length: halves.count * MemoryLayout<Float16>.size,
                options: .storageModeShared
            )
        }
    }

    public static func make(_ device: any MTLDevice, halves: [Float16]) -> any MTLBuffer? {
        halves.withUnsafeBufferPointer { ptr in
            unsafe device.makeBuffer(
                bytes: ptr.baseAddress!,
                length: halves.count * MemoryLayout<Float16>.size,
                options: .storageModeShared
            )
        }
    }

    public static func make(_ device: any MTLDevice, count: Int) -> any MTLBuffer? {
        device.makeBuffer(
            length: count * MemoryLayout<Float16>.size,
            options: .storageModeShared
        )
    }

    public static func read(_ buf: any MTLBuffer, count: Int) -> [Float] {
        var out = [Float](repeating: 0, count: count)
        let base = unsafe buf.contents()
        for i in 0..<count {
            let h = unsafe base.load(
                fromByteOffset: i * MemoryLayout<Float16>.size,
                as: Float16.self
            )
            out[i] = Float(h)
        }
        return out
    }

    public static func readHalf(_ buf: any MTLBuffer, count: Int) -> [Float16] {
        var out = [Float16](repeating: 0, count: count)
        let base = unsafe buf.contents()
        for i in 0..<count {
            out[i] = unsafe base.load(
                fromByteOffset: i * MemoryLayout<Float16>.size,
                as: Float16.self
            )
        }
        return out
    }
}
