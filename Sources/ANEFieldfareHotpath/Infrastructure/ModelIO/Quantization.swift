public enum Quantization {
    /// The affine INT8 router's group size, and the width every 4-bit kernel
    /// requires a row to be a multiple of.
    public static let groupSize: Int = 64
    /// Q4_0: values per FP16 scale.
    public static let q4GroupSize: Int = 32

    // BF16 = top 16 bits of FP32 (8-bit exponent, 7-bit mantissa). Stored on
    // disk and on the GPU as the native `bfloat` type; in Swift we carry the
    // raw bits as `UInt16`. Encode rounds half to even and does not
    // special-case NaN/Inf, so callers pass finite values.
    @inline(__always)
    public static func bf16Bits(_ x: Float) -> UInt16 {
        let bits = x.bitPattern
        let lsb  = (bits >> 16) & 1
        let roundingBias: UInt32 = 0x7FFF &+ lsb
        let rounded = (bits &+ roundingBias) >> 16
        return UInt16(truncatingIfNeeded: rounded)
    }

    @inline(__always)
    public static func bf16ToFloat(_ bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }
}
