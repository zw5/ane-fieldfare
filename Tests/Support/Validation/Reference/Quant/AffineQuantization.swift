public import ANEFieldfareHotpath

/// Reference quantizers used to build kernel fixtures: Q4_0 for every 4-bit
/// weight and the affine INT8 format of the router.
extension Quantization {
    // MARK: - Q4_0

    /// A Q4_0 row: packed unsigned nibbles and one FP16 scale `d` per group
    /// of 32, `w = d * (q - 8)`. `scales` carries FP16 bit patterns so the
    /// array uploads to a Metal `device const half*` unchanged.
    public struct Q4Row {
        public let packed: [UInt8]   // N / 2 bytes; low nibble = even index, high = odd
        public let scales: [UInt16]  // N / 32 FP16 bits

        public init(packed: [UInt8], scales: [UInt16]) {
            self.packed = packed
            self.scales = scales
        }
    }

    /// ggml's reference Q4_0 quantizer: per group, `d = m / -8` for the first
    /// value `m` of largest magnitude, rounded to FP16, and
    /// `q = min(15, trunc(x / d + 8.5))` against the unrounded `d`.
    public static func quantizeQ4_0(_ row: [Float]) -> Q4Row {
        let group = q4GroupSize
        precondition(row.count % group == 0,
                     "row length \(row.count) is not a multiple of \(group)")
        var packed = [UInt8](repeating: 0, count: row.count / 2)
        var scales = [UInt16](repeating: 0, count: row.count / group)
        for g in 0..<(row.count / group) {
            var amax: Float = 0, extreme: Float = 0
            for k in 0..<group where amax < abs(row[g * group + k]) {
                amax = abs(row[g * group + k])
                extreme = row[g * group + k]
            }
            let d = extreme / -8
            let inv: Float = d != 0 ? 1 / d : 0
            scales[g] = Float16(d).bitPattern
            for k in 0..<group {
                let i = g * group + k
                let q = UInt8(min(15, Int(row[i] * inv + 8.5)))
                packed[i / 2] |= (i & 1) == 0 ? q : q << 4
            }
        }
        return Q4Row(packed: packed, scales: scales)
    }

    public static func dequantizeQ4_0(_ r: Q4Row, n: Int) -> [Float] {
        precondition(n == r.packed.count * 2)
        return (0..<n).map { i in
            let b = r.packed[i / 2]
            let q = (i & 1) == 0 ? Int(b & 0x0F) : Int(b >> 4)
            return Float(Float16(bitPattern: r.scales[i / q4GroupSize])) * Float(q - 8)
        }
    }

    // MARK: - INT8 affine

    public struct Int8AffineRow {
        public let packed: [UInt8]   // N unsigned bytes
        public let scales: [UInt16]  // N / 64 BF16 bits
        public let biases: [UInt16]  // N / 64 BF16 bits

        public init(packed: [UInt8], scales: [UInt16], biases: [UInt16]) {
            self.packed = packed
            self.scales = scales
            self.biases = biases
        }
    }

    public static func quantizeInt8Affine(_ row: [Float]) -> Int8AffineRow {
        precondition(row.count % groupSize == 0,
                     "row length \(row.count) is not a multiple of \(groupSize)")

        let nGroups = row.count / groupSize
        var packed = [UInt8](repeating: 0, count: row.count)
        var scales = [UInt16](repeating: 0, count: nGroups)
        var biases = [UInt16](repeating: 0, count: nGroups)

        for g in 0..<nGroups {
            var wmin: Float =  .infinity
            var wmax: Float = -.infinity
            for k in 0..<groupSize {
                let w = row[g * groupSize + k]
                if w < wmin { wmin = w }
                if w > wmax { wmax = w }
            }
            let scaleF: Float
            let biasF:  Float
            if wmax == wmin {
                scaleF = 1
                biasF  = wmin
            } else {
                scaleF = (wmax - wmin) / 255.0
                biasF  = wmin
            }
            let sBits = bf16Bits(scaleF)
            let bBits = bf16Bits(biasF)
            scales[g] = sBits
            biases[g] = bBits
            let scale = bf16ToFloat(sBits)
            let bias  = bf16ToFloat(bBits)
            let invScale = scale == 0 ? Float(0) : 1.0 / scale

            for k in 0..<groupSize {
                let w = row[g * groupSize + k]
                var q = Int(((w - bias) * invScale).rounded())
                q = max(0, min(255, q))
                packed[g * groupSize + k] = UInt8(q)
            }
        }
        return Int8AffineRow(packed: packed, scales: scales, biases: biases)
    }

    public static func dequantizeInt8Affine(_ r: Int8AffineRow, n: Int) -> [Float] {
        precondition(n == r.packed.count)
        var out = [Float](repeating: 0, count: n)
        let nGroups = n / groupSize
        for g in 0..<nGroups {
            let scale = bf16ToFloat(r.scales[g])
            let bias  = bf16ToFloat(r.biases[g])
            for k in 0..<groupSize {
                out[g * groupSize + k] = Float(r.packed[g * groupSize + k]) * scale + bias
            }
        }
        return out
    }
}
