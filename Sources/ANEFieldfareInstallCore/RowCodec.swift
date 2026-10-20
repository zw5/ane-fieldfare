/// Row conversions from GGUF tensor data to the `.fieldfare` encodings.
///
/// Every function converts whole rows: `rows` consecutive source rows of
/// `cols` values into `rows` consecutive destination rows, appending to the
/// output arrays.
package enum RowCodec {
    /// GGUF Q4_0 → `.fieldfare` q4_0. A GGUF block is an FP16 scale and 16
    /// bytes whose low nibbles are values 0–15 and high nibbles values 16–31;
    /// the destination keeps the scale and packs values 2k, 2k+1 into byte k.
    /// The values and scales are unchanged.
    package static func repackQ4_0(_ src: ArraySlice<UInt8>, cols: Int,
                                   weights: inout [UInt8], scales: inout [UInt8]) {
        precondition(cols % 32 == 0 && src.count % (cols / 32 * 18) == 0)
        var block = src.startIndex
        while block < src.endIndex {
            scales.append(src[block])
            scales.append(src[block + 1])
            let qs = block + 2
            for i in 0..<8 {
                weights.append((src[qs + 2 * i] & 0x0F) | ((src[qs + 2 * i + 1] & 0x0F) << 4))
            }
            for i in 0..<8 {
                weights.append((src[qs + 2 * i] >> 4) | (src[qs + 2 * i + 1] & 0xF0))
            }
            block += 18
        }
    }

    /// GGUF Q6_K → `.fieldfare` q4_0: dequantize each row exactly as ggml
    /// does, then quantize it with ggml's reference Q4_0 quantizer.
    package static func q6KToQ4_0(_ src: ArraySlice<UInt8>, cols: Int,
                                  weights: inout [UInt8], scales: inout [UInt8]) {
        let rowBytes = cols / 256 * 210
        precondition(cols % 256 == 0 && src.count % rowBytes == 0)
        var values = [Float](repeating: 0, count: cols)
        var row = src.startIndex
        while row < src.endIndex {
            dequantizeQ6_K(src[row..<(row + rowBytes)], into: &values)
            quantizeQ4_0(values, weights: &weights, scales: &scales)
            row += rowBytes
        }
    }

    /// ggml `dequantize_row_q6_K`. A block is `ql[128] qh[64] scales[16] d`.
    static func dequantizeQ6_K(_ src: ArraySlice<UInt8>, into out: inout [Float]) {
        var block = src.startIndex
        var y = 0
        while block < src.endIndex {
            let d = Float(Float16(bitPattern: UInt16(src[block + 208]) | UInt16(src[block + 209]) << 8))
            for half in 0..<2 {
                let ql = block + 64 * half
                let qh = block + 128 + 32 * half
                let sc = block + 192 + 8 * half
                func scale(_ i: Int) -> Float { Float(Int8(bitPattern: src[sc + i])) }
                for l in 0..<32 {
                    let s = l / 16
                    let h = Int(src[qh + l])
                    let q1 = Int(src[ql + l] & 0x0F) | ((h >> 0) & 3) << 4
                    let q2 = Int(src[ql + l + 32] & 0x0F) | ((h >> 2) & 3) << 4
                    let q3 = Int(src[ql + l] >> 4) | ((h >> 4) & 3) << 4
                    let q4 = Int(src[ql + l + 32] >> 4) | ((h >> 6) & 3) << 4
                    out[y + l] = d * scale(s) * Float(q1 - 32)
                    out[y + l + 32] = d * scale(s + 2) * Float(q2 - 32)
                    out[y + l + 64] = d * scale(s + 4) * Float(q3 - 32)
                    out[y + l + 96] = d * scale(s + 6) * Float(q4 - 32)
                }
                y += 128
            }
            block += 210
        }
    }

    /// ggml `quantize_row_q4_0_ref`, written in the `.fieldfare` nibble order:
    /// `d = m / -8` for the first value `m` of largest magnitude, and
    /// `q = min(15, trunc(x / d + 8.5))`.
    package static func quantizeQ4_0(_ values: [Float], weights: inout [UInt8], scales: inout [UInt8]) {
        precondition(values.count % 32 == 0)
        var q = [UInt8](repeating: 0, count: 32)
        for g in stride(from: 0, to: values.count, by: 32) {
            var amax: Float = 0
            var max: Float = 0
            for j in g..<(g + 32) where amax < abs(values[j]) {
                amax = abs(values[j])
                max = values[j]
            }
            let d = max / -8
            let id: Float = d != 0 ? 1 / d : 0
            let bits = Float16(d).bitPattern
            scales.append(UInt8(truncatingIfNeeded: bits))
            scales.append(UInt8(truncatingIfNeeded: bits >> 8))
            for j in 0..<32 {
                q[j] = UInt8(Swift.min(15, Int8(values[g + j] * id + 8.5)))
            }
            for k in 0..<16 { weights.append(q[2 * k] | q[2 * k + 1] << 4) }
        }
    }

    /// Little-endian F32 → BF16, rounding to nearest even.
    package static func f32ToBF16(_ src: ArraySlice<UInt8>, out: inout [UInt8]) {
        precondition(src.count % 4 == 0)
        for i in stride(from: src.startIndex, to: src.endIndex, by: 4) {
            let bits = UInt32(src[i]) | UInt32(src[i + 1]) << 8 | UInt32(src[i + 2]) << 16 | UInt32(src[i + 3]) << 24
            let rounded = UInt16(truncatingIfNeeded: (bits &+ 0x7FFF &+ ((bits >> 16) & 1)) >> 16)
            out.append(UInt8(truncatingIfNeeded: rounded))
            out.append(UInt8(truncatingIfNeeded: rounded >> 8))
        }
    }

    /// Little-endian F32 rows → affine8: per 64 values, `bias = min` and
    /// `scale = (max - min) / 255` rounded to BF16, then
    /// `q = round((x - bias) / scale)` clamped to 0...255.
    package static func f32ToAffine8(_ src: ArraySlice<UInt8>, cols: Int, weights: inout [UInt8],
                                     scales: inout [UInt8], biases: inout [UInt8]) {
        precondition(cols % 64 == 0 && src.count % (cols * 4) == 0)
        func bf16(_ x: Float) -> UInt16 {
            let bits = x.bitPattern
            return UInt16(truncatingIfNeeded: (bits &+ 0x7FFF &+ ((bits >> 16) & 1)) >> 16)
        }
        func float(_ b: UInt16) -> Float { Float(bitPattern: UInt32(b) << 16) }
        func append(_ b: UInt16, to out: inout [UInt8]) {
            out.append(UInt8(truncatingIfNeeded: b))
            out.append(UInt8(truncatingIfNeeded: b >> 8))
        }
        var group = [Float](repeating: 0, count: 64)
        for g in stride(from: src.startIndex, to: src.endIndex, by: 256) {
            for j in 0..<64 {
                let i = g + 4 * j
                group[j] = Float(bitPattern: UInt32(src[i]) | UInt32(src[i + 1]) << 8
                                 | UInt32(src[i + 2]) << 16 | UInt32(src[i + 3]) << 24)
            }
            let lo = group.min() ?? 0
            let hi = group.max() ?? 0
            let s = bf16((hi - lo) / 255)
            let b = bf16(lo)
            append(s, to: &scales)
            append(b, to: &biases)
            let scale = float(s)
            let bias = float(b)
            for x in group {
                let q = scale > 0 ? ((x - bias) / scale).rounded() : 0
                weights.append(UInt8(Swift.max(0, Swift.min(255, q))))
            }
        }
    }
}
