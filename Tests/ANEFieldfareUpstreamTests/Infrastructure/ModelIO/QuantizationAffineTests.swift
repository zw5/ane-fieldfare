import Testing
import Foundation
import ANEFieldfareHotpath
import ANEFieldfareValidationSupport

@Suite struct QuantizationAffineTests {
    @Test func quantizeDequantizeInt4Affine_roundtrip() {
        var row = [Float]()
        row.reserveCapacity(128)
        for k in 0..<64 { row.append(0.10 + Float(k) * 0.01) }   // 0.10 .. 0.73 (positive-only, bias > 0)
        for k in 0..<64 { row.append(-0.50 - Float(k) * 0.005) } // -0.50 .. -0.815 (negative-only, bias < 0)

        let q = Quantization.quantizeInt4Affine(row)
        #expect(q.packed.count == 64)
        #expect(q.scales.count == 2)
        #expect(q.biases.count == 2)

        let r = Quantization.dequantizeInt4Affine(q, n: 128)
        // 4-bit affine, group-0 scale ≈ (0.73 - 0.10) / 15 ≈ 0.042; group-1 scale
        // ≈ 0.315/15 ≈ 0.021. Worst-case rounding ≤ scale/2 plus BF16 rounding
        // of the scale (rel ≤ 2^-7) — bound under 0.025.
        for i in 0..<128 {
            #expect(abs(r[i] - row[i]) < 0.025,
                    "i=\(i) ref=\(row[i]) got=\(r[i])")
        }
    }
}
