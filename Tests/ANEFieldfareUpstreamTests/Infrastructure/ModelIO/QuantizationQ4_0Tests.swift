import Testing
import Foundation
import ANEFieldfareHotpath
import ANEFieldfareValidationSupport

@Suite struct QuantizationQ4_0Tests {
    @Test func quantizeDequantizeQ4_0_roundtrip() {
        var row = [Float]()
        row.reserveCapacity(128)
        for k in 0..<64 { row.append(0.10 + Float(k) * 0.01) }   // 0.10 .. 0.73
        for k in 0..<64 { row.append(-0.50 - Float(k) * 0.005) } // -0.50 .. -0.815

        let q = Quantization.quantizeQ4_0(row)
        #expect(q.packed.count == 64)
        #expect(q.scales.count == 4)

        let r = Quantization.dequantizeQ4_0(q, n: 128)
        // Symmetric 4-bit: the step is |max| / 8 per group of 32, at most
        // 0.815 / 8 ≈ 0.102; rounding error is under half a step, plus a
        // full step on the clamped side.
        for i in 0..<128 {
            let step = abs(row[(i / 32) * 32..<((i / 32) + 1) * 32].max { abs($0) < abs($1) } ?? 0) / 8
            #expect(abs(r[i] - row[i]) <= step * 1.01,
                    "i=\(i) ref=\(row[i]) got=\(r[i])")
        }
        // The largest magnitude of each group is exact.
        #expect(r[63] == Float(Float16(row[63])) || abs(r[63] - row[63]) < 1e-3)
    }
}
