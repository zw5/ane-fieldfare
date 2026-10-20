import Foundation
import Accelerate
public import ANEFieldfareHotpath

/// FP32 reference for the INT4-affine groupwise GEMV `y = W * x`: dequantize
/// each row, then `vDSP_dotpr`, so staging and summation order differ from
/// the kernel's fused inner loop.
public enum DequantInt4GemvRef {
    public static func apply(
        weightRows: [Quantization.Int4AffineRow],
        x: [Float],
        n: Int
    ) -> [Float] {
        precondition(!weightRows.isEmpty)
        precondition(x.count == n)
        precondition(n % Quantization.groupSize == 0,
                     "N must be a multiple of \(Quantization.groupSize)")

        let m = weightRows.count
        var y = [Float](repeating: 0, count: m)

        for row in 0..<m {
            let wRow = Quantization.dequantizeInt4Affine(weightRows[row], n: n)
            var dot: Float = 0
            wRow.withUnsafeBufferPointer { pw in
                x.withUnsafeBufferPointer { px in
                    unsafe vDSP_dotpr(
                        pw.baseAddress!, 1,
                        px.baseAddress!, 1,
                        &dot,
                        vDSP_Length(n)
                    )
                }
            }
            y[row] = dot
        }
        return y
    }
}
