import Foundation
import Accelerate

/// FP32 RMSNorm reference on Accelerate, so its summation tree differs from
/// the kernel's `simd_sum` block reduction.
public enum RmsNormRef {
    /// y[i] = x[i] * weight[i] * 1 / sqrt(mean(x^2) + eps)
    public static func apply(x: [Float], weight: [Float], eps: Float) -> [Float] {
        precondition(x.count == weight.count, "x and weight must match length")
        let d = x.count

        var sumSq: Float = 0
        x.withUnsafeBufferPointer { p in
            unsafe vDSP_svesq(p.baseAddress!, 1, &sumSq, vDSP_Length(d))
        }
        let invRms = 1.0 / (sumSq / Float(d) + eps).squareRoot()

        var y = [Float](repeating: 0, count: d)
        x.withUnsafeBufferPointer { px in
            weight.withUnsafeBufferPointer { pw in
                y.withUnsafeMutableBufferPointer { py in
                    unsafe vDSP_vmul(
                        px.baseAddress!, 1,
                        pw.baseAddress!, 1,
                        py.baseAddress!, 1,
                        vDSP_Length(d)
                    )
                    var s = invRms
                    unsafe vDSP_vsmul(
                        py.baseAddress!, 1,
                        &s,
                        py.baseAddress!, 1,
                        vDSP_Length(d)
                    )
                }
            }
        }
        return y
    }
}
