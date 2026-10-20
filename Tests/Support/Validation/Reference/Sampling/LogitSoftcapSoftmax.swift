import Foundation
import Accelerate

/// FP32 two-pass reference for `softmax(softcap * tanh(x / softcap))`; the
/// kernel is single-pass online softmax, so its merge logic is not mirrored.
public enum LogitSoftcapSoftmaxRef {
    public static func apply(x: [Float], softcap: Float) -> [Float] {
        let v = x.count
        let invC = 1.0 / softcap

        // 1. Apply softcap: y = softcap * tanh(x / softcap)
        //    Two-step: y = x * invC; y = tanh(y); y = y * softcap.
        var y = [Float](repeating: 0, count: v)
        var s = invC
        x.withUnsafeBufferPointer { px in
            y.withUnsafeMutableBufferPointer { py in
                unsafe vDSP_vsmul(px.baseAddress!, 1, &s, py.baseAddress!, 1, vDSP_Length(v))
            }
        }
        y = vForce.tanh(y)
        var c = softcap
        y.withUnsafeMutableBufferPointer { py in
            unsafe vDSP_vsmul(py.baseAddress!, 1, &c, py.baseAddress!, 1, vDSP_Length(v))
        }

        // 2. Numerically stable softmax: subtract max, exp, divide by sum.
        var mx: Float = -.infinity
        y.withUnsafeBufferPointer { py in
            unsafe vDSP_maxv(py.baseAddress!, 1, &mx, vDSP_Length(v))
        }
        var negMax = -mx
        y.withUnsafeMutableBufferPointer { py in
            unsafe vDSP_vsadd(py.baseAddress!, 1, &negMax, py.baseAddress!, 1, vDSP_Length(v))
        }
        y = vForce.exp(y)

        var sum: Float = 0
        y.withUnsafeBufferPointer { py in
            unsafe vDSP_sve(py.baseAddress!, 1, &sum, vDSP_Length(v))
        }
        var invSum = 1.0 / sum
        y.withUnsafeMutableBufferPointer { py in
            unsafe vDSP_vsmul(py.baseAddress!, 1, &invSum, py.baseAddress!, 1, vDSP_Length(v))
        }
        return y
    }
}
