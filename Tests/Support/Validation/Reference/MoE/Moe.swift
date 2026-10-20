import Foundation
import Accelerate
public import ANEFieldfareHotpath

/// FP32 reference for the MoE FFN: gate / up / down INT4-affine projections
/// with GeGLU, composed from `DequantInt4GemvRef`. It materializes the F-dim
/// intermediate per expert, so its summation order differs from the kernel's.
public enum MoeRef {
    static let geluCoeff: Float = 0.7978845608028654  // sqrt(2/π)
    static let geluCubic: Float = 0.044715

    /// Per-element `gelu_pytorch_tanh(x) = 0.5 * x * (1 + tanh(c * (x + k * x^3)))`.
    public static func geluTanh(_ x: [Float]) -> [Float] {
        var y = [Float](repeating: 0, count: x.count)
        for i in 0..<x.count {
            let xv = x[i]
            let inner = geluCoeff * (xv + geluCubic * xv * xv * xv)
            y[i] = 0.5 * xv * (1.0 + Foundation.tanh(inner))
        }
        return y
    }

    /// Runs one FFN block: `down(gelu(gate(x)) * up(x))`.
    /// `gateRows` / `upRows` are F-by-D affine INT4. `downRows` is D-by-F.
    public static func runFFN(
        gateRows: [Quantization.Q4Row],
        upRows:   [Quantization.Q4Row],
        downRows: [Quantization.Q4Row],
        x: [Float],
        d: Int,
        f: Int
    ) -> [Float] {
        precondition(gateRows.count == f, "gateRows must be F=\(f)")
        precondition(upRows.count == f, "upRows must be F=\(f)")
        precondition(downRows.count == d, "downRows must be D=\(d)")
        precondition(x.count == d)

        let gateOut = DequantInt4GemvRef.apply(weightRows: gateRows, x: x, n: d)
        let upOut   = DequantInt4GemvRef.apply(weightRows: upRows,   x: x, n: d)
        let gated   = geluTanh(gateOut)
        var act = [Float](repeating: 0, count: f)
        unsafe vDSP_vmul(gated, 1, upOut, 1, &act, 1, vDSP_Length(f))
        return DequantInt4GemvRef.apply(weightRows: downRows, x: act, n: f)
    }

    /// Routed half of Gemma 4's parallel-MoE block:
    /// `y = residual + Σ wₑ · routed_e(x)`.
    public static func applyStreamedRouted(
        x: [Float],
        residual: [Float],
        routedGate: [[Quantization.Q4Row]],
        routedUp:   [[Quantization.Q4Row]],
        routedDown: [[Quantization.Q4Row]],
        indices: [Int],
        routingWeights: [Float],
        d: Int,
        f: Int
    ) -> [Float] {
        precondition(indices.count == routingWeights.count)
        precondition(residual.count == d)

        var y = residual
        for slot in 0..<indices.count {
            let e = indices[slot]
            let w = routingWeights[slot]
            let out = runFFN(
                gateRows: routedGate[e],
                upRows:   routedUp[e],
                downRows: routedDown[e],
                x: x, d: d, f: f
            )
            var scale = w
            var scaled = [Float](repeating: 0, count: d)
            unsafe vDSP_vsmul(out, 1, &scale, &scaled, 1, vDSP_Length(d))
            unsafe vDSP_vadd(y, 1, scaled, 1, &y, 1, vDSP_Length(d))
        }
        return y
    }
}
