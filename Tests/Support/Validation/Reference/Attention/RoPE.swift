import Foundation
import Accelerate

/// FP32 RoPE reference. Computes the `(cos, sin)` table once with vForce
/// before touching any input, unlike the kernel's per-pair `cosf`/`sinf`.
public enum RopeRef {
    /// NeoX-convention RoPE. Pairs `(x[i], x[i+headDim/2])` for `i ∈ [0, rotatedPairs)`.
    /// Frequencies divide by `headDim` (not `2 * rotatedPairs`) — matching HF
    /// Gemma 4's proportional-RoPE init. Setting `rotatedPairs == headDim/2`
    /// recovers full NeoX rotation.
    public static func applyNeox(
        input: [Float],
        numTokens: Int,
        numHeads: Int,
        headDim: Int,
        rotatedPairs: Int,
        position: Int,
        theta: Float
    ) -> [Float] {
        precondition(input.count == numTokens * numHeads * headDim,
                     "input size mismatch")
        precondition(rotatedPairs * 2 <= headDim, "rotatedPairs * 2 must not exceed headDim")
        let halfDim = headDim / 2
        var angles = [Float](repeating: 0, count: rotatedPairs)
        let logTheta = Foundation.log(theta)
        let positionF = Float(position)
        for i in 0..<rotatedPairs {
            let exponent = -Float(2 * i) / Float(headDim)
            angles[i] = positionF * Foundation.exp(exponent * logTheta)
        }
        let cosTable = vForce.cos(angles)
        let sinTable = vForce.sin(angles)

        var out = input
        for t in 0..<numTokens {
            for h in 0..<numHeads {
                let base = (t * numHeads + h) * headDim
                for i in 0..<rotatedPairs {
                    let i0 = base + i
                    let i1 = base + halfDim + i
                    let x0 = input[i0]
                    let x1 = input[i1]
                    let c = cosTable[i]
                    let s = sinTable[i]
                    out[i0] = x0 * c - x1 * s
                    out[i1] = x0 * s + x1 * c
                }
            }
        }
        return out
    }
}
