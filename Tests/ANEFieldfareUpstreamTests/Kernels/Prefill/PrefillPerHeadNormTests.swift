import Metal
import Testing
@testable import ANEFieldfareHotpath
import ANEFieldfareValidationSupport

/// Per-head RMS norms against a CPU reference, in place and not, with padded
/// token strides: head sizes that are multiples of 32 take the SIMD-row
/// kernels, 80 the threadgroup-per-row ones.
@Suite struct PrefillPerHeadNormTests {
    @Test(arguments: [(128, true), (256, false), (512, true), (96, false), (80, true)])
    func perHeadNormMatchesReference(headDim: Int, inPlace: Bool) throws {
        let t = 9, heads = 5, stride = heads * headDim + 16, eps: Float = 1e-6
        let x = (0..<(t * stride)).map { Float16(Float(($0 * 37) % 101 - 50) * 0.03) }
        let weightBits = (0..<headDim).map { Quantization.bf16Bits(0.5 + Float($0 % 13) * 0.07) }
        let ctx = try MetalContext()
        let norm = try PrefillPerHeadNorm(context: ctx)
        for scaled in [true, false] {
            guard let input = Fp16Buffer.make(ctx.device, halves: x),
                  let output = Fp16Buffer.make(ctx.device, halves: [Float16](repeating: -7, count: t * stride)),
                  let weight = unsafe ctx.device.makeBuffer(bytes: weightBits, length: headDim * 2,
                                                            options: .storageModeShared),
                  let cb = ctx.queue.makeCommandBuffer() else {
                Issue.record("allocation failed")
                return
            }
            let out = inPlace ? input : output
            if scaled {
                try norm.encodeBF16W(commandBuffer: cb, x: input, weight: weight, weightOffset: 0, out: out,
                                     queryCount: UInt32(t), headDim: UInt32(headDim), numHeads: UInt32(heads),
                                     tokenStrideElements: UInt32(stride), eps: eps)
            } else {
                try norm.encodeNoScale(commandBuffer: cb, x: input, out: out,
                                       queryCount: UInt32(t), headDim: UInt32(headDim), numHeads: UInt32(heads),
                                       tokenStrideElements: UInt32(stride), eps: eps)
            }
            cb.commit()
            cb.waitUntilCompleted()
            #expect(cb.error == nil)
            let actual = Fp16Buffer.readHalf(out, count: t * stride)
            var maxError: Float = 0
            for token in 0..<t {
                for h in 0..<heads {
                    let base = token * stride + h * headDim
                    let row = (0..<headDim).map { Float(x[base + $0]) }
                    let inv = 1 / (row.map { $0 * $0 }.reduce(0, +) / Float(headDim) + eps).squareRoot()
                    for d in 0..<headDim {
                        let w = scaled ? Quantization.bf16ToFloat(weightBits[d]) : 1
                        maxError = max(maxError, abs(Float(actual[base + d]) - row[d] * inv * w))
                    }
                }
                // The padding past the heads is untouched.
                for d in (heads * headDim)..<stride {
                    #expect(actual[token * stride + d] == (inPlace ? x[token * stride + d] : -7))
                }
            }
            #expect(maxError <= 4e-3, "headDim=\(headDim) scaled=\(scaled) maxError=\(maxError)")
        }
    }
}
