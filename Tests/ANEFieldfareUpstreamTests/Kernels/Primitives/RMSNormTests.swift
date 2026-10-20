import Testing
import Foundation
import Metal
@testable import ANEFieldfareHotpath
import ANEFieldfareValidationSupport

/// Compares the Metal `rmsnorm` kernel against the Accelerate-based
/// `RmsNormRef`. The reference uses `vDSP_svesq` + `vDSP_vmul` + `vDSP_vsmul`
/// (Accelerate's pipeline). The kernel uses per-thread block reduction with
/// `simd_sum`. Different summation tree, different rounding chain — bugs in
/// either are unlikely to match the other.
@Suite struct RMSNormTests {

    private static let eps: Float = 1e-6

    /// Every Gemma 4 norm weight is BF16; the reference reads the same
    /// BF16-rounded weights. D = 2816 is the hidden size and not a multiple of
    /// the reduction block.
    @Test func rmsNorm_matchesReference_hiddenSize() throws {
        let d = 2816
        var rng = SeedTree(0xD4).key("rmsnorm-d\(d)-weighted=true")
        let x = (0..<d).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let wBits = (0..<d).map { _ in Quantization.bf16Bits(rng.uniform(0.5, 1.5)) }

        let ctx = try MetalContext()
        let kernel = try RMSNorm(context: ctx)
        let xBuf = try #require(Fp16Buffer.make(ctx.device, halves: x))
        let yBuf = try #require(Fp16Buffer.make(ctx.device, count: d))
        let wBuf = try #require(unsafe ctx.device.makeBuffer(bytes: wBits, length: wBits.count * 2,
                                                             options: .storageModeShared))
        let cb = try #require(ctx.queue.makeCommandBuffer())
        kernel.encodeBF16W(commandBuffer: cb, x: xBuf, weight: wBuf, out: yBuf,
                           d: UInt32(d), eps: Self.eps)
        cb.commit(); cb.waitUntilCompleted()

        let ref = RmsNormRef.apply(x: x.map { Float($0) },
                                   weight: wBits.map { Quantization.bf16ToFloat($0) },
                                   eps: Self.eps)
        let actual = Fp16Buffer.read(yBuf, count: d)
        let relErr = RelError.compute(actual: actual, reference: ref)
        let maxAbs = RelError.maxAbsDiff(actual, ref)
        #expect(relErr < Tolerance.fp16Reduction,
                "relErr=\(relErr) maxAbsDiff=\(maxAbs)")
    }
}
