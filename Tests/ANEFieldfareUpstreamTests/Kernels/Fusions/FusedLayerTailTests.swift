import Testing
import Foundation
import Metal
@testable import ANEFieldfareHotpath
import ANEFieldfareValidationSupport

@Suite struct FusedLayerTailTests {
    private static let d = ArchConfig.gemma4_26B_A4B.hiddenSize
    private static let eps: Float = 1e-6

    /// The norm-fusing tail against the three dispatches it replaces in
    /// decode (post-FFN-1 norm of h1, the tail, the next layer's input norm),
    /// bitwise; and the separate tail against a CPU reference.
    @Test func fusedLayerTailNorms_matchSeparateDispatchesAndReference() throws {
        var rng = SplitMix64(seed: 0x7A11_0001)
        func halves(_ lo: Float, _ hi: Float) -> [Float16] { (0..<Self.d).map { _ in Float16(rng.uniform(lo, hi)) } }
        func bf16(_ lo: Float, _ hi: Float) -> [UInt16] { (0..<Self.d).map { _ in Quantization.bf16Bits(rng.uniform(lo, hi)) } }
        let h2 = halves(-1, 1), h1 = halves(-3, 3), hidden = halves(-2, 2)
        let weights = [bf16(0.5, 1.5), bf16(0.5, 1.5), bf16(0.5, 1.5), bf16(0.5, 1.5)]  // post 1, post 2, post, next input
        let layerScalar = rng.uniform(0.75, 1.25)

        let ctx = try MetalContext()
        let tail = try FusedLayerTail(context: ctx)
        let rms = try RMSNorm(context: ctx)
        try #require(tail.fusesNorms)
        func buffer<T: BitwiseCopyable>(_ values: [T]) throws -> any MTLBuffer {
            try #require(unsafe ctx.device.makeBuffer(bytes: values, length: values.count * MemoryLayout<T>.stride,
                                                      options: .storageModeShared))
        }
        func view(_ b: any MTLBuffer) -> TensorView {
            TensorView(buffer: b, offset: 0, length: UInt64(b.length), scaleOffset: 0, scaleLength: 0,
                       biasOffset: 0, biasLength: 0, shape: (UInt32(Self.d), 1, 1, 1), dtype: 1)
        }
        func halves(of b: any MTLBuffer) -> [UInt16] {
            (0..<Self.d).map { unsafe b.contents().load(fromByteOffset: $0 * 2, as: UInt16.self) }
        }
        let w = try weights.map(buffer)
        let d = UInt32(Self.d)

        // Separate: rmsnorm(h1) in place, tail, rmsnorm(hidden) -> normed.
        let sepH2 = try buffer(h2), sepH1 = try buffer(h1), sepHidden = try buffer(hidden)
        let sepNormed = try buffer([Float16](repeating: 0, count: Self.d))
        let cb1 = try #require(ctx.queue.makeCommandBuffer())
        rms.encodeBF16W(commandBuffer: cb1, x: sepH1, weight: w[0], out: sepH1, d: d, eps: Self.eps)
        tail.encode(commandBuffer: cb1, h2: sepH2, h1: sepH1, hidden: sepHidden,
                    postFFN2Weight: w[1], postFFNWeight: w[2], d: d, eps: Self.eps, layerScalar: layerScalar)
        rms.encodeBF16W(commandBuffer: cb1, x: sepHidden, weight: w[3], out: sepNormed, d: d, eps: Self.eps)
        cb1.commit()
        cb1.waitUntilCompleted()
        if let error = cb1.error { throw error }

        // The tail's CPU oracle, fed the reference post-FFN-1 norm of h1.
        let h1Normed = RmsNormRef.apply(x: h1.map(Float.init),
                                        weight: weights[0].map(Quantization.bf16ToFloat),
                                        eps: Self.eps).map { Float16($0) }
        let ref = Self.cpuReference(h2: h2, h1: h1Normed, hidden: hidden,
                                    wPostFFN2: weights[1], wPostFFN: weights[2], layerScalar: layerScalar)
        let separate = halves(of: sepHidden).map { Float(Float16(bitPattern: $0)) }
        let rel = RelError.compute(actual: separate, reference: ref)
        #expect(rel < 4e-3, "tail vs CPU reference rel=\(rel)")

        // Fused.
        let h2Buf = try buffer(h2), h1Buf = try buffer(h1), hiddenBuf = try buffer(hidden)
        let normed = try buffer([Float16](repeating: 0, count: Self.d))
        let cb2 = try #require(ctx.queue.makeCommandBuffer())
        tail.encodeWithNorms(commandBuffer: cb2, h2: h2Buf, h1: h1Buf, hidden: hiddenBuf,
                             postFFN1Weight: view(w[0]), postFFN2Weight: view(w[1]), postFFNWeight: view(w[2]),
                             next: (view(w[3]), normed),
                             d: d, eps: Self.eps, layerScalar: layerScalar)
        cb2.commit()
        cb2.waitUntilCompleted()
        if let error = cb2.error { throw error }

        #expect(halves(of: hiddenBuf) == halves(of: sepHidden))
        #expect(halves(of: normed) == halves(of: sepNormed))
        // The fused tail leaves h1 as it found it: a patch reruns the tail from it.
        #expect(halves(of: h1Buf) == h1.map(\.bitPattern))
    }

    private static func cpuReference(h2: [Float16],
                                     h1: [Float16],
                                     hidden: [Float16],
                                     wPostFFN2: [UInt16],
                                     wPostFFN: [UInt16],
                                     layerScalar: Float) -> [Float] {
        let w2 = wPostFFN2.map { Quantization.bf16ToFloat($0) }
        let w = wPostFFN.map { Quantization.bf16ToFloat($0) }
        let h2Norm = RmsNormRef.apply(x: h2.map(Float.init), weight: w2, eps: eps)
            .map { Float(Float16($0)) }
        let h12 = zip(h1, h2Norm).map { Float(Float16(Float($0.0) + $0.1)) }
        let h12Norm = RmsNormRef.apply(x: h12, weight: w, eps: eps)
            .map { Float(Float16($0)) }
        let hScale = Float(Float16(layerScalar))
        return zip(hidden, h12Norm).map { Float(Float16(Float(Float16(Float($0.0) + $0.1)) * hScale)) }
    }
}
