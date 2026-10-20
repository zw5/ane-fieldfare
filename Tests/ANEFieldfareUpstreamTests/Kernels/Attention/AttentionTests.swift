import Testing
import Foundation
import Metal
@testable import ANEFieldfareHotpath
import ANEFieldfareValidationSupport

/// Compares the Metal `attention` kernel against `AttentionRef`, which
/// materializes the full attention matrix per Q head via `vDSP_dotpr` and
/// then softmaxes/outputs. The kernel runs FlashAttention-style tiled
/// online softmax; this reference does no online merging. Different code
/// shape, different summation order.
///
/// Gemma 4 decode passes `scale: 1.0` on both layer kinds, so kernel and
/// reference run at that scale: a kernel that ignored the argument would fall
/// back to rsqrt(head_dim) and miss the reference.
@Suite struct AttentionTests {

    /// `window` nil is full attention. `ringCapacity` > 0 stores position `p`
    /// at slot `p % ringCapacity`, as the decode KV ring does once it wraps;
    /// the reference always reads the linear K/V.
    private static func runAndCompare(headDim: Int,
                                      numQHeads: Int,
                                      numKVHeads: Int,
                                      seqLen: Int,
                                      window: Int?,
                                      ringCapacity: Int = 0,
                                      seed: UInt64) throws {
        var rng = SeedTree(seed).key("attn-h\(numQHeads)-kv\(numKVHeads)-d\(headDim)-T\(seqLen)")
        let qCount = numQHeads * headDim
        let kvStride = numKVHeads * headDim
        let kvCount = seqLen * kvStride
        let q = (0..<qCount).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let k = (0..<kvCount).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let v = (0..<kvCount).map { _ in Float16(rng.uniform(-0.5, 0.5)) }

        var kStored = k
        var vStored = v
        if ringCapacity > 0 {
            kStored = [Float16](repeating: 0, count: ringCapacity * kvStride)
            vStored = kStored
            for p in 0..<seqLen {
                let dst = (p % ringCapacity) * kvStride
                let src = p * kvStride
                kStored.replaceSubrange(dst..<(dst + kvStride), with: k[src..<(src + kvStride)])
                vStored.replaceSubrange(dst..<(dst + kvStride), with: v[src..<(src + kvStride)])
            }
        }

        let ctx = try MetalContext()
        let kernel = try Attention(context: ctx)
        let qBuf = try #require(Fp16Buffer.make(ctx.device, halves: q))
        let kBuf = try #require(Fp16Buffer.make(ctx.device, halves: kStored))
        let vBuf = try #require(Fp16Buffer.make(ctx.device, halves: vStored))
        let outBuf = try #require(Fp16Buffer.make(ctx.device, count: qCount))
        let cb = try #require(ctx.queue.makeCommandBuffer())
        if let window {
            kernel.encodeSWA(commandBuffer: cb,
                             q: qBuf, k: kBuf, v: vBuf, out: outBuf,
                             headDim: UInt32(headDim),
                             numQHeads: UInt32(numQHeads),
                             numKVHeads: UInt32(numKVHeads),
                             seqLen: UInt32(seqLen),
                             window: UInt32(window),
                             scale: 1.0,
                             ringCapacity: UInt32(ringCapacity))
        } else {
            kernel.encodeFull(commandBuffer: cb,
                              q: qBuf, k: kBuf, v: vBuf, out: outBuf,
                              headDim: UInt32(headDim),
                              numQHeads: UInt32(numQHeads),
                              numKVHeads: UInt32(numKVHeads),
                              seqLen: UInt32(seqLen),
                              scale: 1.0)
        }
        cb.commit()
        cb.waitUntilCompleted()

        let ref = AttentionRef.apply(q: q.map { Float($0) },
                                     k: k.map { Float($0) },
                                     v: v.map { Float($0) },
                                     headDim: headDim,
                                     numQHeads: numQHeads,
                                     numKVHeads: numKVHeads,
                                     seqLen: seqLen,
                                     window: window,
                                     scale: 1.0)
        let actual = Fp16Buffer.read(outBuf, count: qCount)
        let rel = RelError.compute(actual: actual, reference: ref)
        #expect(rel < Tolerance.fp16ChainedReduction,
                "window=\(String(describing: window)) ring=\(ringCapacity) T=\(seqLen) rel=\(rel)")
    }

    @Test func attentionSWA_shorterThanWindow() throws {
        try Self.runAndCompare(headDim: 64, numQHeads: 4, numKVHeads: 2,
                               seqLen: 32, window: 128, seed: 0x172)
    }

    /// The production SWA shape past the ring's wrap: 1187 positions in a
    /// 1152-slot ring, read through a 1024 window.
    @Test func attentionSWA_ringWrapRealShape() throws {
        try Self.runAndCompare(headDim: 256, numQHeads: 16, numKVHeads: 8,
                               seqLen: 1187, window: 1024, ringCapacity: 1152, seed: 0x181)
    }

    @Test func attentionFull_realShape() throws {
        try Self.runAndCompare(headDim: 512, numQHeads: 16, numKVHeads: 2,
                               seqLen: 128, window: nil, seed: 0x175)
    }
}
