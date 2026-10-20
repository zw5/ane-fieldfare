import Testing
import Foundation
import Metal
@testable import ANEFieldfareHotpath
import ANEFieldfareValidationSupport

@Suite struct FusedPostAttentionSetupTests {
    private static let d = ArchConfig.gemma4_26B_A4B.hiddenSize
    private static let eps: Float = 1e-6

    @Test func fusedPostAttentionSetup_matchesPrimitiveChainBitwise_realShape() throws {
        var rng = SplitMix64(seed: 0x9057_A77E)
        let hidden = (0..<Self.d).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let attn = (0..<Self.d).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let wPost = (0..<Self.d).map { _ in Quantization.bf16Bits(rng.uniform(0.5, 1.5)) }
        let wPre = (0..<Self.d).map { _ in Quantization.bf16Bits(rng.uniform(0.5, 1.5)) }
        let wPre2 = (0..<Self.d).map { _ in Quantization.bf16Bits(rng.uniform(0.5, 1.5)) }

        let ctx = try MetalContext()
        let rms = try RMSNorm(context: ctx)
        let fused = try FusedPostAttentionSetup(context: ctx)

        func buffer<T: BitwiseCopyable>(_ values: [T]) throws -> any MTLBuffer {
            try #require(unsafe ctx.device.makeBuffer(bytes: values,
                                                      length: values.count * MemoryLayout<T>.stride,
                                                      options: .storageModeShared))
        }
        func scratch() throws -> any MTLBuffer {
            try #require(ctx.device.makeBuffer(length: Self.bytes(Self.d), options: .storageModeShared))
        }
        let hiddenLegacy = try buffer(hidden), attnLegacy = try buffer(attn)
        let denseLegacy = try scratch(), routedLegacy = try scratch(), routerLegacy = try scratch()
        let hiddenFused = try buffer(hidden), attnFused = try buffer(attn)
        let denseFused = try scratch(), routedFused = try scratch(), routerFused = try scratch()
        let wPostBuf = try buffer(wPost), wPreBuf = try buffer(wPre), wPre2Buf = try buffer(wPre2)
        // The router norm has no weight; a unit BF16 weight leaves it exact.
        let unitBuf = try buffer([UInt16](repeating: Quantization.bf16Bits(1.0), count: Self.d))

        // The legacy residual add runs on the CPU between two command buffers.
        let attentionNormCB = try #require(ctx.queue.makeCommandBuffer())
        rms.encodeBF16W(commandBuffer: attentionNormCB,
                        x: attnLegacy,
                        weight: wPostBuf,
                        out: attnLegacy,
                        d: UInt32(Self.d),
                        eps: Self.eps)
        attentionNormCB.commit()
        attentionNormCB.waitUntilCompleted()
        if let error = attentionNormCB.error {
            Issue.record("Command buffer failed: \(error)")
            return
        }

        let hiddenPointer = unsafe hiddenLegacy.contents().assumingMemoryBound(to: Float16.self)
        let attentionPointer = unsafe attnLegacy.contents().assumingMemoryBound(to: Float16.self)
        for index in 0..<Self.d {
            unsafe hiddenPointer[index] += attentionPointer[index]
        }

        let cb = try #require(ctx.queue.makeCommandBuffer())
        rms.encodeBF16W(commandBuffer: cb,
                        x: hiddenLegacy,
                        weight: wPreBuf,
                        out: denseLegacy,
                        d: UInt32(Self.d),
                        eps: Self.eps)
        rms.encodeBF16W(commandBuffer: cb,
                        x: hiddenLegacy,
                        weight: wPre2Buf,
                        out: routedLegacy,
                        d: UInt32(Self.d),
                        eps: Self.eps)
        rms.encodeBF16W(commandBuffer: cb,
                        x: hiddenLegacy,
                        weight: unitBuf,
                        out: routerLegacy,
                        d: UInt32(Self.d),
                        eps: Self.eps)
        fused.encode(commandBuffer: cb,
                     hidden: hiddenFused,
                     attn: attnFused,
                     denseX: denseFused,
                     routedX: routedFused,
                     routerX: routerFused,
                     postAttentionWeight: wPostBuf,
                     preFFNWeight: wPreBuf,
                     preFFN2Weight: wPre2Buf,
                     d: UInt32(Self.d),
                     eps: Self.eps)
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error {
            Issue.record("Command buffer failed: \(err)")
            return
        }

        #expect(Self.bufferBytes(hiddenLegacy) == Self.bufferBytes(hiddenFused))
        #expect(Self.bufferBytes(denseLegacy) == Self.bufferBytes(denseFused))
        #expect(Self.bufferBytes(routedLegacy) == Self.bufferBytes(routedFused))
        #expect(Self.bufferBytes(routerLegacy) == Self.bufferBytes(routerFused))
    }

    private static func bytes(_ count: Int) -> Int {
        count * MemoryLayout<Float16>.size
    }

    private static func bufferBytes(_ buffer: any MTLBuffer) -> [UInt8] {
        let ptr = unsafe buffer.contents().assumingMemoryBound(to: UInt8.self)
        return unsafe Array(UnsafeBufferPointer(start: ptr, count: Self.bytes(Self.d)))
    }
}
