import Testing
import Foundation
import Metal
@testable import ANEFieldfareHotpath
import ANEFieldfareValidationSupport

@Suite struct PrefillQKVEpilogueTests {
    @Test func blockQKVEpilogueMatchesReferenceSWA() throws {
        _ = try Self.expectMatchesReference(rows: 3,
                                                 numQHeads: 16,
                                                 numKVHeads: 8,
                                                 headDim: 256,
                                                 startPosition: 1023,
                                                 theta: 10_000.0,
                                                 rotatedPairs: 128,
                                                 sharedKVRaw: false,
                                                 seed: 0x7A50_5A01)
    }

    @Test func blockQKVEpiloguePreservesFullLayerSharedRawKVQuirk() throws {
        let result = try Self.expectMatchesReference(rows: 2,
                                                          numQHeads: 16,
                                                          numKVHeads: 2,
                                                          headDim: 512,
                                                          startPosition: 4095,
                                                          theta: 1_000_000.0,
                                                          rotatedPairs: 64,
                                                          sharedKVRaw: true,
                                                          seed: 0x7A50_F011)
        #expect(result.kUsed != result.vUsed,
                "full-layer K/V must diverge after K norm+RoPE and V no-scale norm")
    }

    private static func expectMatchesReference(rows: Int,
                                                    numQHeads: Int,
                                                    numKVHeads: Int,
                                                    headDim: Int,
                                                    startPosition: Int,
                                                    theta: Float,
                                                    rotatedPairs: UInt32,
                                                    sharedKVRaw: Bool,
                                                    seed: UInt64) throws -> (kUsed: [Float], vUsed: [Float]) {
        let qUsed = numQHeads * headDim
        let kvUsed = numKVHeads * headDim
        let qStride = qUsed + 13
        let kvStride = kvUsed + 17
        let q = Self.makeBlock(rows: rows,
                               rowStride: qStride,
                               usedPerRow: qUsed,
                               seed: seed,
                               label: "q")
        let k = Self.makeBlock(rows: rows,
                               rowStride: kvStride,
                               usedPerRow: kvUsed,
                               seed: seed &+ 1,
                               label: "k")
        let v = sharedKVRaw
            ? k
            : Self.makeBlock(rows: rows,
                             rowStride: kvStride,
                             usedPerRow: kvUsed,
                             seed: seed &+ 2,
                             label: "v")
        var rng = SeedTree(seed).key("norm-weights")
        let qWeight = (0..<headDim).map { _ in Quantization.bf16Bits(rng.uniform(0.5, 1.5)) }
        let kWeight = (0..<headDim).map { _ in Quantization.bf16Bits(rng.uniform(0.5, 1.5)) }

        let ctx = try MetalContext()
        let block = try PrefillQKVEpilogue(context: ctx)
        guard
            let qWeightBuf = unsafe ctx.device.makeBuffer(bytes: qWeight,
                                                   length: qWeight.count * MemoryLayout<UInt16>.size,
                                                   options: .storageModeShared),
            let kWeightBuf = unsafe ctx.device.makeBuffer(bytes: kWeight,
                                                   length: kWeight.count * MemoryLayout<UInt16>.size,
                                                   options: .storageModeShared),
            let qBlock = Fp16Buffer.make(ctx.device, halves: q),
            let kBlock = Fp16Buffer.make(ctx.device, halves: k),
            let vBlock = Fp16Buffer.make(ctx.device, halves: v),
            let cb = ctx.queue.makeCommandBuffer()
        else {
            Issue.record("alloc failed")
            return ([], [])
        }

        block.encode(commandBuffer: cb,
                     q: qBlock,
                     k: kBlock,
                     v: vBlock,
                     qWeight: qWeightBuf,
                     kWeight: kWeightBuf,
                     startPosition: UInt32(startPosition),
                     queryCount: UInt32(rows),
                     headDim: UInt32(headDim),
                     numQHeads: UInt32(numQHeads),
                     numKVHeads: UInt32(numKVHeads),
                     qTokenStrideElements: UInt32(qStride),
                     kvTokenStrideElements: UInt32(kvStride),
                     theta: theta,
                     rotatedPairs: rotatedPairs,
                     eps: 1e-6)
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error {
            Issue.record("command buffer failed: \(err)")
            return ([], [])
        }

        var qRef: [Float] = [], kRef: [Float] = [], vRef: [Float] = []
        for row in 0..<rows {
            let rope = (position: startPosition + row, theta: theta, rotatedPairs: Int(rotatedPairs))
            func used(_ values: [Float16], _ stride: Int, _ count: Int) -> [Float] {
                values[(row * stride)..<(row * stride + count)].map { Float($0) }
            }
            qRef += QKVEpilogueReference.apply(used(q, qStride, qUsed), headDim: headDim,
                                               weight: qWeight, rope: rope)
            kRef += QKVEpilogueReference.apply(used(k, kvStride, kvUsed), headDim: headDim,
                                               weight: kWeight, rope: rope)
            vRef += QKVEpilogueReference.apply(used(v, kvStride, kvUsed), headDim: headDim,
                                               weight: nil, rope: nil)
        }
        let kActual = Self.usedRows(Fp16Buffer.read(kBlock, count: rows * kvStride),
                                    rows: rows, rowStride: kvStride, usedPerRow: kvUsed)
        let vActual = Self.usedRows(Fp16Buffer.read(vBlock, count: rows * kvStride),
                                    rows: rows, rowStride: kvStride, usedPerRow: kvUsed)
        Self.expectClose(Self.usedRows(Fp16Buffer.read(qBlock, count: rows * qStride),
                                       rows: rows, rowStride: qStride, usedPerRow: qUsed),
                         qRef, label: "Q")
        Self.expectClose(kActual, kRef, label: "K")
        Self.expectClose(vActual, vRef, label: "V")
        Self.assertPaddingUnchanged(qBlock,
                                    original: q,
                                    rows: rows,
                                    rowStride: qStride,
                                    usedPerRow: qUsed)
        Self.assertPaddingUnchanged(kBlock,
                                    original: k,
                                    rows: rows,
                                    rowStride: kvStride,
                                    usedPerRow: kvUsed)
        Self.assertPaddingUnchanged(vBlock,
                                    original: v,
                                    rows: rows,
                                    rowStride: kvStride,
                                    usedPerRow: kvUsed)

        return (kActual, vActual)
    }

    private static func makeBlock(rows: Int,
                                  rowStride: Int,
                                  usedPerRow: Int,
                                  seed: UInt64,
                                  label: String) -> [Float16] {
        var rng = SeedTree(seed).key("prefill-qkv-\(label)")
        var values = [Float16](repeating: Float16(-9.0), count: rows * rowStride)
        for row in 0..<rows {
            for i in 0..<usedPerRow {
                values[row * rowStride + i] = Float16(rng.uniform(-1.0, 1.0))
            }
        }
        return values
    }

    private static func usedRows(_ values: [Float],
                                 rows: Int,
                                 rowStride: Int,
                                 usedPerRow: Int) -> [Float] {
        var out: [Float] = []
        out.reserveCapacity(rows * usedPerRow)
        for row in 0..<rows {
            out.append(contentsOf: values[(row * rowStride)..<(row * rowStride + usedPerRow)])
        }
        return out
    }

    private static func expectClose(_ actual: [Float],
                                    _ reference: [Float],
                                    label: String) {
        let rel = RelError.compute(actual: actual, reference: reference)
        #expect(rel < Tolerance.fp16Reduction,
                "\(label) rel=\(rel) maxAbs=\(RelError.maxAbsDiff(actual, reference))")
    }

    private static func assertPaddingUnchanged(_ buffer: any MTLBuffer,
                                               original: [Float16],
                                               rows: Int,
                                               rowStride: Int,
                                               usedPerRow: Int) {
        let ptr = unsafe buffer.contents().bindMemory(to: Float16.self, capacity: rows * rowStride)
        for row in 0..<rows {
            for i in usedPerRow..<rowStride {
                unsafe #expect(ptr[row * rowStride + i] == original[row * rowStride + i],
                        "padding changed at row=\(row) i=\(i)")
            }
        }
    }
}
