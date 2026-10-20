import Foundation
import Metal
import Testing
@testable import ANEFieldfareHotpath
import ANEFieldfareValidationSupport

@Suite struct PrefillSharedExpertTests {
    private static let rows = 4
    private static let d = 128
    private static let f = 64
    private static let xStride = 144
    private static let yStride = 137
    private static let sentinel = Float16(-7.25)

    @Test func blockSharedExpertMatchesRepeatedScalarRows() throws {
        try Self.runBlockSharedExpertMatchesRepeatedScalarRows()
    }

    private static func runBlockSharedExpertMatchesRepeatedScalarRows() throws {
        var rng = SeedTree(0xC0FFEE).key("prefill-shared-expert")
        let ctx = try MetalContext()
        let scalar = try SharedExpertInt8(context: ctx)
        let prefill = try PrefillSharedExpert(context: ctx)

        let x = makeInputBlock(rng: &rng)
        let gate = makeWeights(rows: f, cols: d, rng: &rng)
        let up = makeWeights(rows: f, cols: d, rng: &rng)
        let down = makeWeights(rows: d, cols: f, rng: &rng)

        guard let xBuf = Fp16Buffer.make(ctx.device, halves: x),
              let yRef = Fp16Buffer.make(ctx.device, halves: Array(repeating: sentinel, count: rows * yStride)),
              let yGot = Fp16Buffer.make(ctx.device, halves: Array(repeating: sentinel, count: rows * yStride)),
              let scratchAct = Fp16Buffer.make(ctx.device, count: f) else {
            Issue.record("buffer allocation failed")
            return
        }

        let gateProj = makeProjection(ctx: ctx, packed: gate, rows: f, cols: d)
        let upProj = makeProjection(ctx: ctx, packed: up, rows: f, cols: d)
        let downProj = makeProjection(ctx: ctx, packed: down, rows: d, cols: f)

        let halfBytes = MemoryLayout<Float16>.stride
        let refCB = ctx.queue.makeCommandBuffer()!
        for row in 0..<rows {
            try scalar.encode(commandBuffer: refCB,
                              x: xBuf,
                              xOffset: row * xStride * halfBytes,
                              gate: gateProj,
                              up: upProj,
                              down: downProj,
                              y: yRef,
                              yOffset: row * yStride * halfBytes,
                              scratchAct: scratchAct)
        }
        refCB.commit()
        refCB.waitUntilCompleted()
        #expect(refCB.error == nil)

        let gotCB = ctx.queue.makeCommandBuffer()!
        try prefill.encodeBlock(commandBuffer: gotCB,
                                x: xBuf,
                                y: yGot,
                                gate: gateProj,
                                up: upProj,
                                down: downProj,
                                scratchAct: scratchAct,
                                queryCount: rows,
                                d: d,
                                intermediate: f,
                                xStrideElements: xStride,
                                yStrideElements: yStride)
        gotCB.commit()
        gotCB.waitUntilCompleted()
        #expect(gotCB.error == nil)

        let ref = Fp16Buffer.readHalf(yRef, count: rows * yStride)
        let got = Fp16Buffer.readHalf(yGot, count: rows * yStride)
        // Both outputs start as sentinels, so this also covers the padding.
        #expect(got == ref)
    }

    private static func makeInputBlock(rng: inout SplitMix64) -> [Float16] {
        var block = Array(repeating: sentinel, count: rows * xStride)
        for row in 0..<rows {
            for col in 0..<d {
                block[row * xStride + col] = Float16(rng.uniform(-0.35, 0.35))
            }
        }
        return block
    }

    private static func makeWeights(rows: Int,
                                    cols: Int,
                                    rng: inout SplitMix64)
        -> (packed: [UInt8], scales: [UInt16], biases: [UInt16])
    {
        let groupsPerRow = cols / Quantization.groupSize
        var packed = [UInt8](repeating: 0, count: rows * cols)
        var scales = [UInt16](repeating: 0, count: rows * groupsPerRow)
        var biases = [UInt16](repeating: 0, count: rows * groupsPerRow)
        for row in 0..<rows {
            let values = (0..<cols).map { _ in rng.uniform(-0.4, 0.4) }
            let q = Quantization.quantizeInt8Affine(values)
            for col in 0..<cols {
                packed[row * cols + col] = q.packed[col]
            }
            for group in 0..<groupsPerRow {
                scales[row * groupsPerRow + group] = q.scales[group]
                biases[row * groupsPerRow + group] = q.biases[group]
            }
        }
        return (packed, scales, biases)
    }

    private static func makeProjection(
        ctx: MetalContext,
        packed: (packed: [UInt8], scales: [UInt16], biases: [UInt16]),
        rows: Int,
        cols: Int
    ) -> SharedExpertProjection {
        let w = unsafe ctx.device.makeBuffer(bytes: packed.packed,
                                      length: packed.packed.count,
                                      options: .storageModeShared)!
        let s = unsafe ctx.device.makeBuffer(bytes: packed.scales,
                                      length: packed.scales.count * MemoryLayout<UInt16>.stride,
                                      options: .storageModeShared)!
        let b = unsafe ctx.device.makeBuffer(bytes: packed.biases,
                                      length: packed.biases.count * MemoryLayout<UInt16>.stride,
                                      options: .storageModeShared)!
        return SharedExpertProjection(weights: w,
                                    scales: s,
                                    biases: b,
                                    rows: UInt32(rows),
                                    cols: UInt32(cols))
    }
}
