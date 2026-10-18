import Testing
import Foundation
import Metal
@testable import ANEFieldfareHotpath
import ANEFieldfareValidationSupport

/// Compares the small-batch INT4 GEMM (`DequantInt4GEMV.encodeRows`, SIMD-group
/// matrix tiles over FP16-dequantized weights) against `DequantInt4GemvRef`
/// applied row by row. Shapes cover a partial 32-column tile, a K with a
/// non-multiple-of-four group count (the shared expert's 2112), one and two
/// 8-row tiles, and the resident layout's 2-aligned weight offset.
@Suite struct DequantInt4SmallBatchTests {
    private static func run(m: Int, n: Int, rows: Int, seed: UInt64, weightByteOffset: Int = 0) throws {
        var rng = SeedTree(seed).key("int4-small-batch-m\(m)-n\(n)-r\(rows)-off\(weightByteOffset)")
        let weightRows = (0..<m).map { _ in
            Quantization.quantizeInt4Affine((0..<n).map { _ in rng.uniform(-0.5, 0.5) })
        }
        var packed = [UInt8](repeating: 0, count: weightByteOffset)
        var scales: [UInt16] = [], biases: [UInt16] = []
        for row in weightRows {
            packed += row.packed
            scales += row.scales
            biases += row.biases
        }
        let x = (0..<(rows * n)).map { _ in Float16(rng.uniform(-1.0, 1.0)) }

        let ctx = try MetalContext()
        let kernel = try DequantInt4GEMV(context: ctx)
        guard let wBuf = ctx.device.makeBuffer(bytes: packed, length: packed.count, options: .storageModeShared),
              let sBuf = ctx.device.makeBuffer(bytes: scales, length: scales.count * 2, options: .storageModeShared),
              let bBuf = ctx.device.makeBuffer(bytes: biases, length: biases.count * 2, options: .storageModeShared),
              let xBuf = Fp16Buffer.make(ctx.device, halves: x),
              let yBuf = Fp16Buffer.make(ctx.device, count: rows * m),
              let cmd = ctx.queue.makeCommandBuffer() else {
            Issue.record("allocation failed"); return
        }
        kernel.encodeRows(commandBuffer: cmd, weights: wBuf, weightsOffset: weightByteOffset,
                          scales: sBuf, scalesOffset: 0, biases: bBuf, biasesOffset: 0,
                          x: xBuf, y: yBuf, m: UInt32(m), n: UInt32(n), rows: rows,
                          xStrideElements: n, yStrideElements: m)
        cmd.commit()
        cmd.waitUntilCompleted()
        let actual = Fp16Buffer.read(yBuf, count: rows * m)
        for r in 0..<rows {
            let xr = x[(r * n)..<((r + 1) * n)].map { Float($0) }
            let ref = DequantInt4GemvRef.apply(weightRows: weightRows, x: Array(xr), n: n)
            let got = Array(actual[(r * m)..<((r + 1) * m)])
            let rel = RelError.compute(actual: got, reference: ref)
            #expect(rel < Tolerance.fp16Reduction, "M=\(m) N=\(n) row \(r)/\(rows): rel=\(rel)")
        }
    }

    @Test func oneRow() throws { try Self.run(m: 1024, n: 2816, rows: 1, seed: 0xD1) }
    @Test func eightRows() throws { try Self.run(m: 1024, n: 2816, rows: 8, seed: 0xD2) }
    @Test func sixteenRows() throws { try Self.run(m: 1024, n: 2816, rows: 16, seed: 0xD3) }
    @Test func partialTileAndOddRows() throws { try Self.run(m: 100, n: 256, rows: 11, seed: 0xD4) }
    @Test func sharedExpertDownShape() throws { try Self.run(m: 2816, n: 2112, rows: 9, seed: 0xD5) }
    @Test func twoAlignedWeightOffset() throws {
        try Self.run(m: 256, n: 2816, rows: 8, seed: 0xD6, weightByteOffset: 2)
    }
}
