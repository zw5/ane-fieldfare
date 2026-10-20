import Testing
import Foundation
import Metal
@testable import ANEFieldfareHotpath
import ANEFieldfareValidationSupport

/// Compares the multi-row INT4 GEMV (`DequantInt4GEMV.encodeRows`) against
/// `DequantInt4GemvRef` applied row by row, and, below
/// `DequantInt4GEMV.matrixMinRows` rows, against the single-row GEMV, which the
/// ALU kernel must match bitwise; from that many rows the SIMD-group matrix
/// kernel matches the reference only. The single-row match is also the INT4
/// GEMV's CPU oracle. Shapes cover a partial 32-row tile, a K with a
/// non-multiple-of-four group count (the shared expert's 2112, whose last 64-k
/// group the matrix kernel takes as a half step), and the resident layout's
/// 2-aligned weight offset.
@Suite struct DequantInt4SmallBatchTests {
    private static func run(m: Int, n: Int, rows: Int, seed: UInt64, weightByteOffset: Int = 0,
                            matchSingleRow: Bool = false) throws {
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
        guard let wBuf = unsafe ctx.device.makeBuffer(bytes: packed, length: packed.count, options: .storageModeShared),
              let sBuf = unsafe ctx.device.makeBuffer(bytes: scales, length: scales.count * 2, options: .storageModeShared),
              let bBuf = unsafe ctx.device.makeBuffer(bytes: biases, length: biases.count * 2, options: .storageModeShared),
              let xBuf = Fp16Buffer.make(ctx.device, halves: x),
              let yBuf = Fp16Buffer.make(ctx.device, count: rows * m),
              let cmd = ctx.queue.makeCommandBuffer() else {
            Issue.record("allocation failed"); return
        }
        try kernel.encodeRows(commandBuffer: cmd, weights: wBuf, weightsOffset: weightByteOffset,
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
        guard matchSingleRow else { return }
        precondition(rows < DequantInt4GEMV.matrixMinRows, "the matrix kernel is not bitwise the GEMV")
        for r in 0..<rows {
            guard let single = Fp16Buffer.make(ctx.device, count: m),
                  let cb = ctx.queue.makeCommandBuffer() else { Issue.record("allocation failed"); return }
            kernel.encode(commandBuffer: cb, weights: wBuf, weightsOffset: weightByteOffset,
                          scales: sBuf, scalesOffset: 0, biases: bBuf, biasesOffset: 0,
                          x: xBuf, xOffset: r * n * 2, y: single, m: UInt32(m), n: UInt32(n))
            cb.commit()
            cb.waitUntilCompleted()
            let gemv = Fp16Buffer.read(single, count: m)
            let rowsResult = Array(actual[(r * m)..<((r + 1) * m)])
            #expect(gemv.map(\.bitPattern) == rowsResult.map(\.bitPattern), "row \(r) differs from the GEMV")
        }
    }

    /// The last run binds the weights at the resident layout's 2-aligned (not
    /// 4-aligned) offset, which the vectorized `ushort` load must honour.
    @Test func matchesSingleRowGEMVBitwise() throws {
        try Self.run(m: 300, n: 2112, rows: 3, seed: 0xD8, matchSingleRow: true)   // 2 weight rows per SIMD
        try Self.run(m: 301, n: 2816, rows: 4, seed: 0xD9, weightByteOffset: 2,
                     matchSingleRow: true)                                         // odd tail row
    }
    @Test func matrixKernelRowCounts() throws {
        try Self.run(m: 264, n: 8192, rows: 7, seed: 0xDD)                         // partial 32-row tile
        try Self.run(m: 256, n: 2112, rows: 12, seed: 0xDA)                        // final 64-k group
    }

    /// BF16 in and out, as the drafter runs it: the ALU kernel through 4
    /// rows, the matrix kernel past them.
    @Test(arguments: [3, 13])
    func bf16Rows(rows: Int) throws {
        let m = 384, n = 2816
        var rng = SeedTree(0xBF).key("int4-bf16-rows-\(rows)")
        let weightRows = (0..<m).map { _ in
            Quantization.quantizeInt4Affine((0..<n).map { _ in rng.uniform(-0.5, 0.5) })
        }
        let x = (0..<(rows * n)).map { _ in Quantization.bf16Bits(rng.uniform(-1.0, 1.0)) }
        let ctx = try MetalContext()
        let kernel = try DequantInt4GEMV(context: ctx)
        let device = ctx.device
        func buffer<T: BitwiseCopyable>(_ values: [T]) -> any MTLBuffer {
            values.withUnsafeBytes {
                unsafe device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
            }
        }
        let w = buffer(weightRows.flatMap(\.packed)), sc = buffer(weightRows.flatMap(\.scales))
        let bi = buffer(weightRows.flatMap(\.biases)), xb = buffer(x)
        let y = device.makeBuffer(length: rows * m * 2, options: .storageModeShared)!
        let cb = ctx.queue.makeCommandBuffer()!
        try kernel.encodeRows(commandBuffer: cb, weights: w, weightsOffset: 0, scales: sc, scalesOffset: 0,
                              biases: bi, biasesOffset: 0, x: xb, y: y, m: UInt32(m), n: UInt32(n), rows: rows,
                              xStrideElements: n, yStrideElements: m, types: .bf16)
        cb.commit(); cb.waitUntilCompleted()
        let out = unsafe y.contents().assumingMemoryBound(to: UInt16.self)
        for r in 0..<rows {
            let xr = x[(r * n)..<((r + 1) * n)].map(Quantization.bf16ToFloat)
            let ref = DequantInt4GemvRef.apply(weightRows: weightRows, x: xr, n: n)
            let got = (0..<m).map { unsafe Quantization.bf16ToFloat(out[r * m + $0]) }
            let rel = RelError.compute(actual: got, reference: ref)
            #expect(rel < 1e-2, "row \(r)/\(rows): rel=\(rel)")
        }
    }
}
