import Accelerate
import Metal
import Testing
@testable import ANEFieldfareHotpath
import ANEFieldfareValidationSupport

@Suite struct LMHeadChainInt4Tests {
    private static let rmsEps: Float = 1e-6

    private static func cpuGreedy(hiddenFp16: [Float16],
                                  normWeightBF16: [UInt16],
                                  rows: [Quantization.Int4AffineRow],
                                  d: Int,
                                  v: Int) -> UInt32 {
        let hidden = hiddenFp16.map(Float.init)
        let normWeight = normWeightBF16.map(Quantization.bf16ToFloat)
        let normed = RmsNormRef.apply(x: hidden, weight: normWeight, eps: rmsEps)
            .map { Float(Float16($0)) }

        var bestValue: Float = -.infinity
        var bestIndex: UInt32 = 0
        for row in 0..<v {
            let dequantized = Quantization.dequantizeInt4Affine(rows[row], n: d)
            var dot: Float = 0
            dequantized.withUnsafeBufferPointer { weights in
                normed.withUnsafeBufferPointer { input in
                    unsafe vDSP_dotpr(weights.baseAddress!, 1,
                               input.baseAddress!, 1,
                               &dot,
                               vDSP_Length(d))
                }
            }
            if dot > bestValue {
                bestValue = dot
                bestIndex = UInt32(row)
            }
        }
        return bestIndex
    }

    private static func packRows(_ rows: [Quantization.Int4AffineRow])
        -> (packed: [UInt8], scales: [UInt16], biases: [UInt16]) {
        let packedRowCount = rows[0].packed.count
        let groupCount = rows[0].scales.count
        var packed = [UInt8](repeating: 0, count: rows.count * packedRowCount)
        var scales = [UInt16](repeating: 0, count: rows.count * groupCount)
        var biases = [UInt16](repeating: 0, count: rows.count * groupCount)
        for row in rows.indices {
            for index in 0..<packedRowCount {
                packed[row * packedRowCount + index] = rows[row].packed[index]
            }
            for index in 0..<groupCount {
                scales[row * groupCount + index] = rows[row].scales[index]
                biases[row * groupCount + index] = rows[row].biases[index]
            }
        }
        return (packed, scales, biases)
    }

    private static func gpuGreedy(hiddenFp16: [Float16],
                                  normBF16: [UInt16],
                                  rows: [Quantization.Int4AffineRow],
                                  d: Int,
                                  v: Int,
                                  hiddenOffset: Int = 0) throws -> UInt32 {
        let (packed, scales, biases) = Self.packRows(rows)
        let context = try MetalContext()
        let chain = try LMHeadChainInt4(context: context, maxD: d, maxVocab: v)

        guard let hidden = unsafe context.device.makeBuffer(
                  bytes: hiddenFp16,
                  length: hiddenFp16.count * MemoryLayout<Float16>.stride,
                  options: .storageModeShared),
              let norm = unsafe context.device.makeBuffer(
                  bytes: normBF16,
                  length: normBF16.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared),
              let weights = unsafe context.device.makeBuffer(
                  bytes: packed,
                  length: packed.count,
                  options: .storageModeShared),
              let scaleBuffer = unsafe context.device.makeBuffer(
                  bytes: scales,
                  length: scales.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared),
              let biasBuffer = unsafe context.device.makeBuffer(
                  bytes: biases,
                  length: biases.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared),
              let output = context.device.makeBuffer(
                  length: MemoryLayout<UInt32>.stride,
                  options: .storageModeShared),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed")
            return 0
        }

        chain.encodeGreedyDecode(commandBuffer: commandBuffer,
                                 hidden: hidden,
                                 hiddenOffset: hiddenOffset,
                                 normWeight: norm,
                                 weights: weights,
                                 scales: scaleBuffer,
                                 biases: biasBuffer,
                                 outToken: output,
                                 d: UInt32(d),
                                 vocab: UInt32(v))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return unsafe output.contents().load(as: UInt32.self)
    }

    @Test func rawGreedyMatchesCPUReference() throws {
        let d = 64
        let v = 1024
        var rng = SplitMix64(seed: 0x1A4E_4EAD_2026_0603)
        let hidden = (0..<d).map { _ in Float16(rng.uniform(-1, 1)) }
        let norm = (0..<d).map { _ in Quantization.bf16Bits(rng.uniform(0.5, 1.5)) }
        let rows = (0..<v).map { _ in
            Quantization.quantizeInt4Affine((0..<d).map { _ in rng.uniform(-1, 1) })
        }

        let reference = Self.cpuGreedy(hiddenFp16: hidden,
                                       normWeightBF16: norm,
                                       rows: rows,
                                       d: d,
                                       v: v)
        let result = try Self.gpuGreedy(hiddenFp16: hidden,
                                        normBF16: norm,
                                        rows: rows,
                                        d: d,
                                        v: v)
        #expect(result == reference)
    }

    @Test func allInvalidCandidatesFallBackToZero() throws {
        let d = 64
        let v = 1024
        let hidden = [Float16](repeating: .nan, count: d)
        let norm = (0..<d).map { _ in Quantization.bf16Bits(1) }
        let rows = (0..<v).map { _ in
            Quantization.quantizeInt4Affine([Float](repeating: 0.25, count: d))
        }

        let result = try Self.gpuGreedy(hiddenFp16: hidden,
                                        normBF16: norm,
                                        rows: rows,
                                        d: d,
                                        v: v)
        #expect(result == 0)
    }

    @Test(arguments: [1024, 2048, 4096])
    func greedyMatchesCPUReferenceAcrossVocabSizes(v: Int) throws {
        let d = 64
        let target = v - 3
        let hidden = (0..<d).map { Float16(0.25 + Float($0 % 7) * 0.01) }
        let norm = (0..<d).map { _ in Quantization.bf16Bits(1) }
        let rows = (0..<v).map { row in
            let value: Float = row == target ? 1 : -0.5
            return Quantization.quantizeInt4Affine([Float](repeating: value, count: d))
        }

        let reference = Self.cpuGreedy(hiddenFp16: hidden,
                                       normWeightBF16: norm,
                                       rows: rows,
                                       d: d,
                                       v: v)
        let result = try Self.gpuGreedy(hiddenFp16: hidden,
                                        normBF16: norm,
                                        rows: rows,
                                        d: d,
                                        v: v)
        #expect(result == reference)
    }

    @Test func tiesResolveToLowestIndex() throws {
        let d = 64
        let v = 1024
        let hidden = (0..<d).map { Float16(0.2 + Float($0 % 5) * 0.02) }
        let norm = (0..<d).map { _ in Quantization.bf16Bits(1) }
        let rows = (0..<v).map { row in
            let value: Float = (row == 17 || row == 311) ? 1 : -0.25
            return Quantization.quantizeInt4Affine([Float](repeating: value, count: d))
        }

        let result = try Self.gpuGreedy(hiddenFp16: hidden,
                                        normBF16: norm,
                                        rows: rows,
                                        d: d,
                                        v: v)
        #expect(result == 17)
    }

    @Test func readsSelectedHiddenRow() throws {
        let d = 64
        let v = 1024
        let rowStride = d + 16
        let selectedRow = 3
        let target = 777
        var hiddenRows = [Float16](repeating: 0, count: 5 * rowStride)
        for row in 0..<5 {
            for index in 0..<d {
                let value = row == selectedRow
                    ? Float(0.25 + Float(index % 7) * 0.01)
                    : Float(-0.4 - Float(index % 5) * 0.01)
                hiddenRows[row * rowStride + index] = Float16(value)
            }
        }
        let norm = (0..<d).map { _ in Quantization.bf16Bits(1) }
        let rows = (0..<v).map { row in
            let value: Float = row == target ? 1 : -0.25
            return Quantization.quantizeInt4Affine([Float](repeating: value, count: d))
        }

        let result = try Self.gpuGreedy(
            hiddenFp16: hiddenRows,
            normBF16: norm,
            rows: rows,
            d: d,
            v: v,
            hiddenOffset: selectedRow * rowStride * MemoryLayout<Float16>.stride)
        #expect(result == UInt32(target))
    }

    /// `encodeGreedyCheck` over 1-4 saved rows gives each row the token the
    /// single-row head gives it, ties to the lowest index included; and the
    /// hot-vocabulary guess is the single-row head's argmax over its listed
    /// rows. Decode's shape (D = 2816), a 4,099-row head with a tied pair.
    @Test func checkAndHotGuessMatchSingleRowHead() throws {
        let d = 2816, v = 4099, rowsChecked = 4
        var rng = SplitMix64(seed: 0xC4EC_4EAD)
        var rows = (0..<v).map { _ in
            Quantization.quantizeInt4Affine((0..<d).map { _ in rng.uniform(-0.5, 0.5) })
        }
        rows[3001] = rows[1200]
        let (packed, scales, biases) = Self.packRows(rows)
        let hidden = (0..<(rowsChecked * d)).map { _ in Float16(rng.uniform(-1, 1)) }
        let norm = (0..<d).map { _ in Quantization.bf16Bits(rng.uniform(0.5, 1.5)) }
        let context = try MetalContext()
        let device = context.device
        let chain = try LMHeadChainInt4(context: context, maxD: d, maxVocab: v)
        func buffer<T: BitwiseCopyable>(_ values: [T]) -> any MTLBuffer {
            values.withUnsafeBytes {
                unsafe device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
            }
        }
        let hiddenBuffer = buffer(hidden), normBuffer = buffer(norm)
        let w = buffer(packed), s = buffer(scales), b = buffer(biases)
        let normed = device.makeBuffer(length: rowsChecked * d * 2, options: .storageModeShared)!
        let out = device.makeBuffer(length: 16, options: .storageModeShared)!

        var single: [UInt32] = []
        for row in 0..<rowsChecked {
            let cb = context.queue.makeCommandBuffer()!
            chain.encodeGreedyDecode(commandBuffer: cb, hidden: hiddenBuffer, hiddenOffset: row * d * 2,
                                     normWeight: normBuffer, weights: w, scales: s, biases: b,
                                     outToken: out, d: UInt32(d), vocab: UInt32(v))
            cb.commit(); cb.waitUntilCompleted()
            single.append(unsafe out.contents().load(as: UInt32.self))
        }
        for count in 1...rowsChecked {
            let cb = context.queue.makeCommandBuffer()!
            for row in 0..<count {
                chain.encodeNormRow(commandBuffer: cb, hidden: hiddenBuffer, hiddenOffset: row * d * 2,
                                    normWeight: normBuffer, normedRows: normed, row: row, d: UInt32(d))
            }
            chain.encodeGreedyCheck(commandBuffer: cb, normedRows: normed, rowCount: count,
                                    weights: w, scales: s, biases: b, outTokens: out,
                                    d: UInt32(d), vocab: UInt32(v))
            cb.commit(); cb.waitUntilCompleted()
            let checked = (0..<count).map { unsafe out.contents().load(fromByteOffset: 4 * $0, as: UInt32.self) }
            #expect(checked == Array(single.prefix(count)), "\(count) rows")
        }

        // A list holding each row's winner and every other row but a few.
        let listed = (0..<UInt32(v)).filter { $0 % 7 != 3 || single.contains($0) }
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("hot-\(UUID()).bin")
        var data = Data(HotVocabulary.magic)
        for value in [UInt32(v), UInt32(listed.count)] + listed.reversed() {
            withUnsafeBytes(of: value.littleEndian) { unsafe data.append(contentsOf: $0) }
        }
        try data.write(to: file)
        defer { try? FileManager.default.removeItem(at: file) }
        let table = try HotVocabulary(url: file, vocab: v, device: device)
        for row in 0..<rowsChecked {
            let cb = context.queue.makeCommandBuffer()!
            chain.encodeHotGreedyDecode(commandBuffer: cb, hidden: hiddenBuffer, hiddenOffset: row * d * 2,
                                        normWeight: normBuffer, table: table, weights: w, scales: s, biases: b,
                                        normedRows: normed, row: row, outToken: out, d: UInt32(d))
            cb.commit(); cb.waitUntilCompleted()
            #expect(unsafe out.contents().load(as: UInt32.self) == single[row], "hot guess row \(row)")
        }
    }
}
