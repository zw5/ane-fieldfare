import XCTest
import Metal
@testable import ANEFieldfareHotpath

final class ANEQuantizedCarrierTests: XCTestCase {
    func testDistinctAffineWeightsPartialBatchAndRejectedOutput() throws {
        let context = try MetalContext()
        let backend = try ANERoutedPrefillBackend(context: context, layers: 1)
        let d = 2816, f = 704, rows = 3
        let w = [0, 1_115_136, 2_230_272], s = [991_232, 2_106_368, 3_221_504]
        let b = [1_053_184, 2_168_320, 3_283_456]
        let offsets = MoEExpertOffsets(gateWOff: UInt32(w[0]), gateSOff: UInt32(s[0]), gateBOff: UInt32(b[0]),
            upWOff: UInt32(w[1]), upSOff: UInt32(s[1]), upBOff: UInt32(b[1]),
            downWOff: UInt32(w[2]), downSOff: UInt32(s[2]), downBOff: UInt32(b[2]))
        var views: [TensorView] = []
        var referenceWeights: [[Double]] = []
        for slot in 0..<rows {
            let blob = try XCTUnwrap(context.device.makeBuffer(length: 3_358_720 + 128, options: .storageModeShared))
            memset(blob.contents(), 0, blob.length)
            let bytes = blob.contents().advanced(by: 64).assumingMemoryBound(to: UInt8.self)
            var expanded = [Double](repeating: 0, count: 3 * d * f)
            for m in 0..<3 {
                for g in 0..<(d * f / 64) {
                    let scale = Float(1 + (g + slot) % 3) / 512
                    let bias = scale * (-7 + Float((g + m) % 5) / 4)
                    let sh = UInt16(scale.bitPattern >> 16), bh = UInt16(bias.bitPattern >> 16)
                    bytes[s[m] + g * 2] = UInt8(truncatingIfNeeded: sh)
                    bytes[s[m] + g * 2 + 1] = UInt8(sh >> 8)
                    bytes[b[m] + g * 2] = UInt8(truncatingIfNeeded: bh)
                    bytes[b[m] + g * 2 + 1] = UInt8(bh >> 8)
                    for k in 0..<64 {
                        let i = g * 64 + k, code = (i / 17 + i * 7 + slot * 3 + m * 5) % 16
                        bytes[w[m] + i / 2] |= UInt8(code << ((i % 2) * 4))
                        expanded[m * d * f + i] = Double(Float16(Float(code) * scale + bias))
                    }
                }
            }
            referenceWeights.append(expanded)
            views.append(TensorView(buffer: blob, offset: 64, length: 3_358_720,
                scaleOffset: 0, scaleLength: 0, biasOffset: 0, biasLength: 0, shape: (0,0,0,0), dtype: 0))
        }
        let hidden = try XCTUnwrap(context.device.makeBuffer(length: rows * d * 2, options: .storageModeShared))
        let inputs = hidden.contents().assumingMemoryBound(to: Float16.self)
        for r in 0..<rows { for c in 0..<d { inputs[r * d + c] = Float16(Float((c * 13 + r * 17) % 31 - 15) / 50) } }
        let pairs = (0..<rows).map { PrefillTokenExpertPair(token: UInt32(rows - 1 - $0), expert: UInt32($0), rank: UInt32($0 + 2), weight: 1) }
        let pairBuffer = try pairs.withUnsafeBytes {
            try XCTUnwrap(context.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared))
        }
        let partials = try XCTUnwrap(context.device.makeBuffer(length: rows * 8 * d * 2, options: .storageModeShared))
        memset(partials.contents(), 0x7e, partials.length)
        let jobs = views.enumerated().map { ANERoutedPrefillBackend.Job(view: $0.element, start: $0.offset, count: 1, slots: []) }
        let badOutput = try XCTUnwrap(context.device.makeBuffer(length: 16, options: .storageModeShared))
        let rejected = try backend.prepareBatch(bank: 0, jobs: jobs, offsets: offsets, hidden: hidden, pairs: pairBuffer)
        XCTAssertThrowsError(try backend.prepareBatch(bank: 0, jobs: jobs, offsets: offsets, hidden: hidden, pairs: pairBuffer))
        XCTAssertThrowsError(try backend.submit(bank: 0, producer: rejected, partials: badOutput, pairs: pairBuffer))
        let producer = try backend.prepareBatch(bank: 0, jobs: jobs, offsets: offsets, hidden: hidden, pairs: pairBuffer)
        producer.waitUntilCompleted()
        let packed = backend.banks[0].input.contents().assumingMemoryBound(to: Float16.self)
        var packErrors = 0
        for slot in 0..<rows { for column in 0..<d { for middle in 0..<f {
            let lanes = backend.lanes
            let base = (slot * d + column) * (lanes + 3 * f)
            for (lane, source) in [(lanes, middle * d + column), (lanes + f, d * f + middle * d + column), (lanes + 2 * f, 2 * d * f + column * f + middle)] {
                if Double(packed[base + lane + middle]) != referenceWeights[slot][source] { packErrors += 1 }
            }
        } } }
        XCTAssertEqual(packErrors, 0, "Metal expansion must exactly match all BF16-affine CPU weights")
        let consumer = try backend.submit(bank: 0, producer: producer, partials: partials, pairs: pairBuffer)
        let unrelatedConsumer = try XCTUnwrap(context.queue.makeCommandBuffer())
        XCTAssertThrowsError(try backend.finish(bank: 0, consumer: unrelatedConsumer))
        XCTAssertThrowsError(try backend.submit(bank: 0, producer: producer, partials: partials, pairs: pairBuffer))
        consumer.commit(); try backend.finish(bank: 0, consumer: consumer)
        XCTAssertThrowsError(try backend.finish(bank: 0, consumer: consumer))
        var errorSquared = 0.0, referenceSquared = 0.0, maxError = 0.0
        let output = partials.contents().assumingMemoryBound(to: Float16.self)
        for slot in 0..<rows {
            let weights = referenceWeights[slot], token = Int(pairs[slot].token), rank = Int(pairs[slot].rank)
            var act = [Double](repeating: 0, count: f)
            for r in 0..<f {
                var gate = 0.0, up = 0.0
                for c in 0..<d {
                    let x = Double(inputs[token * d + c])
                    gate += weights[r * d + c] * x
                    up += weights[d * f + r * d + c] * x
                }
                act[r] = 0.5 * gate * (1 + tanh(0.7978845608028654 * (gate + 0.044715 * gate * gate * gate))) * up
            }
            for r in 0..<d {
                var expected = 0.0
                for c in 0..<f { expected += weights[2 * d * f + r * f + c] * act[c] }
                let actual = Double(output[(token * 8 + rank) * d + r])
                XCTAssertTrue(actual.isFinite)
                let error = actual - expected
                errorSquared += error * error; referenceSquared += expected * expected
                maxError = max(maxError, abs(error))
            }
        }
        // Full FP64 affine/GELU oracle, independently indexed from compact
        // checkpoint layout. ANE's FP16 intermediates are explicitly tolerated.
        XCTAssertLessThan(maxError, 0.0002)
        XCTAssertLessThan(sqrt(errorSquared / referenceSquared), 0.02)
        XCTAssertEqual(backend.evaluatedRows, 3)
        print("ANE affine oracle: 8448 outputs, max_abs=\(maxError), relative_l2=\(sqrt(errorSquared / referenceSquared))")
        inputs[0] = .nan
        let invalidProducer = try backend.prepareBatch(bank: 1, jobs: jobs, offsets: offsets, hidden: hidden, pairs: pairBuffer)
        let invalidConsumer = try backend.submit(bank: 1, producer: invalidProducer, partials: partials, pairs: pairBuffer)
        invalidConsumer.commit()
        XCTAssertThrowsError(try backend.finish(bank: 1, consumer: invalidConsumer))
        XCTAssertEqual(backend.evaluatedRows, 3, "rejected input must not count as evaluated rows")
        XCTAssertThrowsError(try backend.prepareBatch(bank: 1, jobs: jobs, offsets: offsets, hidden: hidden, pairs: pairBuffer))
    }
}
