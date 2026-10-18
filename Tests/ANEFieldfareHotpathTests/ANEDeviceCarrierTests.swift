import XCTest
import Metal
@testable import ANEFieldfareHotpath

final class ANEDeviceCarrierTests: XCTestCase {
    func testRealANEDeviceWeightsAndABAReuse() throws {
        let context = try MetalContext()
        let backend = try ANERoutedPrefillBackend(context: context, layers: 1)
        let d = 2816, f = 704, rows = 7
        let blob = try XCTUnwrap(context.device.makeBuffer(length: 3_358_720, options: .storageModeShared))
        memset(blob.contents(), 0, blob.length)
        let w = [0, 1_115_136, 2_230_272], s = [991_232, 2_106_368, 3_221_504]
        let b = [1_053_184, 2_168_320, 3_283_456]
        let bytes = blob.contents().assumingMemoryBound(to: UInt8.self)
        for m in 0..<3 {
            for g in 0..<(d * f / 64) { bytes[s[m] + g * 2] = 0x80; bytes[s[m] + g * 2 + 1] = 0x3f }
            let cols = m == 2 ? f : d
            for r in 0..<f { let c = m == 0 ? (r * 3 + 1) % d : m == 1 ? (r * 5 + 2) % d : (r + 13) % f; let i = r * cols + c; bytes[w[m] + i / 2] |= i % 2 == 0 ? 1 : 16 }
        }
        let view = TensorView(buffer: blob, offset: 0, length: UInt64(blob.length), scaleOffset: 0, scaleLength: 0,
                              biasOffset: 0, biasLength: 0, shape: (0,0,0,0), dtype: 0)
        let offsets = MoEExpertOffsets(gateWOff: UInt32(w[0]), gateSOff: UInt32(s[0]), gateBOff: UInt32(b[0]),
            upWOff: UInt32(w[1]), upSOff: UInt32(s[1]), upBOff: UInt32(b[1]),
            downWOff: UInt32(w[2]), downSOff: UInt32(s[2]), downBOff: UInt32(b[2]))
        let hidden = try XCTUnwrap(context.device.makeBuffer(length: rows * d * 2, options: .storageModeShared))
        let partials = try XCTUnwrap(context.device.makeBuffer(length: rows * 8 * d * 2, options: .storageModeShared))
        let pairs = (0..<rows).map { PrefillTokenExpertPair(token: UInt32(rows - 1 - $0), expert: 0,
                                                          rank: UInt32($0 % 8), weight: 1) }
        let pairBuffer = try pairs.withUnsafeBytes {
            try XCTUnwrap(context.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared))
        }
        var first: Data?
        for (run, amplitude) in [Float(0.1), Float(-0.2), Float(0.1)].enumerated() {
            let inputs = hidden.contents().assumingMemoryBound(to: Float16.self)
            for r in 0..<rows { for c in 0..<d { inputs[r * d + c] = Float16(amplitude + Float(r) * 0.01 + Float(c % 7) * 0.001) } }
            memset(partials.contents(), 0, partials.length)
            let bank = run % 2
            let jobs = [(0,2),(2,3),(5,1),(6,1)].map {
                ANERoutedPrefillBackend.Job(view: view, start: $0.0, count: $0.1, slots: [])
            }
            let producer = try backend.prepareBatch(bank: bank, jobs: jobs, offsets: offsets,
                                                    hidden: hidden, pairs: pairBuffer)
            let consumer = try backend.submit(bank: bank, producer: producer, partials: partials,
                                               pairs: pairBuffer)
            consumer.commit(); try backend.finish(bank: bank, consumer: consumer)
            let out = partials.contents().assumingMemoryBound(to: Float16.self)
            var maximum = 0.0
            for pair in pairs { for c in 0..<d {
                let middle = (c + 13) % f
                let x = Double(inputs[Int(pair.token) * d + (middle * 3 + 1) % d])
                let up = Double(inputs[Int(pair.token) * d + (middle * 5 + 2) % d])
                let expected = c < f ? up * (0.5 * x * (1 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))) : 0
                let actual = Double(out[(Int(pair.token) * 8 + Int(pair.rank)) * d + c])
                XCTAssertTrue(actual.isFinite)
                maximum = max(maximum, abs(actual - expected))
            } }
            XCTAssertLessThan(maximum, 0.0003)
            let data = Data(bytes: partials.contents(), count: partials.length)
            if run == 0 { first = data }; if run == 2 { XCTAssertEqual(first, data) }
        }
        XCTAssertEqual(backend.evaluatedRows, 21)
    }
}
