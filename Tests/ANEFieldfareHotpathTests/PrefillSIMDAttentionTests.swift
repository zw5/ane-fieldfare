import XCTest
import Metal
@testable import ANEFieldfareHotpath

final class PrefillSIMDAttentionTests: XCTestCase {
    func testCausalityAndSlidingRingAgainstIndependentOracle() throws {
        let context = try MetalContext()
        let kernel = try PrefillAttention(context: context)
        for d in [256, 512] {
            // A wrapped 16-slot ring under a 9-key window, and a wrapped
            // 2048-slot ring with a bidirectional image block.
            for (start, count, window, capacity, imageBlock) in [
                (40,7,9,16,false), (4092,7,1024,2048,true)
            ] {
                let heads = 4, kvHeads = 2, valid = start + count
                let slots = capacity == 0 ? valid : capacity
                let q = (0..<(count * heads * d)).map { Float16(sin(Float($0) * 0.133) * 0.35) }
                var k = [Float16](repeating: 0, count: slots * kvHeads * d)
                var v = k
                for key in max(0, valid-slots)..<valid {
                    for h in 0..<kvHeads { for c in 0..<d {
                        let i = ((key % slots) * kvHeads + h) * d + c
                        k[i] = Float16(cos(Float(key * 17 + h * 3 + c) * 0.031) * 0.25)
                        v[i] = Float16(sin(Float(key * 13 + h * 7 + c) * 0.071) * 2)
                    } }
                }
                func buffer(_ values: [Float16]) throws -> any MTLBuffer {
                    try values.withUnsafeBytes { unsafe try XCTUnwrap(context.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)) }
                }
                let qb = try buffer(q), kb = try buffer(k), vb = try buffer(v)
                let out = try buffer(Array(repeating: .nan, count: q.count))
                let reference = try buffer(Array(repeating: .nan, count: q.count))
                let params = PrefillAttentionParams(startPosition: UInt32(start), queryCount: UInt32(count), headDim: UInt32(d),
                    numQHeads: UInt32(heads), numKVHeads: UInt32(kvHeads), kvValidCount: UInt32(valid), slidingWindow: UInt32(window),
                    kvTokenStrideElements: UInt32(kvHeads*d), qTokenStrideElements: UInt32(heads*d), oTokenStrideElements: UInt32(heads*d), scale: 1,
                    bidirectionalBlockStart: UInt32(imageBlock ? start : 0),
                    bidirectionalBlockEnd: UInt32(imageBlock ? valid : 0))
                let cb = try XCTUnwrap(context.queue.makeCommandBuffer())
                for (output, path) in [(out, RuntimePrefillAttentionPath.causalSIMD), (reference, .causalTiled)] {
                    try kernel.encodeCausal(commandBuffer: cb, q: qb, k: kb, v: vb, out: output, params: params,
                        kvRingCapacity: UInt32(capacity), layerKind: .slidingWindow, path: path)
                }
                cb.commit(); cb.waitUntilCompleted(); XCTAssertNil(cb.error)
                unsafe XCTAssertEqual(Data(bytes: out.contents(), count: out.length),
                               Data(bytes: reference.contents(), count: reference.length),
                               "SIMD schedule must preserve the reference FP16 values")
                let actual = unsafe out.contents().assumingMemoryBound(to: Float16.self)
                var maximum = 0.0
                for row in 0..<count { for h in 0..<heads {
                    let first = window > 0 ? max(0, start + row + 1 - window) : 0
                    let keys = Array(first..<(imageBlock ? valid : start + row + 1))
                    let scores: [Double] = keys.map { key in
                        var score = 0.0
                        for c in 0..<d { score += Double(q[(row*heads+h)*d+c]) * Double(k[((key%slots)*kvHeads+h/2)*d+c]) }
                        return score
                    }
                    let mx = scores.max()!
                    let weights = scores.map { exp($0-mx) }, sum = weights.reduce(0,+)
                    for c in 0..<d {
                        var expected = 0.0
                        for (i,key) in keys.enumerated() { expected += weights[i] * Double(v[((key%slots)*kvHeads+h/2)*d+c]) / sum }
                        let value = unsafe Double(actual[(row*heads+h)*d+c])
                        XCTAssertTrue(value.isFinite)
                        maximum = max(maximum, abs(value-expected))
                    }
                } }
                XCTAssertLessThan(maximum, 0.0011, "D=\(d), ring=\(capacity)")
            }
        }
    }
}
