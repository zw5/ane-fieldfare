import Foundation
import Metal
import Testing
@testable import ANEFieldfareHotpath

private let visionMPPTensorOpsAvailable = MTLCreateSystemDefaultDevice()?
    .supportsFamily(.apple8) == true

@Suite struct VisionAttentionTests {
    /// The MPP kernel over the padded head-major store against the FP32
    /// reference. 5 fits one 64-key tile; 200 spans four tiles and is the only
    /// case that exercises the online rescale between tiles.
    @Test(.enabled(if: visionMPPTensorOpsAvailable,
                   "vision MPP tensor shapes require Apple8 or newer"),
          arguments: [5, 200])
    func onlineAttentionMatchesFP32Reference(length: Int) throws {
        let context = try MetalContext()
        let heads = 2
        let dimension = VisionAttention.headDimension
        var q = [Float](repeating: 0, count: length * heads * dimension)
        var k = q
        var v = q
        for index in q.indices {
            q[index] = Float((index * 13) % 17 - 8) / 32
            k[index] = Float((index * 7) % 19 - 9) / 32
            v[index] = Float((index * 5) % 23 - 11) / 16
        }
        func make(_ values: [Float]) throws -> any MTLBuffer {
            let bits = values.map(Quantization.bf16Bits)
            return try #require(unsafe context.device.makeBuffer(
                bytes: bits, length: bits.count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared))
        }
        // Head-major [head][row][80] copies for the fused layout, which reads the
        // padding the projections would otherwise have written.
        let padded = VisionAttention.paddedHeadDimension
        func makePadded(_ values: [Float]) throws -> any MTLBuffer {
            var laid = [Float](repeating: 0, count: heads * length * padded)
            for row in 0..<length {
                for head in 0..<heads {
                    for d in 0..<dimension {
                        laid[(head * length + row) * padded + d] =
                            values[(row * heads + head) * dimension + d]
                    }
                }
            }
            return try make(laid)
        }
        let paddedQBuffer = try makePadded(q)
        let paddedKBuffer = try makePadded(k)
        let paddedVBuffer = try makePadded(v)
        let output = try #require(context.device.makeBuffer(
            length: q.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared))

        let attention = try VisionAttention(context: context)
        unsafe memset(output.contents(), 0, output.length)
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        try attention.encode(commandBuffer: commandBuffer,
                             q: paddedQBuffer, k: paddedKBuffer, v: paddedVBuffer,
                             output: output,
                             sequenceLength: length, numHeads: heads,
                             inputRowStride: length)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        let actualBits = unsafe output.contents().bindMemory(
            to: UInt16.self, capacity: q.count)
        // The tolerance below is wider than this fixture's signal, so it
        // cannot tell an unwritten row from a correct one: it once let a
        // fallback that wrote only half its rows pass.
        for query in 0..<length {
            let start = query * heads * dimension
            let written = (start..<(start + heads * dimension)).contains { unsafe actualBits[$0] != 0 }
            #expect(written, "row \(query) was never written")
        }
        for query in 0..<length {
            for head in 0..<heads {
                var scores = [Float](repeating: 0, count: length)
                for key in 0..<length {
                    for d in 0..<dimension {
                        scores[key] += q[(query * heads + head) * dimension + d]
                            * k[(key * heads + head) * dimension + d]
                    }
                }
                let maximum = scores.max()!
                let weights = scores.map { exp($0 - maximum) }
                let denominator = weights.reduce(0, +)
                for d in 0..<dimension {
                    var expected: Float = 0
                    for key in 0..<length {
                        expected += weights[key]
                            * v[(key * heads + head) * dimension + d]
                    }
                    expected /= denominator
                    let index = (query * heads + head) * dimension + d
                    let actual = unsafe Quantization.bf16ToFloat(actualBits[index])
                    #expect(abs(actual - expected) <= 0.02,
                            "index=\(index) actual=\(actual) expected=\(expected)")
                }
            }
        }
    }
}
