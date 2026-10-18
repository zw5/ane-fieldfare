import Metal
import Testing
@testable import ANEFieldfareHotpath

private let visionMPPTensorOpsAvailable = MTLCreateSystemDefaultDevice()?
    .supportsFamily(.apple8) == true

@Suite struct VisionAttentionTests {
    /// The layout is the caller's decision, signalled by the row stride:
    /// stride 0 is row-major Q/K/V on the native kernel, a positive stride is
    /// the padded head-major store on the MPP kernel. Reading row-major input
    /// as 80-stride head-major would be silent garbage.
    @Test(.enabled(if: visionMPPTensorOpsAvailable,
                   "vision MPP tensor shapes require Apple8 or newer"))
    func rowMajorInputsAreHonouredBesideTheFusedLayout() throws {
        let context = try MetalContext()
        let attention = try VisionAttention(context: context)

        // 200 spans four 64-key tiles; the reference test validates both paths
        // against FP32 at this length, so any divergence here is the layout.
        let length = 200
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
        func make(_ values: [Float]) throws -> MTLBuffer {
            let bits = values.map(Quantization.bf16Bits)
            return try #require(context.device.makeBuffer(
                bytes: bits, length: bits.count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared))
        }
        // The same values laid out head-major with the 80-lane stride, which is
        // what the fused branch expects.
        let padded = VisionAttention.paddedHeadDimension
        func makePadded(_ values: [Float]) throws -> MTLBuffer {
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

        func run(rowMajor: Bool) throws -> [Float] {
            let output = try #require(context.device.makeBuffer(
                length: q.count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared))
            let commandBuffer = try #require(context.queue.makeCommandBuffer())
            attention.encode(
                commandBuffer: commandBuffer,
                q: rowMajor ? try make(q) : try makePadded(q),
                k: rowMajor ? try make(k) : try makePadded(k),
                v: rowMajor ? try make(v) : try makePadded(v),
                output: output,
                sequenceLength: length, numHeads: heads,
                inputRowStride: rowMajor ? 0 : length)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            #expect(commandBuffer.error == nil)
            let bits = output.contents().bindMemory(to: UInt16.self, capacity: q.count)
            return (0..<q.count).map { Quantization.bf16ToFloat(bits[$0]) }
        }

        // Same attention, same values, two layouts. A layout misread is a gross
        // error, so this compares the two paths against each other rather than
        // against a tolerance on an FP32 reference.
        let fallback = try run(rowMajor: true)
        let fused = try run(rowMajor: false)
        // The tolerance below is wider than the signal in this fixture, so it
        // cannot tell a correct output from an empty one. It let a fallback
        // that wrote only half its rows pass. Every row must be written.
        for row in 0..<length {
            let start = row * heads * dimension
            let slice = fallback[start..<(start + heads * dimension)]
            #expect(slice.contains { $0 != 0 },
                    "row \(row) of the fallback output was never written")
        }
        for index in fallback.indices {
            let detail = "row-major fallback diverged at \(index): "
                + "\(fallback[index]) vs fused \(fused[index])"
            #expect(abs(fallback[index] - fused[index]) <= 0.02, "\(detail)")
        }
    }

    /// 5 fits one 64-key tile; 200 spans four tiles and is the only case that
    /// exercises the online rescale between tiles.
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
        func make(_ values: [Float]) throws -> MTLBuffer {
            let bits = values.map(Quantization.bf16Bits)
            return try #require(context.device.makeBuffer(
                bytes: bits, length: bits.count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared))
        }
        let qBuffer = try make(q)
        let kBuffer = try make(k)
        let vBuffer = try make(v)
        // Head-major [head][row][80] copies for the fused layout, which reads the
        // padding the projections would otherwise have written.
        let padded = VisionAttention.paddedHeadDimension
        func makePadded(_ values: [Float]) throws -> MTLBuffer {
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
        for fused in [false, true] {
            let commandBuffer = try #require(context.queue.makeCommandBuffer())
            attention.encode(commandBuffer: commandBuffer,
                             q: fused ? paddedQBuffer : qBuffer,
                             k: fused ? paddedKBuffer : kBuffer,
                             v: fused ? paddedVBuffer : vBuffer,
                             output: output,
                             sequenceLength: length, numHeads: heads,
                             inputRowStride: fused ? length : 0)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            #expect(commandBuffer.error == nil)

            let actualBits = output.contents().bindMemory(
                to: UInt16.self, capacity: q.count)
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
                        let actual = Quantization.bf16ToFloat(actualBits[index])
                        #expect(abs(actual - expected) <= 0.02,
                                "fused=\(fused) index=\(index) actual=\(actual) expected=\(expected)")
                    }
                }
            }
        }
    }
}
