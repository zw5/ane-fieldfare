import Testing
import Foundation
import Metal
@testable import ANEFieldfareHotpath
import ANEFieldfareValidationSupport

/// `sample` kernel exercises. Inputs are pre-softmaxed FP16 probability
/// vectors (no upstream softmax needed) so the tests can hand-craft known
/// distributions and assert exact behaviour.
@Suite struct SampleTests {

    private static func makeProbs(_ values: [Float]) -> [Float16] {
        // Renormalize so the test inputs always sum to 1.0 exactly. The kernel
        // does not require a normalized input but the unit tests are clearer
        // when the inputs are actual probabilities.
        let sum = values.reduce(0, +)
        return values.map { Float16($0 / sum) }
    }

    private static func runSampler(probs: [Float16],
                                   temperature: Float,
                                   topK: UInt32,
                                   topP: Float,
                                   seed: UInt64) throws -> UInt32 {
        let ctx    = try MetalContext()
        let kernel = try Sample(context: ctx)

        let v = probs.count
        guard let inBuf  = unsafe ctx.device.makeBuffer(bytes: probs,
                                                 length: v * MemoryLayout<Float16>.size,
                                                 options: .storageModeShared),
              let outBuf = ctx.device.makeBuffer(length: MemoryLayout<UInt32>.size,
                                                 options: .storageModeShared),
              let cmd    = ctx.queue.makeCommandBuffer() else {
            Issue.record("Failed to allocate Metal resources")
            return UInt32.max
        }

        kernel.encode(commandBuffer: cmd,
                      probs: inBuf, outToken: outBuf,
                      v: UInt32(v),
                      temperature: temperature,
                      topK: topK, topP: topP, seed: seed)
        cmd.commit()
        cmd.waitUntilCompleted()

        return unsafe outBuf.contents().load(as: UInt32.self)
    }

    /// Temperature = 0 → argmax. We hand-place the max at a known index and
    /// expect that index back. Run at V=2048 to stress the cross-SIMD reduce.
    @Test func temperature_zero_picksArgmax() throws {
        let v = 2048
        var values = [Float](repeating: 0.01, count: v)
        values[1337] = 5.0   // clearly the argmax
        let probs = Self.makeProbs(values)

        let id = try Self.runSampler(probs: probs,
                                     temperature: 0.0,
                                     topK: 0, topP: 1.0,
                                     seed: 1)
        #expect(id == 1337, "got \(id)")
    }

    /// top_k = 4: the sampled token must be one of the top-4 indices, run
    /// across many trials. We hand-craft a distribution where the top-4 are
    /// well-separated from the rest.
    @Test func top_k_zerosOutMassOutsideTop() throws {
        let v = 1024
        var values = [Float](repeating: 0.0001, count: v)
        // Designated top-4 indices, with masses chosen so that even after
        // pow(p, 1/T) they dominate the long tail.
        let topIndices: [Int] = [10, 200, 500, 900]
        values[topIndices[0]] = 0.40
        values[topIndices[1]] = 0.30
        values[topIndices[2]] = 0.20
        values[topIndices[3]] = 0.10
        let probs = Self.makeProbs(values)
        let topSet = Set(topIndices.map { UInt32($0) })

        let trials = 32
        for t in 0..<trials {
            // Seed must be non-zero — xorshift64 has a fixed point at 0.
            let seed = (UInt64(t) &+ 1) &* 0x9E3779B97F4A7C15
            let id = try Self.runSampler(probs: probs,
                                         temperature: 1.0,
                                         topK: 4, topP: 1.0,
                                         seed: seed)
            #expect(topSet.contains(id),
                    "trial \(t): id=\(id) not in top-4 \(topIndices)")
        }
    }
}
