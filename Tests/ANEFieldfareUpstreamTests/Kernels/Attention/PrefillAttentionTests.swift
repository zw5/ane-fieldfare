import Testing
import Foundation
import Metal
@testable import ANEFieldfareHotpath
import ANEFieldfareValidationSupport

@Suite struct PrefillAttentionTests {
    private struct Fixture {
        var q: [Float]
        var k: [Float]
        var v: [Float]
        var qStride: Int
        var kvStride: Int
        var oStride: Int
        var headDim: Int
        var qHeads: Int
        var kvHeads: Int
        var start: Int
        var chunk: Int
        var kvValid: Int
        var window: Int
        var scale: Float
        var bidirectionalBlock: Range<Int>?
    }

    @Test func prefillAttentionMatchesCPUReferenceFullAndSWA() throws {
        let cases: [(label: String, start: Int, chunk: Int, window: Int, imageBlock: Range<Int>?)] = [
            ("full-offset", 4, 3, 0, nil),
            ("swa-truncated", 7, 4, 5, nil),
            // Sliding layers see an image block bidirectionally within the
            // window; full layers ignore the block and stay causal.
            ("swa-image-block", 4, 6, 5, 5..<9),
            ("full-image-block-stays-causal", 4, 6, 0, 5..<9),
        ]

        for (index, c) in cases.enumerated() {
            var fixture = Self.makeFixture(start: c.start,
                                           chunk: c.chunk,
                                           window: c.window,
                                           seed: 0xA510 + UInt64(index))
            fixture.bidirectionalBlock = c.imageBlock
            try Self.runAndCompare(fixture, label: c.label)
        }
    }

    @Test(arguments: [1, 65, 129, 1_025])
    func tensorOps2DFullAttentionMatchesReferenceAtTileBoundaries(_ visibleKeys: Int) throws {
        let context = try MetalContext()
        // GPU family is not the capability test: this pipeline compiles and
        // dispatches on the project's Apple8 M2 minimum.
        guard (try? context.pipeline("attention_prefill_full_tensorops_2d_validity_v2")) != nil else { return }
        let fixture = Self.makeFixture(start: visibleKeys - 1,
                                       chunk: 1,
                                       window: 0,
                                       seed: 0xA870 + UInt64(visibleKeys),
                                       headDim: 512,
                                       qHeads: 16,
                                       kvHeads: 2)
        let candidate = try Self.runKernel(
            fixture,
            path: .fullTensorOps2DValidityV2)
        let repeated = try Self.runKernel(
            fixture,
            path: .fullTensorOps2DValidityV2)
        let reference = Self.reference(fixture)
        let maxAbs = RelError.maxAbsDiff(candidate, reference)
        let rel = RelError.compute(actual: candidate, reference: reference)
        #expect(candidate == repeated,
                "TensorOps 2D full attention is not byte-stable at \(visibleKeys) keys")
        #expect(maxAbs <= 2e-2,
                "TensorOps 2D maxAbs=\(maxAbs) rel=\(rel) keys=\(visibleKeys)")
        #expect(rel <= 2e-2,
                "TensorOps 2D rel=\(rel) maxAbs=\(maxAbs) keys=\(visibleKeys)")
    }

    private static func makeFixture(start: Int,
                                    chunk: Int,
                                    window: Int,
                                    seed: UInt64,
                                    headDim: Int = 8,
                                    qHeads: Int = 4,
                                    kvHeads: Int = 2) -> Fixture {
        let qStride = qHeads * headDim + 3
        let kvStride = kvHeads * headDim + 5
        let oStride = qHeads * headDim + 7
        let kvValid = start + chunk
        var rng = SeedTree(seed).key("prefill-attn-start\(start)-chunk\(chunk)-window\(window)")
        var q = [Float](repeating: 0, count: chunk * qStride)
        var k = [Float](repeating: 0, count: kvValid * kvStride)
        var v = [Float](repeating: 0, count: kvValid * kvStride)

        for t in 0..<chunk {
            for h in 0..<qHeads {
                for d in 0..<headDim {
                    q[t * qStride + h * headDim + d] = rng.uniform(-0.35, 0.35)
                }
            }
        }
        for pos in 0..<kvValid {
            for h in 0..<kvHeads {
                for d in 0..<headDim {
                    k[pos * kvStride + h * headDim + d] = rng.uniform(-0.35, 0.35)
                    v[pos * kvStride + h * headDim + d] = rng.uniform(-0.35, 0.35)
                }
            }
        }

        return Fixture(q: q, k: k, v: v,
                       qStride: qStride, kvStride: kvStride, oStride: oStride,
                       headDim: headDim, qHeads: qHeads, kvHeads: kvHeads,
                       start: start, chunk: chunk, kvValid: kvValid,
                       window: window, scale: 1.0)
    }

    private static func runAndCompare(_ fixture: Fixture, label: String) throws {
        let actual = try Self.runKernel(fixture)
        let reference = Self.reference(fixture)
        let maxAbs = RelError.maxAbsDiff(actual, reference)
        let rel = RelError.compute(actual: actual, reference: reference)
        #expect(maxAbs <= 2e-2, "\(label) maxAbs=\(maxAbs) rel=\(rel)")
        #expect(rel <= 2e-2, "\(label) rel=\(rel) maxAbs=\(maxAbs)")
    }

    private static func runKernel(
        _ fixture: Fixture,
        path: RuntimePrefillAttentionPath = .causalTiled
    ) throws -> [Float] {
        let ctx = try MetalContext()
        let prefill = try PrefillAttention(context: ctx)
        let outCount = fixture.chunk * fixture.oStride

        guard let qBuf = Fp16Buffer.make(ctx.device, values: fixture.q),
              let kBuf = Fp16Buffer.make(ctx.device, values: fixture.k),
              let vBuf = Fp16Buffer.make(ctx.device, values: fixture.v),
              let outBuf = Fp16Buffer.make(ctx.device, count: outCount) else {
            Issue.record("alloc failed")
            return []
        }

        let params = PrefillAttentionParams(
            startPosition: UInt32(fixture.start),
            queryCount: UInt32(fixture.chunk),
            headDim: UInt32(fixture.headDim),
            numQHeads: UInt32(fixture.qHeads),
            numKVHeads: UInt32(fixture.kvHeads),
            kvValidCount: UInt32(fixture.kvValid),
            slidingWindow: UInt32(fixture.window),
            kvTokenStrideElements: UInt32(fixture.kvStride),
            qTokenStrideElements: UInt32(fixture.qStride),
            oTokenStrideElements: UInt32(fixture.oStride),
            scale: fixture.scale,
            bidirectionalBlockStart: UInt32(fixture.bidirectionalBlock?.lowerBound ?? 0),
            bidirectionalBlockEnd: UInt32(fixture.bidirectionalBlock?.upperBound ?? 0))

        let cb = ctx.queue.makeCommandBuffer()!
        try prefill.encodeCausal(commandBuffer: cb,
                             q: qBuf,
                             k: kBuf,
                             v: vBuf,
                             out: outBuf,
                             params: params,
                             kvRingCapacity: 0,
                             layerKind: fixture.window == 0 ? .full : .slidingWindow,
                             path: path)
        cb.commit()
        cb.waitUntilCompleted()
        try checkCommandBufferError(cb)

        let out = Fp16Buffer.read(outBuf, count: outCount)
        var compact = [Float](repeating: 0, count: fixture.chunk * fixture.qHeads * fixture.headDim)
        for t in 0..<fixture.chunk {
            for h in 0..<fixture.qHeads {
                for d in 0..<fixture.headDim {
                    compact[(t * fixture.qHeads + h) * fixture.headDim + d] =
                        out[t * fixture.oStride + h * fixture.headDim + d]
                }
            }
        }
        return compact
    }

    private static func reference(_ fixture: Fixture) -> [Float] {
        var out = [Float](repeating: 0, count: fixture.chunk * fixture.qHeads * fixture.headDim)
        let qPerKV = fixture.qHeads / fixture.kvHeads
        for t in 0..<fixture.chunk {
            let absQ = fixture.start + t
            let first: Int
            if fixture.window == 0 {
                first = 0
            } else {
                first = max(0, absQ + 1 - fixture.window)
            }
            // Only sliding-window layers make an image block bidirectional.
            let block = fixture.window == 0 ? nil : fixture.bidirectionalBlock
            let last: Int
            if let block, block.contains(absQ) {
                last = min(fixture.kvValid, block.upperBound)
            } else {
                last = min(fixture.kvValid, absQ + 1)
            }
            for qh in 0..<fixture.qHeads {
                let kvh = qh / qPerKV
                var scores: [Float] = []
                scores.reserveCapacity(last - first)
                for key in first..<last {
                    var score: Float = 0
                    for d in 0..<fixture.headDim {
                        let qv = fixture.q[t * fixture.qStride + qh * fixture.headDim + d]
                        let kv = fixture.k[key * fixture.kvStride + kvh * fixture.headDim + d]
                        score += qv * kv
                    }
                    scores.append(score * fixture.scale)
                }
                let maxScore = scores.max() ?? -.infinity
                var denom: Float = 0
                for score in scores {
                    denom += Foundation.exp(score - maxScore)
                }
                for d in 0..<fixture.headDim {
                    var acc: Float = 0
                    for (i, key) in (first..<last).enumerated() {
                        let w = Foundation.exp(scores[i] - maxScore)
                        acc += w * fixture.v[key * fixture.kvStride + kvh * fixture.headDim + d]
                    }
                    out[(t * fixture.qHeads + qh) * fixture.headDim + d] = denom > 0 ? acc / denom : 0
                }
            }
        }
        return out
    }
}
