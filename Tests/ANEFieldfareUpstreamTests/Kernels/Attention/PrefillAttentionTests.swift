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
        let fixture = Self.makeFixture(start: visibleKeys - 1,
                                       chunk: 1,
                                       window: 0,
                                       seed: 0xA870 + UInt64(visibleKeys),
                                       headDim: 512,
                                       qHeads: 16,
                                       kvHeads: 2)
        // Full attention at the production shape runs the TensorOps kernel.
        let candidate = try Self.runKernel(fixture)
        let repeated = try Self.runKernel(fixture)
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

    /// Gemma's attention shapes on the grouped-query tensor kernel, across
    /// its query groups and key blocks: head size 256 with 16 query heads
    /// over 8 KV heads (sliding layers, with and without a wrapped cache ring
    /// that cuts blocks at the wrap; like production's, the ring holds the
    /// window plus the chunk) and 512 over 2 (full layers).
    @Test(arguments: [(256, 8, 0, 7, 0, 0), (256, 8, 61, 70, 0, 0), (256, 8, 0, 70, 13, 0),
                      (256, 8, 300, 150, 200, 0), (256, 8, 500, 150, 200, 360), (256, 8, 700, 90, 200, 291),
                      (512, 2, 0, 7, 0, 0), (512, 2, 61, 70, 0, 0), (512, 2, 0, 130, 0, 0), (512, 2, 1_000, 5, 0, 0)])
    func gqaTensorAttentionMatchesReference(
        _ shape: (headDim: Int, kvHeads: Int, start: Int, chunk: Int, window: Int, ring: Int)
    ) throws {
        let (headDim, kvHeads, start, chunk, window, ring) = shape
        let qHeads = 16
        let qStride = qHeads * headDim, kvStride = kvHeads * headDim, oStride = qHeads * headDim
        let kvValid = start + chunk
        let scale = 1 / Float(headDim).squareRoot()
        var rng = SeedTree(0x6E44).key("gqa-prefill-attn-\(headDim)-\(start)-\(chunk)")
        let q = (0..<(chunk * qStride)).map { _ in rng.uniform(-1, 1) }
        let k = (0..<(kvValid * kvStride)).map { _ in rng.uniform(-1, 1) }
        let v = (0..<(kvValid * kvStride)).map { _ in rng.uniform(-1, 1) }

        let ctx = try MetalContext()
        let prefill = try PrefillAttention(context: ctx)
        let qBuf = try #require(Fp16Buffer.make(ctx.device, values: q))
        // A ring holds logical key `key` at slot `key % ring`; the newest
        // `ring` keys are present.
        func cache(_ values: [Float]) -> [Float] {
            guard ring > 0 else { return values }
            var slots = [Float](repeating: 0, count: ring * kvStride)
            for key in max(0, kvValid - ring)..<kvValid {
                for i in 0..<kvStride { slots[(key % ring) * kvStride + i] = values[key * kvStride + i] }
            }
            return slots
        }
        let kBuf = try #require(Fp16Buffer.make(ctx.device, values: cache(k)))
        let vBuf = try #require(Fp16Buffer.make(ctx.device, values: cache(v)))
        let outBuf = try #require(Fp16Buffer.make(ctx.device, count: chunk * oStride))
        let params = PrefillAttentionParams(
            startPosition: UInt32(start), queryCount: UInt32(chunk), headDim: UInt32(headDim),
            numQHeads: UInt32(qHeads), numKVHeads: UInt32(kvHeads), kvValidCount: UInt32(kvValid),
            slidingWindow: UInt32(window), kvTokenStrideElements: UInt32(kvStride),
            qTokenStrideElements: UInt32(qStride), oTokenStrideElements: UInt32(oStride),
            scale: scale, bidirectionalBlockStart: 0, bidirectionalBlockEnd: 0)
        let cb = try #require(ctx.queue.makeCommandBuffer())
        try prefill.encodeCausal(commandBuffer: cb, q: qBuf, k: kBuf, v: vBuf, out: outBuf, params: params,
                                 kvRingCapacity: UInt32(ring), layerKind: window == 0 ? .full : .slidingWindow)
        cb.commit()
        cb.waitUntilCompleted()
        try checkCommandBufferError(cb)
        let out = Fp16Buffer.read(outBuf, count: chunk * oStride)

        // The rounded inputs the kernel sees.
        let qh = q.map { Float(Float16($0)) }, kh = k.map { Float(Float16($0)) }, vh = v.map { Float(Float16($0)) }
        var reference = [Float](repeating: 0, count: chunk * oStride)
        for t in 0..<chunk {
            let last = start + t + 1
            let first = window > 0 ? max(0, last - window) : 0
            for h in 0..<qHeads {
                let kvh = h / (qHeads / kvHeads)
                var scores = [Float](repeating: -.infinity, count: last)
                for key in first..<last {
                    var dot: Float = 0
                    for d in 0..<headDim { dot += qh[t * qStride + h * headDim + d] * kh[key * kvStride + kvh * headDim + d] }
                    scores[key] = dot * scale
                }
                let m = scores.max() ?? -.infinity
                var denom: Float = 0
                for score in scores { denom += Foundation.exp(score - m) }
                for d in 0..<headDim {
                    var acc: Float = 0
                    for key in first..<last { acc += Foundation.exp(scores[key] - m) * vh[key * kvStride + kvh * headDim + d] }
                    reference[t * oStride + h * headDim + d] = acc / denom
                }
            }
        }
        let maxAbs = RelError.maxAbsDiff(out, reference)
        #expect(maxAbs <= 2e-2, "head \(headDim) start \(start) chunk \(chunk) window \(window) ring \(ring) maxAbs=\(maxAbs)")
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

    private static func runKernel(_ fixture: Fixture) throws -> [Float] {
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
                             layerKind: fixture.window == 0 ? .full : .slidingWindow)
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
