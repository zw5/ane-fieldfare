import Foundation
import Metal
import Testing
@testable import ANEFieldfareHotpath
import ANEFieldfareValidationSupport

/// The fused QKV GEMV + norm/RoPE epilogue against the INT4 GEMV followed by
/// the CPU per-head norms and NeoX RoPE, at both decode layer shapes:
/// sliding-window layers rotate every pair (default NeoX), full-attention
/// layers rotate 64 of their 256 pairs (proportional).
@Suite struct FusedQKVPipelineTests {
    @Test(arguments: [true, false])
    func fusedPipelineMatchesReference(slidingWindow: Bool) throws {
        let headDim = slidingWindow ? 256 : 512
        let numQHeads = 16
        let numKVHeads = slidingWindow ? 8 : 2
        let qRows = numQHeads * headDim
        let kvRows = numKVHeads * headDim
        let n = 2816
        let rotatedPairs: UInt32 = slidingWindow ? 128 : 64
        let theta: Float = slidingWindow ? 10_000 : 1_000_000
        let position = 12
        let qProjection = Self.makeProjection(rows: qRows, n: n, seed: 0x5156_5049)
        let kvProjection = Self.makeProjection(rows: kvRows, n: n, seed: 0x4B56_5049)
        var rng = SplitMix64(seed: 0x585F_5049)
        let x = (0..<n).map { _ in Float16(rng.uniform(-1, 1)) }
        let qNorm = (0..<headDim).map { _ in
            Quantization.bf16Bits(rng.uniform(0.5, 1.5))
        }
        let kNorm = (0..<headDim).map { _ in
            Quantization.bf16Bits(rng.uniform(0.5, 1.5))
        }

        let ctx = try MetalContext()
        let gemv = try DequantInt4GEMV(context: ctx)
        let fusedGEMV = try FusedQKVGEMV(context: ctx)
        let fusedEpilogue = try FusedQKVEpilogue(context: ctx)

        func buffer<T: BitwiseCopyable>(_ values: [T]) throws -> any MTLBuffer {
            try #require(unsafe ctx.device.makeBuffer(bytes: values,
                                                      length: values.count * MemoryLayout<T>.stride,
                                                      options: .storageModeShared))
        }
        func output(_ count: Int) throws -> any MTLBuffer {
            try #require(Fp16Buffer.make(ctx.device, count: count))
        }
        let qWeights = try buffer(qProjection.weights)
        let qScales = try buffer(qProjection.scales)
        let qBiases = try buffer(qProjection.biases)
        let kvWeights = try buffer(kvProjection.weights)
        let kvScales = try buffer(kvProjection.scales)
        let kvBiases = try buffer(kvProjection.biases)
        let xBuffer = try buffer(x)
        let qNormBuffer = try buffer(qNorm)
        let kNormBuffer = try buffer(kNorm)
        let qProjected = try output(qRows), kvProjected = try output(kvRows)
        let qFused = try output(qRows), kFused = try output(kvRows), vFused = try output(kvRows)
        let commandBuffer = try #require(ctx.queue.makeCommandBuffer())

        try gemv.encode(commandBuffer: commandBuffer,
                        weights: qWeights, scales: qScales, biases: qBiases,
                        x: xBuffer, y: qProjected,
                        m: UInt32(qRows), n: UInt32(n))
        try gemv.encode(commandBuffer: commandBuffer,
                        weights: kvWeights, scales: kvScales, biases: kvBiases,
                        x: xBuffer, y: kvProjected,
                        m: UInt32(kvRows), n: UInt32(n))
        try fusedGEMV.encode(commandBuffer: commandBuffer,
                             qWeights: qWeights, qScales: qScales, qBiases: qBiases,
                             kWeights: kvWeights, kScales: kvScales, kBiases: kvBiases,
                             vWeights: kvWeights, vScales: kvScales, vBiases: kvBiases,
                             x: xBuffer,
                             qOut: qFused, kOut: kFused, vOut: vFused,
                             qRows: UInt32(qRows), kvRows: UInt32(kvRows), n: UInt32(n))
        try fusedEpilogue.encode(commandBuffer: commandBuffer,
                                 q: qFused, k: kFused, v: vFused,
                                 qWeight: qNormBuffer, kWeight: kNormBuffer,
                                 headDim: UInt32(headDim),
                                 numQHeads: UInt32(numQHeads),
                                 numKVHeads: UInt32(numKVHeads),
                                 position: UInt32(position),
                                 theta: theta,
                                 rotatedPairs: rotatedPairs,
                                 eps: 1e-6)

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            Issue.record("Command buffer failed: \(error)")
            return
        }

        let kv = Fp16Buffer.read(kvProjected, count: kvRows)
        let rope = (position: position, theta: theta, rotatedPairs: Int(rotatedPairs))
        let checks = [
            ("Q", Fp16Buffer.read(qFused, count: qRows),
             QKVEpilogueReference.apply(Fp16Buffer.read(qProjected, count: qRows), headDim: headDim,
                                        weight: qNorm, rope: rope)),
            ("K", Fp16Buffer.read(kFused, count: kvRows),
             QKVEpilogueReference.apply(kv, headDim: headDim, weight: kNorm, rope: rope)),
            ("V", Fp16Buffer.read(vFused, count: kvRows),
             QKVEpilogueReference.apply(kv, headDim: headDim, weight: nil, rope: nil)),
        ]
        for (label, actual, reference) in checks {
            let rel = RelError.compute(actual: actual, reference: reference)
            #expect(rel < Tolerance.fp16Reduction, "\(label) rel=\(rel)")
        }
    }

    private static func makeProjection(rows: Int,
                                       n: Int,
                                       seed: UInt64) -> (weights: [UInt8],
                                                        scales: [UInt16],
                                                        biases: [UInt16]) {
        let packedCount = rows * n / 2
        let groupCount = rows * n / Quantization.groupSize
        let weights = (0..<packedCount).map { index in
            UInt8(truncatingIfNeeded: (UInt64(index) &* 0x9E37_79B9) &+ seed)
        }
        let scales = (0..<groupCount).map { index in
            Quantization.bf16Bits(0.015625 + Float(index % 7) * 0.001953125)
        }
        let biases = (0..<groupCount).map { index in
            Quantization.bf16Bits(-0.125 + Float(index % 5) * 0.03125)
        }
        return (weights, scales, biases)
    }
}

/// CPU Gemma 4 Q/K/V epilogue of one token: per-head RMSNorm (eps 1e-6) with
/// the shared BF16 `weight` (Q/K) or none (V), then NeoX RoPE when `rope` is
/// given (Q/K).
enum QKVEpilogueReference {
    static func apply(_ x: [Float],
                      headDim: Int,
                      weight: [UInt16]?,
                      rope: (position: Int, theta: Float, rotatedPairs: Int)?) -> [Float] {
        let gain = weight.map { $0.map(Quantization.bf16ToFloat) }
            ?? Array(repeating: 1, count: headDim)
        let heads = x.count / headDim
        let normed = (0..<heads).flatMap { head in
            RmsNormRef.apply(x: Array(x[(head * headDim)..<((head + 1) * headDim)]),
                             weight: gain, eps: 1e-6)
        }
        guard let rope else { return normed }
        return RopeRef.applyNeox(input: normed, numTokens: 1, numHeads: heads, headDim: headDim,
                                 rotatedPairs: rope.rotatedPairs, position: rope.position,
                                 theta: rope.theta)
    }
}
