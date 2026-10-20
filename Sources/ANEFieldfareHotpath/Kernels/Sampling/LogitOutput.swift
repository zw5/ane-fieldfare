import Foundation
import Metal

/// Swift wrapper for `logit_softcap_softmax`.
///
///   z'[i] = softcap * tanh(z[i] / softcap)
///   p[i]  = exp(z'[i] - max_j z'[j]) / sum_j exp(z'[j] - max_j z'[j])
///
/// One threadgroup, single online-softmax pass (Milakov & Gimelshein). FP16
/// storage in and out, FP32 accumulation. A very large `softcap` disables
/// the cap.
final class LogitSoftcapSoftmax {
    private let pso: any MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("logit_softcap_softmax")
    }

    /// Encodes the kernel onto `commandBuffer`. `logits` and `probs` are FP16
    /// buffers of length `v`. Gemma 4 uses `softcap=30.0`.
    func encode(commandBuffer: any MTLCommandBuffer,
                       logits: any MTLBuffer,
                       probs: any MTLBuffer,
                       v: UInt32,
                       softcap: Float = 30.0) throws {
        let enc = try commandBuffer.makeTracedComputeEncoder()
        enc.setComputePipelineState(pso)
        enc.setBuffer(logits, offset: 0, index: 0)
        enc.setBuffer(probs,  offset: 0, index: 1)
        let vVar       = v
        enc.setValue(vVar, index: 2)
        enc.setValue(softcap, index: 3)

        let threadsPerGroup = min(Int(pso.maxTotalThreadsPerThreadgroup), 256)
        let gridSize = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        let tgSize   = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        enc.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
    }
}

/// Swift wrapper for `sample`.
///
/// Reads softmaxed probabilities (output of `LogitSoftcapSoftmax`) and writes
/// one UInt32 token id. With `temperature == 0` performs greedy argmax and
/// ignores top-k / top-p / seed. With `temperature > 0` applies top-p against
/// the full distribution, caps the surviving set with top-k, then applies
/// temperature sharpening (`p^(1/T)`) and samples via a seeded PRNG. Plain
/// temperature (no top-k/top-p) takes a Gumbel-max draw keyed by
/// `(seed, position, token)`.
final class Sample {
    private let pso: any MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("sample")
    }

    /// Encodes the sampler. `probs` is an FP16 buffer of length `v`. `outToken`
    /// must point at storage for one UInt32. The seed is the full 64-bit PRNG
    /// state.
    func encode(commandBuffer: any MTLCommandBuffer,
                       probs: any MTLBuffer,
                       outToken: any MTLBuffer,
                       v: UInt32,
                       temperature: Float = 1.0,
                       topK: UInt32 = 0,
                       topP: Float = 1.0,
                       seed: UInt64,
                       position: UInt32 = 0) throws {
        let enc = try commandBuffer.makeTracedComputeEncoder()
        enc.setComputePipelineState(pso)
        enc.setBuffer(probs,    offset: 0, index: 0)
        enc.setBuffer(outToken, offset: 0, index: 1)
        let vVar    = v
        let tVar    = temperature
        let kVar    = topK
        let pVar    = topP
        let sVar    = seed
        let posVar  = position
        enc.setValue(vVar, index: 2)
        enc.setValue(tVar, index: 3)
        enc.setValue(kVar, index: 4)
        enc.setValue(pVar, index: 5)
        enc.setValue(sVar, index: 6)
        enc.setValue(posVar, index: 7)

        let threadsPerGroup = min(Int(pso.maxTotalThreadsPerThreadgroup), 256)
        let gridSize = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        let tgSize   = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        enc.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
    }
}
enum SampleTopK64Error: Error {
    case unsupportedVocabulary(Int)
    case scratchAllocationFailed
}

/// Three-stage Top-64 sampler (temperature > 0, top-k 64). Intermediate
/// pairs remain in private GPU memory.
final class SampleTopK64 {
    private static let tileSize = 1024
    private static let keptPerTile = 64

    private let stage1PSO: any MTLComputePipelineState
    private let reducePSO: any MTLComputePipelineState
    private let finalPSO: any MTLComputePipelineState
    private let stage1Values: any MTLBuffer
    private let stage1Indices: any MTLBuffer
    private let stage2Values: any MTLBuffer
    private let stage2Indices: any MTLBuffer
    private let stage1Groups: Int
    private let stage2Groups: Int
    private let stage1Count: Int
    private let stage2Count: Int

    private let vocab: Int

    public init(context: MetalContext, vocab: Int) throws {
        guard vocab > 0, vocab <= 262_144 else {
            throw SampleTopK64Error.unsupportedVocabulary(vocab)
        }
        self.vocab = vocab
        self.stage1PSO = try context.pipeline("sample_topk64_stage1")
        self.reducePSO = try context.pipeline("sample_topk64_reduce")
        self.finalPSO = try context.pipeline("sample_topk64_final")

        self.stage1Groups = (vocab + Self.tileSize - 1) / Self.tileSize
        self.stage1Count = stage1Groups * Self.keptPerTile
        self.stage2Groups = (stage1Count + Self.tileSize - 1) / Self.tileSize
        self.stage2Count = stage2Groups * Self.keptPerTile

        let stage1ValueBytes = stage1Count * MemoryLayout<Float>.stride
        let stage1IndexBytes = stage1Count * MemoryLayout<UInt32>.stride
        let stage2ValueBytes = stage2Count * MemoryLayout<Float>.stride
        let stage2IndexBytes = stage2Count * MemoryLayout<UInt32>.stride

        guard let stage1Values = context.device.makeBuffer(
                  length: stage1ValueBytes, options: .storageModePrivate),
              let stage1Indices = context.device.makeBuffer(
                  length: stage1IndexBytes, options: .storageModePrivate),
              let stage2Values = context.device.makeBuffer(
                  length: stage2ValueBytes, options: .storageModePrivate),
              let stage2Indices = context.device.makeBuffer(
                  length: stage2IndexBytes, options: .storageModePrivate)
        else {
            throw SampleTopK64Error.scratchAllocationFailed
        }
        self.stage1Values = stage1Values
        self.stage1Indices = stage1Indices
        self.stage2Values = stage2Values
        self.stage2Indices = stage2Indices
    }

    public func encode(commandBuffer: any MTLCommandBuffer,
                       probs: any MTLBuffer,
                       outToken: any MTLBuffer,
                       temperature: Float,
                       topP: Float,
                       seed: UInt64) throws {
        let threads = MTLSize(width: 256, height: 1, depth: 1)

        let stage1 = try commandBuffer.makeTracedComputeEncoder()
        stage1.setComputePipelineState(stage1PSO)
        stage1.setBuffer(probs, offset: 0, index: 0)
        stage1.setBuffer(stage1Values, offset: 0, index: 1)
        stage1.setBuffer(stage1Indices, offset: 0, index: 2)
        stage1.setValue(UInt32(vocab), index: 3)
        stage1.dispatchThreadgroups(MTLSize(width: stage1Groups, height: 1, depth: 1),
                                    threadsPerThreadgroup: threads)
        stage1.endEncoding()

        let reduce = try commandBuffer.makeTracedComputeEncoder()
        reduce.setComputePipelineState(reducePSO)
        reduce.setBuffer(stage1Values, offset: 0, index: 0)
        reduce.setBuffer(stage1Indices, offset: 0, index: 1)
        reduce.setBuffer(stage2Values, offset: 0, index: 2)
        reduce.setBuffer(stage2Indices, offset: 0, index: 3)
        reduce.setValue(UInt32(stage1Count), index: 4)
        reduce.dispatchThreadgroups(MTLSize(width: stage2Groups, height: 1, depth: 1),
                                    threadsPerThreadgroup: threads)
        reduce.endEncoding()

        let final = try commandBuffer.makeTracedComputeEncoder()
        final.setComputePipelineState(finalPSO)
        final.setBuffer(stage2Values, offset: 0, index: 0)
        final.setBuffer(stage2Indices, offset: 0, index: 1)
        final.setBuffer(outToken, offset: 0, index: 2)
        final.setValue(UInt32(stage2Count), index: 3)
        final.setValue(temperature, index: 4)
        final.setValue(topP, index: 5)
        final.setValue(seed, index: 6)
        final.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                   threadsPerThreadgroup: threads)
        final.endEncoding()
    }
}
