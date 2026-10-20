import Foundation
import Metal

@frozen
public struct PrefillTokenExpertPair: Equatable, Sendable {
    public var token: UInt32
    public var expert: UInt32
    public var rank: UInt32
    public var weightBitsAndReserved: UInt32

    public init(token: UInt32, expert: UInt32, rank: UInt32, weight: Float16) {
        self.token = token
        self.expert = expert
        self.rank = rank
        self.weightBitsAndReserved = UInt32(weight.bitPattern)
    }

    public var weight: Float16 {
        Float16(bitPattern: UInt16(truncatingIfNeeded: weightBitsAndReserved))
    }
}

final class PrefillRouter {
    private let pso: any MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("prefill_router_gemma4_block")
    }

    func encodeGemma4Block(commandBuffer: any MTLCommandBuffer,
                           weights: any MTLBuffer,
                           weightsOffset: Int = 0,
                           scales: any MTLBuffer,
                           scalesOffset: Int = 0,
                           biases: any MTLBuffer,
                           biasesOffset: Int = 0,
                           hidden: any MTLBuffer,
                           effectiveScale: any MTLBuffer,
                           perExpertScale: any MTLBuffer,
                           perExpertScaleOffset: Int = 0,
                           outIndices: any MTLBuffer,
                           outWeights: any MTLBuffer,
                           queryCount: UInt32,
                           numExperts: UInt32,
                           d: UInt32,
                           topK: UInt32,
                           hiddenStrideElements: UInt32) throws {
        precondition(queryCount > 0, "queryCount must be positive")
        precondition(numExperts <= 256, "numExperts > 256 is not supported")
        precondition(topK > 0 && topK <= 64, "topK must be in 1...64")
        precondition(d % UInt32(Quantization.groupSize) == 0,
                     "D must be a multiple of \(Quantization.groupSize)")
        precondition(hiddenStrideElements >= d, "hidden stride is too small")
        let enc = try commandBuffer.makeTracedComputeEncoder()
        enc.setComputePipelineState(pso)
        enc.setBuffer(weights, offset: weightsOffset, index: 0)
        enc.setBuffer(scales, offset: scalesOffset, index: 1)
        enc.setBuffer(biases, offset: biasesOffset, index: 2)
        enc.setBuffer(hidden, offset: 0, index: 3)
        enc.setBuffer(effectiveScale, offset: 0, index: 4)
        enc.setBuffer(perExpertScale, offset: perExpertScaleOffset, index: 5)
        enc.setBuffer(outIndices, offset: 0, index: 6)
        enc.setBuffer(outWeights, offset: 0, index: 7)
        enc.setValue(queryCount, index: 8)
        enc.setValue(numExperts, index: 9)
        enc.setValue(d, index: 10)
        enc.setValue(topK, index: 11)
        enc.setValue(hiddenStrideElements, index: 12)
        let tgWidth = min(max(Int(numExperts), 32), pso.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreadgroups(MTLSize(width: Int(queryCount), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
        enc.endEncoding()
    }

    static func makeTokenExpertPairs(indices: [UInt32],
                                     weights: [Float16],
                                     queryCount: Int,
                                     topK: Int) -> [PrefillTokenExpertPair] {
        precondition(queryCount >= 0, "queryCount must be non-negative")
        precondition(topK >= 0, "topK must be non-negative")
        precondition(indices.count == queryCount * topK, "indices count mismatch")
        precondition(weights.count == queryCount * topK, "weights count mismatch")
        var pairs: [PrefillTokenExpertPair] = []
        pairs.reserveCapacity(indices.count)
        for token in 0..<queryCount {
            for rank in 0..<topK {
                let i = token * topK + rank
                pairs.append(PrefillTokenExpertPair(token: UInt32(token),
                                                    expert: indices[i],
                                                    rank: UInt32(rank),
                                                    weight: weights[i]))
            }
        }
        return pairs
    }
}
