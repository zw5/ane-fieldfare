import Foundation
import Metal

final class PrefillQKVEpilogue {
    private let perHeadNorm: PrefillPerHeadNorm
    private let rope: PrefillRoPE

    init(context: MetalContext) throws {
        self.perHeadNorm = try PrefillPerHeadNorm(context: context)
        self.rope = try PrefillRoPE(context: context)
    }

    func encode(commandBuffer: any MTLCommandBuffer,
                q: any MTLBuffer,
                k: any MTLBuffer,
                v: any MTLBuffer,
                qWeight: any MTLBuffer,
                qWeightOffset: Int = 0,
                kWeight: any MTLBuffer,
                kWeightOffset: Int = 0,
                startPosition: UInt32,
                queryCount: UInt32,
                headDim: UInt32,
                numQHeads: UInt32,
                numKVHeads: UInt32,
                qTokenStrideElements: UInt32,
                kvTokenStrideElements: UInt32,
                theta: Float,
                rotatedPairs: UInt32,
                eps: Float) throws {
        precondition(queryCount > 0, "queryCount must be positive")
        precondition(headDim > 0 && headDim % 2 == 0, "headDim must be positive and even")
        precondition(numQHeads > 0, "numQHeads must be positive")
        precondition(numKVHeads > 0, "numKVHeads must be positive")
        precondition(rotatedPairs > 0, "rotatedPairs must be positive")
        precondition(rotatedPairs * 2 <= headDim, "rotatedPairs must fit inside one head")
        precondition(qTokenStrideElements >= numQHeads * headDim,
                     "Q token stride is too small")
        precondition(kvTokenStrideElements >= numKVHeads * headDim,
                     "KV token stride is too small")

        try perHeadNorm.encodeBF16W(commandBuffer: commandBuffer,
                                    x: q,
                                    weight: qWeight,
                                    weightOffset: qWeightOffset,
                                    out: q,
                                    queryCount: queryCount,
                                    headDim: headDim,
                                    numHeads: numQHeads,
                                    tokenStrideElements: qTokenStrideElements,
                                    eps: eps)
        try perHeadNorm.encodeBF16W(commandBuffer: commandBuffer,
                                    x: k,
                                    weight: kWeight,
                                    weightOffset: kWeightOffset,
                                    out: k,
                                    queryCount: queryCount,
                                    headDim: headDim,
                                    numHeads: numKVHeads,
                                    tokenStrideElements: kvTokenStrideElements,
                                    eps: eps)
        try perHeadNorm.encodeNoScale(commandBuffer: commandBuffer,
                                      x: v,
                                      out: v,
                                      queryCount: queryCount,
                                      headDim: headDim,
                                      numHeads: numKVHeads,
                                      tokenStrideElements: kvTokenStrideElements,
                                      eps: eps)

        if rotatedPairs * 2 == headDim {
            try rope.encodeDefaultNeox(commandBuffer: commandBuffer,
                                       data: q,
                                       startPosition: startPosition,
                                       queryCount: queryCount,
                                       headDim: headDim,
                                       numHeads: numQHeads,
                                       tokenStrideElements: qTokenStrideElements,
                                       theta: theta)
            try rope.encodeDefaultNeox(commandBuffer: commandBuffer,
                                       data: k,
                                       startPosition: startPosition,
                                       queryCount: queryCount,
                                       headDim: headDim,
                                       numHeads: numKVHeads,
                                       tokenStrideElements: kvTokenStrideElements,
                                       theta: theta)
        } else {
            try rope.encodeProportionalNeox(commandBuffer: commandBuffer,
                                            data: q,
                                            startPosition: startPosition,
                                            queryCount: queryCount,
                                            headDim: headDim,
                                            numHeads: numQHeads,
                                            rotatedPairs: rotatedPairs,
                                            tokenStrideElements: qTokenStrideElements,
                                            theta: theta)
            try rope.encodeProportionalNeox(commandBuffer: commandBuffer,
                                            data: k,
                                            startPosition: startPosition,
                                            queryCount: queryCount,
                                            headDim: headDim,
                                            numHeads: numKVHeads,
                                            rotatedPairs: rotatedPairs,
                                            tokenStrideElements: kvTokenStrideElements,
                                            theta: theta)
        }
    }
}
