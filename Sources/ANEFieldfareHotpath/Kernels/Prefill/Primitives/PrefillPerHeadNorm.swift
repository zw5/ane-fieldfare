import Foundation
import Metal

final class PrefillPerHeadNorm {
    private let psoBF16W: any MTLComputePipelineState
    private let psoNoScale: any MTLComputePipelineState
    /// One SIMD group a head row, for head sizes `simdFits` accepts.
    private let psoBF16WSimd: any MTLComputePipelineState
    private let psoNoScaleSimd: any MTLComputePipelineState
    private static let simdRowsPerGroup = 8

    init(context: MetalContext) throws {
        self.psoBF16W = try context.pipeline("prefill_rmsnorm_bf16w_perhead_block")
        self.psoNoScale = try context.pipeline("prefill_rmsnorm_no_scale_perhead_block")
        self.psoBF16WSimd = try context.pipeline("prefill_rmsnorm_bf16w_perhead_simd")
        self.psoNoScaleSimd = try context.pipeline("prefill_rmsnorm_no_scale_perhead_simd")
    }

    private static func simdFits(_ headDim: UInt32) -> Bool {
        headDim.isMultiple(of: 32) && headDim <= 512
    }

    private static func dispatch(_ enc: any MTLComputeCommandEncoder, simd: Bool,
                                 pso: any MTLComputePipelineState, queryCount: UInt32, numHeads: UInt32) {
        if simd {
            let rows = Int(queryCount) * Int(numHeads)
            enc.dispatchThreadgroups(MTLSize(width: (rows + simdRowsPerGroup - 1) / simdRowsPerGroup,
                                             height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32 * simdRowsPerGroup, height: 1, depth: 1))
        } else {
            let threads = min(pso.maxTotalThreadsPerThreadgroup, 256)
            enc.dispatchThreadgroups(MTLSize(width: Int(numHeads), height: Int(queryCount), depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        }
    }

    func encodeBF16W(commandBuffer: any MTLCommandBuffer,
                     x: any MTLBuffer,
                     weight: any MTLBuffer,
                     weightOffset: Int,
                     out: any MTLBuffer,
                     queryCount: UInt32,
                     headDim: UInt32,
                     numHeads: UInt32,
                     tokenStrideElements: UInt32,
                     eps: Float) throws {
        precondition(queryCount > 0, "queryCount must be positive")
        precondition(headDim > 0, "headDim must be positive")
        precondition(numHeads > 0, "numHeads must be positive")
        precondition(tokenStrideElements >= headDim * numHeads,
                     "token stride is too small")
        let enc = try commandBuffer.makeTracedComputeEncoder()
        let simd = Self.simdFits(headDim)
        enc.setComputePipelineState(simd ? psoBF16WSimd : psoBF16W)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(weight, offset: weightOffset, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
        enc.setValue(queryCount, index: 3)
        enc.setValue(headDim, index: 4)
        enc.setValue(numHeads, index: 5)
        enc.setValue(tokenStrideElements, index: 6)
        enc.setValue(eps, index: 7)
        Self.dispatch(enc, simd: simd, pso: psoBF16W, queryCount: queryCount, numHeads: numHeads)
        enc.endEncoding()
    }

    func encodeNoScale(commandBuffer: any MTLCommandBuffer,
                       x: any MTLBuffer,
                       out: any MTLBuffer,
                       queryCount: UInt32,
                       headDim: UInt32,
                       numHeads: UInt32,
                       tokenStrideElements: UInt32,
                       eps: Float) throws {
        precondition(queryCount > 0, "queryCount must be positive")
        precondition(headDim > 0, "headDim must be positive")
        precondition(numHeads > 0, "numHeads must be positive")
        precondition(tokenStrideElements >= headDim * numHeads,
                     "token stride is too small")
        let enc = try commandBuffer.makeTracedComputeEncoder()
        let simd = Self.simdFits(headDim)
        enc.setComputePipelineState(simd ? psoNoScaleSimd : psoNoScale)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(out, offset: 0, index: 1)
        enc.setValue(queryCount, index: 2)
        enc.setValue(headDim, index: 3)
        enc.setValue(numHeads, index: 4)
        enc.setValue(tokenStrideElements, index: 5)
        enc.setValue(eps, index: 6)
        Self.dispatch(enc, simd: simd, pso: psoNoScale, queryCount: queryCount, numHeads: numHeads)
        enc.endEncoding()
    }
}
