import Foundation
import Metal

final class PrefillPerHeadNorm {
    private let psoBF16W: any MTLComputePipelineState
    private let psoNoScale: any MTLComputePipelineState

    init(context: MetalContext) throws {
        self.psoBF16W = try context.pipeline("prefill_rmsnorm_bf16w_perhead_block")
        self.psoNoScale = try context.pipeline("prefill_rmsnorm_no_scale_perhead_block")
    }

    func encodeBF16W(commandBuffer: any MTLCommandBuffer,
                            x: any MTLBuffer,
                            xOffset: Int = 0,
                            weight: any MTLBuffer,
                            weightOffset: Int = 0,
                            out: any MTLBuffer,
                            outOffset: Int = 0,
                            queryCount: UInt32,
                            headDim: UInt32,
                            numHeads: UInt32,
                            tokenStrideElements: UInt32,
                            eps: Float) {
        precondition(queryCount > 0, "queryCount must be positive")
        precondition(headDim > 0, "headDim must be positive")
        precondition(numHeads > 0, "numHeads must be positive")
        precondition(tokenStrideElements >= headDim * numHeads,
                     "token stride is too small")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoBF16W)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(weight, offset: weightOffset, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        let t = queryCount
        let hd = headDim
        let heads = numHeads
        let stride = tokenStrideElements
        let epsVar = eps
        enc.setValue(t, index: 3)
        enc.setValue(hd, index: 4)
        enc.setValue(heads, index: 5)
        enc.setValue(stride, index: 6)
        enc.setValue(epsVar, index: 7)
        let threads = min(psoBF16W.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreadgroups(MTLSize(width: Int(numHeads), height: Int(queryCount), depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }

    func encodeNoScale(commandBuffer: any MTLCommandBuffer,
                              x: any MTLBuffer,
                              xOffset: Int = 0,
                              out: any MTLBuffer,
                              outOffset: Int = 0,
                              queryCount: UInt32,
                              headDim: UInt32,
                              numHeads: UInt32,
                              tokenStrideElements: UInt32,
                              eps: Float) {
        precondition(queryCount > 0, "queryCount must be positive")
        precondition(headDim > 0, "headDim must be positive")
        precondition(numHeads > 0, "numHeads must be positive")
        precondition(tokenStrideElements >= headDim * numHeads,
                     "token stride is too small")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoNoScale)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(out, offset: outOffset, index: 1)
        let t = queryCount
        let hd = headDim
        let heads = numHeads
        let stride = tokenStrideElements
        let epsVar = eps
        enc.setValue(t, index: 2)
        enc.setValue(hd, index: 3)
        enc.setValue(heads, index: 4)
        enc.setValue(stride, index: 5)
        enc.setValue(epsVar, index: 6)
        let threads = min(psoNoScale.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreadgroups(MTLSize(width: Int(numHeads), height: Int(queryCount), depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }
}
