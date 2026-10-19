import Foundation
import Metal

final class PrefillEmbedLookupInt4 {
    private let pso: any MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("prefill_embed_lookup_int4_block")
    }

    func encode(commandBuffer: any MTLCommandBuffer,
                       table: any MTLBuffer, tableOffset: Int = 0,
                       scales: any MTLBuffer, scalesOffset: Int = 0,
                       biases: any MTLBuffer, biasesOffset: Int = 0,
                       tokens: any MTLBuffer, tokensOffset: Int = 0,
                       out: any MTLBuffer, outOffset: Int = 0,
                       t: UInt32,
                       d: UInt32,
                       outScale: Float) {
        precondition(d % UInt32(Quantization.groupSize) == 0,
                     "D must be a multiple of \(Quantization.groupSize)")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(table, offset: tableOffset, index: 0)
        enc.setBuffer(scales, offset: scalesOffset, index: 1)
        enc.setBuffer(biases, offset: biasesOffset, index: 2)
        enc.setBuffer(tokens, offset: tokensOffset, index: 3)
        enc.setBuffer(out, offset: outOffset, index: 4)
        let tVar = t
        let dVar = d
        let scaleVar = outScale
        enc.setValue(tVar, index: 5)
        enc.setValue(dVar, index: 6)
        enc.setValue(scaleVar, index: 7)
        enc.dispatchThreads(MTLSize(width: Int(d), height: Int(t), depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
    }
}

final class PrefillRMSNorm {
    private let psoBF16W: any MTLComputePipelineState

    init(context: MetalContext) throws {
        self.psoBF16W = try context.pipeline("prefill_rmsnorm_bf16w_block")
    }

    func encodeBF16W(commandBuffer: any MTLCommandBuffer,
                            x: any MTLBuffer, xOffset: Int = 0,
                            weight: any MTLBuffer, weightOffset: Int = 0,
                            out: any MTLBuffer, outOffset: Int = 0,
                            t: UInt32,
                            d: UInt32,
                            eps: Float) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoBF16W)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(weight, offset: weightOffset, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        let tVar = t
        let dVar = d
        let epsVar = eps
        enc.setValue(tVar, index: 3)
        enc.setValue(dVar, index: 4)
        enc.setValue(epsVar, index: 5)
        let threads = min(Int(psoBF16W.maxTotalThreadsPerThreadgroup), 256)
        enc.dispatchThreadgroups(MTLSize(width: Int(t), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }
}

final class PrefillInt4QMM {
    private let pso: any MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("prefill_dequant_int4_qmm_f16_block")
    }

    func encode(commandBuffer: any MTLCommandBuffer,
                       weights: any MTLBuffer, weightsOffset: Int = 0,
                       scales: any MTLBuffer, scalesOffset: Int = 0,
                       biases: any MTLBuffer, biasesOffset: Int = 0,
                       x: any MTLBuffer, xOffset: Int = 0,
                       y: any MTLBuffer, yOffset: Int = 0,
                       t: Int,
                       n: Int,
                       k: Int) {
        precondition(k % Quantization.groupSize == 0,
                     "K must be a multiple of \(Quantization.groupSize)")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(weights, offset: weightsOffset, index: 0)
        enc.setBuffer(scales, offset: scalesOffset, index: 1)
        enc.setBuffer(biases, offset: biasesOffset, index: 2)
        enc.setBuffer(x, offset: xOffset, index: 3)
        enc.setBuffer(y, offset: yOffset, index: 4)
        let tVar = UInt32(t)
        let nVar = UInt32(n)
        let kVar = UInt32(k)
        enc.setValue(tVar, index: 5)
        enc.setValue(nVar, index: 6)
        enc.setValue(kVar, index: 7)
        enc.dispatchThreadgroups(
            MTLSize(width: (n + 7) / 8, height: (t + 7) / 8, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        enc.endEncoding()
    }
}
