import Foundation
import Metal

final class FusedQKVGEMV {
    private struct Shape: Hashable {
        var qRows: UInt32
        var kvRows: UInt32
        var n: UInt32
    }

    private let pso: any MTLComputePipelineState
    private let specializedPSOs: [Shape: any MTLComputePipelineState]

    private static let realDecodeShapes: [Shape] = [
        Shape(qRows: 4096, kvRows: 2048, n: 2816),
        Shape(qRows: 8192, kvRows: 1024, n: 2816),
    ]

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("dequant_int4_qkv_gemv_simd",
                                        constants: [],
                                        maxTotalThreadsPerThreadgroup: 512)
        var variants: [Shape: any MTLComputePipelineState] = [:]
        for shape in Self.realDecodeShapes {
            variants[shape] = try context.pipeline(
                "dequant_int4_qkv_gemv_simd",
                constants: [
                    MetalFunctionConstant(index: 23, value: .uint32(shape.qRows)),
                    MetalFunctionConstant(index: 24, value: .uint32(shape.kvRows)),
                    MetalFunctionConstant(index: 25, value: .uint32(shape.n)),
                    MetalFunctionConstant(index: 26, value: .bool(true)),
                ],
                maxTotalThreadsPerThreadgroup: 512)
        }
        self.specializedPSOs = variants
    }

    func encode(commandBuffer: any MTLCommandBuffer,
                       qWeights: any MTLBuffer, qWeightsOffset: Int = 0,
                       qScales: any MTLBuffer, qScalesOffset: Int = 0,
                       qBiases: any MTLBuffer, qBiasesOffset: Int = 0,
                       kWeights: any MTLBuffer, kWeightsOffset: Int = 0,
                       kScales: any MTLBuffer, kScalesOffset: Int = 0,
                       kBiases: any MTLBuffer, kBiasesOffset: Int = 0,
                       vWeights: any MTLBuffer, vWeightsOffset: Int = 0,
                       vScales: any MTLBuffer, vScalesOffset: Int = 0,
                       vBiases: any MTLBuffer, vBiasesOffset: Int = 0,
                       x: any MTLBuffer,
                       qOut: any MTLBuffer, qOutOffset: Int = 0,
                       kOut: any MTLBuffer, kOutOffset: Int = 0,
                       vOut: any MTLBuffer, vOutOffset: Int = 0,
                       qRows: UInt32,
                       kvRows: UInt32,
                       n: UInt32) {
        precondition(n % UInt32(Quantization.groupSize) == 0,
                     "N must be a multiple of \(Quantization.groupSize)")
        precondition(qWeightsOffset % 2 == 0 &&
                     kWeightsOffset % 2 == 0 &&
                     vWeightsOffset % 2 == 0,
                     "FusedQKVGEMV needs 2-aligned weights offsets")
        guard let enc = commandBuffer.makeTracedComputeEncoder() else { return }
        let shape = Shape(qRows: qRows, kvRows: kvRows, n: n)
        enc.setComputePipelineState(specializedPSOs[shape] ?? pso)
        enc.setBuffer(qWeights, offset: qWeightsOffset, index: 0)
        enc.setBuffer(qScales, offset: qScalesOffset, index: 1)
        enc.setBuffer(qBiases, offset: qBiasesOffset, index: 2)
        enc.setBuffer(kWeights, offset: kWeightsOffset, index: 3)
        enc.setBuffer(kScales, offset: kScalesOffset, index: 4)
        enc.setBuffer(kBiases, offset: kBiasesOffset, index: 5)
        enc.setBuffer(vWeights, offset: vWeightsOffset, index: 6)
        enc.setBuffer(vScales, offset: vScalesOffset, index: 7)
        enc.setBuffer(vBiases, offset: vBiasesOffset, index: 8)
        enc.setBuffer(x, offset: 0, index: 9)
        enc.setBuffer(qOut, offset: qOutOffset, index: 10)
        enc.setBuffer(kOut, offset: kOutOffset, index: 11)
        enc.setBuffer(vOut, offset: vOutOffset, index: 12)
        let qVar = qRows
        let kvVar = kvRows
        let nVar = n
        enc.setValue(qVar, index: 13)
        enc.setValue(kvVar, index: 14)
        enc.setValue(nVar, index: 15)
        let totalRows = Int(qRows + 2 * kvRows)
        enc.dispatchThreadgroups(MTLSize(width: (totalRows + 7) / 8,
                                         height: 1,
                                         depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256,
                                                                 height: 1,
                                                                 depth: 1))
        enc.endEncoding()
    }
}
