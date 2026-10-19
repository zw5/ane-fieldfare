package import Metal

package final class VisionPrimitives {
    private let normalize: any MTLComputePipelineState
    private let addPosition: any MTLComputePipelineState
    private let rmsnorm: any MTLComputePipelineState
    private let postnormResidual: any MTLComputePipelineState
    private let qkvNormRoPEPadded: any MTLComputePipelineState
    private let pool: any MTLComputePipelineState
    private let bfloatToHalf: any MTLComputePipelineState

    package init(context: MetalContext) throws {
        normalize = try context.pipeline("vision_normalize_patches")
        addPosition = try context.pipeline("vision_add_position")
        rmsnorm = try context.pipeline("vision_rmsnorm_rows")
        postnormResidual = try context.pipeline("vision_postnorm_residual")
        qkvNormRoPEPadded = try context.pipeline("vision_qkv_norm_rope_padded")
        pool = try context.pipeline("vision_pool_standardize_norm")
        bfloatToHalf = try context.pipeline("vision_bfloat_to_half")
    }

    package func encodeNormalize(commandBuffer: any MTLCommandBuffer,
                                 input: any MTLBuffer, output: any MTLBuffer,
                                 rows: Int, paddedRows: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(normalize)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        let real = UInt32(rows), padded = UInt32(paddedRows)
        encoder.setValue(real, index: 2)
        encoder.setValue(padded, index: 3)
        dispatch1D(encoder, count: paddedRows * 768, pipeline: normalize)
    }

    package func encodeAddPosition(commandBuffer: any MTLCommandBuffer,
                                   hidden: any MTLBuffer,
                                   table: any MTLBuffer, tableOffset: Int,
                                   positions: any MTLBuffer, rows: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(addPosition)
        encoder.setBuffer(hidden, offset: 0, index: 0)
        encoder.setBuffer(table, offset: tableOffset, index: 1)
        encoder.setBuffer(positions, offset: 0, index: 2)
        let count = UInt32(rows)
        encoder.setValue(count, index: 3)
        dispatch1D(encoder, count: rows * 1_152, pipeline: addPosition)
    }

    package func encodeRMSNorm(commandBuffer: any MTLCommandBuffer,
                               input: any MTLBuffer, output: any MTLBuffer,
                               weightBuffer: any MTLBuffer, weightOffset: Int,
                               rows: Int, width: Int = 1_152) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(rmsnorm)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer, offset: weightOffset, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        let rowCount = UInt32(rows), dimension = UInt32(width)
        encoder.setValue(rowCount, index: 3)
        encoder.setValue(dimension, index: 4)
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 288, height: 1, depth: 1))
        encoder.endEncoding()
    }

    package func encodePostnormResidual(commandBuffer: any MTLCommandBuffer,
                                        residual: any MTLBuffer, branch: any MTLBuffer,
                                        weightBuffer: any MTLBuffer, weightOffset: Int,
                                        output: any MTLBuffer, rows: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(postnormResidual)
        encoder.setBuffer(residual, offset: 0, index: 0)
        encoder.setBuffer(branch, offset: 0, index: 1)
        encoder.setBuffer(weightBuffer, offset: weightOffset, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        let rowCount = UInt32(rows), width: UInt32 = 1_152
        encoder.setValue(rowCount, index: 4)
        encoder.setValue(width, index: 5)
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 288, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Q/K/V are in attention's padded head-major layout, `paddedRowStride`
    /// rows per head.
    package func encodeQKV(commandBuffer: any MTLCommandBuffer,
                           q: any MTLBuffer, k: any MTLBuffer, v: any MTLBuffer,
                           weights: any MTLBuffer, qWeightOffset: Int, kWeightOffset: Int,
                           positions: any MTLBuffer, rows: Int, heads: Int,
                           paddedRowStride: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        let pipeline = qkvNormRoPEPadded
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(q, offset: 0, index: 0)
        encoder.setBuffer(k, offset: 0, index: 1)
        encoder.setBuffer(v, offset: 0, index: 2)
        encoder.setBuffer(weights, offset: qWeightOffset, index: 3)
        encoder.setBuffer(weights, offset: kWeightOffset, index: 4)
        encoder.setBuffer(positions, offset: 0, index: 5)
        let rowCount = UInt32(rows), headCount = UInt32(heads)
        encoder.setValue(rowCount, index: 6)
        encoder.setValue(headCount, index: 7)
        let stride = UInt32(paddedRowStride)
        encoder.setValue(stride, index: 8)
        dispatch1D(encoder, count: rows * heads, pipeline: pipeline)
    }

    package func encodePool(commandBuffer: any MTLCommandBuffer,
                            input: any MTLBuffer,
                            weights: any MTLBuffer,
                            stdBiasOffset: Int, stdScaleOffset: Int,
                            output: any MTLBuffer,
                            patchWidth: Int, patchHeight: Int) {
        let rows = (patchWidth / 3) * (patchHeight / 3)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pool)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(weights, offset: stdBiasOffset, index: 1)
        encoder.setBuffer(weights, offset: stdScaleOffset, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        let width = UInt32(patchWidth), height = UInt32(patchHeight)
        encoder.setValue(width, index: 4)
        encoder.setValue(height, index: 5)
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    package func encodeBFloatToHalf(commandBuffer: any MTLCommandBuffer,
                                    input: any MTLBuffer, output: any MTLBuffer,
                                    count: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(bfloatToHalf)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        let value = UInt32(count)
        encoder.setValue(value, index: 2)
        dispatch1D(encoder, count: count, pipeline: bfloatToHalf)
    }

    private func dispatch1D(_ encoder: any MTLComputeCommandEncoder,
                            count: Int, pipeline: any MTLComputePipelineState) {
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
    }

}
