import Foundation
import Metal

/// Q4_0 embedding lookup of one token. `outScale` fuses Gemma's
/// `sqrt(hidden_size)` post-embedding scale into the dequant pass.
final class EmbedLookupInt4 {
    private let pso: any MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("embed_lookup_int4")
    }

    /// Encodes the lookup. `table` and `scales` typically live inside one
    /// resident blob — pass that buffer with the per-region offsets.
    /// Pass `outScale = 1.0` to write the raw dequantized row.
    func encode(commandBuffer: any MTLCommandBuffer,
                       table:  any MTLBuffer, tableOffset:  Int = 0,
                       scales: any MTLBuffer, scalesOffset: Int = 0,
                       out:    any MTLBuffer, outOffset: Int = 0,
                       tokenId: UInt32,
                       d: UInt32,
                       outScale: Float) {
        precondition(d % UInt32(Quantization.groupSize) == 0,
                     "D must be a multiple of \(Quantization.groupSize)")
        guard let enc = commandBuffer.makeTracedComputeEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(table,  offset: tableOffset,  index: 0)
        enc.setBuffer(scales, offset: scalesOffset, index: 1)
        enc.setBuffer(out,    offset: outOffset,    index: 2)
        let tokenVar = tokenId
        let dVar     = d
        let sVar     = outScale
        enc.setValue(tokenVar, index: 3)
        enc.setValue(dVar, index: 4)
        enc.setValue(sVar, index: 5)

        let threadsPerGroup = min(Int(pso.maxTotalThreadsPerThreadgroup), 256)
        let gridSize = MTLSize(width: Int(d), height: 1, depth: 1)
        let tgSize   = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        enc.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
    }
}
