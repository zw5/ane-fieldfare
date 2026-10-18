import Metal

/// Final BF16 RMSNorm, INT4 affine lm-head projection, and greedy argmax.
/// The hot path writes one token ID without materializing vocab-sized logits.
final class LMHeadChainInt4 {
    static let rowsPerThreadgroup = 8
    /// Hidden rows one `encodeGreedyRows` call can score; matches the kernel.
    static let maxGreedyRows = 16
    private static let multiRowVocabRowsPerThreadgroup = 32

    private static let rowSummaryStride = 2
    private static let realDecodeD: UInt32 = 2816
    private static let realDecodeVocab: UInt32 = 262144
    private static let realDecodeHeadConstants: [MetalFunctionConstant] = [
        MetalFunctionConstant(index: 10, value: .uint32(realDecodeD)),
        MetalFunctionConstant(index: 11, value: .uint32(realDecodeVocab)),
        MetalFunctionConstant(index: 13, value: .bool(true)),
    ]

    private let rms: RMSNorm
    private let rowGreedy: MTLComputePipelineState
    private let rowGreedySpecialized: MTLComputePipelineState
    private let rowReducer: MTLComputePipelineState
    private let multiRowGreedy: MTLComputePipelineState
    private let multiRowReducer: MTLComputePipelineState
    private let xNormedRowsBuffer: MTLBuffer
    private let multiRowSummariesBuffer: MTLBuffer
    private let xNormedBuffer: MTLBuffer
    private let rowSummariesBuffer: MTLBuffer
    private let maxD: Int
    private let maxVocab: Int

    init(context: MetalContext,
         maxD: Int = 2816,
         maxVocab: Int = 262144) throws {
        self.rms = try RMSNorm(context: context)
        self.rowGreedy = try context.pipeline("lm_head_greedy_int4_rows_chunk_raw")
        self.rowGreedySpecialized = try context.pipeline(
            "lm_head_greedy_int4_rows_chunk_raw",
            constants: Self.realDecodeHeadConstants)
        self.rowReducer = try context.pipeline("lm_head_greedy_int4_rows_reduce")
        self.multiRowGreedy = try context.pipeline("lm_head_greedy_int4_multirow_chunk")
        self.multiRowReducer = try context.pipeline("lm_head_greedy_int4_multirow_reduce")
        self.maxD = maxD
        self.maxVocab = maxVocab

        let rowGroups = (maxVocab + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup
        let xLength = max(maxD, 1) * MemoryLayout<Float16>.size
        let summaryLength = rowGroups * Self.rowSummaryStride * MemoryLayout<Float>.size
        guard let xNormedBuffer = context.device.makeBuffer(
                  length: xLength,
                  options: .storageModePrivate),
              let rowSummariesBuffer = context.device.makeBuffer(
                  length: summaryLength,
                  options: .storageModePrivate) else {
            throw MetalError.noDevice
        }
        self.xNormedBuffer = xNormedBuffer
        self.rowSummariesBuffer = rowSummariesBuffer
        guard let xNormedRows = context.device.makeBuffer(
                  length: Self.maxGreedyRows * xLength, options: .storageModePrivate),
              let multiRowSummaries = context.device.makeBuffer(
                  length: Self.maxGreedyRows * summaryLength, options: .storageModePrivate) else {
            throw MetalError.noDevice
        }
        self.xNormedRowsBuffer = xNormedRows
        self.multiRowSummariesBuffer = multiRowSummaries
    }

    /// Greedy tokens for `rowCount` consecutive FP16 hidden rows starting at
    /// `hiddenOffset`, written as `UInt32` to `outTokens`. Reads the INT4 table
    /// once for all rows.
    func encodeGreedyRows(commandBuffer: MTLCommandBuffer,
                          hidden: MTLBuffer,
                          hiddenOffset: Int = 0,
                          rowCount: Int,
                          normWeight: MTLBuffer,
                          normOffset: Int = 0,
                          weights: MTLBuffer,
                          weightsOffset: Int = 0,
                          scales: MTLBuffer,
                          scalesOffset: Int = 0,
                          biases: MTLBuffer,
                          biasesOffset: Int = 0,
                          outTokens: MTLBuffer,
                          outTokensOffset: Int = 0,
                          d: UInt32,
                          vocab: UInt32,
                          rmsEps: Float = 1e-6) {
        precondition(rowCount > 0 && rowCount <= Self.maxGreedyRows,
                     "rowCount=\(rowCount) outside 1...\(Self.maxGreedyRows)")
        precondition(Int(d) <= maxD && Int(vocab) <= maxVocab)
        precondition(Int(d) % 256 == 0, "d must be a multiple of 256")
        let rowBytes = Int(d) * MemoryLayout<Float16>.stride
        for row in 0..<rowCount {
            rms.encodeBF16W(commandBuffer: commandBuffer,
                            x: hidden, xOffset: hiddenOffset + row * rowBytes,
                            weight: normWeight, weightOffset: normOffset,
                            out: xNormedRowsBuffer, outOffset: row * rowBytes,
                            d: d, eps: rmsEps)
        }
        let rowGroups = (Int(vocab) + Self.multiRowVocabRowsPerThreadgroup - 1) / Self.multiRowVocabRowsPerThreadgroup
        var dValue = d, vocabValue = vocab, rows = UInt32(rowCount), groups = UInt32(rowGroups)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(multiRowGreedy)
            encoder.setBuffer(xNormedRowsBuffer, offset: 0, index: 0)
            encoder.setBuffer(weights, offset: weightsOffset, index: 1)
            encoder.setBuffer(scales, offset: scalesOffset, index: 2)
            encoder.setBuffer(biases, offset: biasesOffset, index: 3)
            encoder.setBuffer(multiRowSummariesBuffer, offset: 0, index: 4)
            encoder.setBytes(&dValue, length: 4, index: 5)
            encoder.setBytes(&vocabValue, length: 4, index: 6)
            encoder.setBytes(&rows, length: 4, index: 7)
            encoder.dispatchThreadgroups(
                MTLSize(width: rowGroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32 * Self.multiRowVocabRowsPerThreadgroup, height: 1, depth: 1))
            encoder.endEncoding()
        }
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(multiRowReducer)
            encoder.setBuffer(multiRowSummariesBuffer, offset: 0, index: 0)
            encoder.setBuffer(outTokens, offset: outTokensOffset, index: 1)
            encoder.setBytes(&groups, length: 4, index: 2)
            encoder.setBytes(&rows, length: 4, index: 3)
            encoder.dispatchThreadgroups(
                MTLSize(width: rowCount, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
        }
    }

    func encodeGreedyDecode(commandBuffer: MTLCommandBuffer,
                            hidden: MTLBuffer,
                            hiddenOffset: Int = 0,
                            normWeight: MTLBuffer,
                            normOffset: Int = 0,
                            weights: MTLBuffer,
                            weightsOffset: Int = 0,
                            scales: MTLBuffer,
                            scalesOffset: Int = 0,
                            biases: MTLBuffer,
                            biasesOffset: Int = 0,
                            outToken: MTLBuffer,
                            outTokenOffset: Int = 0,
                            d: UInt32,
                            vocab: UInt32,
                            rmsEps: Float = 1e-6) {
        precondition(Int(d) <= maxD, "d=\(d) exceeds wrapper maxD=\(maxD)")
        precondition(Int(vocab) <= maxVocab,
                     "vocab=\(vocab) exceeds wrapper maxVocab=\(maxVocab)")
        precondition(Int(d) % Quantization.groupSize == 0,
                     "d must be a multiple of \(Quantization.groupSize)")
        precondition(hiddenOffset >= 0, "hiddenOffset must be non-negative")
        precondition(weightsOffset % 2 == 0,
                     "lm_head_greedy_int4_rows_chunk_raw needs a 2-aligned weightsOffset")

        let rowGroups = (Int(vocab) + Self.rowsPerThreadgroup - 1)
            / Self.rowsPerThreadgroup
        rms.encodeBF16W(commandBuffer: commandBuffer,
                        x: hidden,
                        xOffset: hiddenOffset,
                        weight: normWeight,
                        weightOffset: normOffset,
                        out: xNormedBuffer,
                        d: d,
                        eps: rmsEps)

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            let specialized = d == Self.realDecodeD && vocab == Self.realDecodeVocab
            encoder.setComputePipelineState(specialized ? rowGreedySpecialized : rowGreedy)
            encoder.setBuffer(xNormedBuffer, offset: 0, index: 0)
            encoder.setBuffer(weights, offset: weightsOffset, index: 1)
            encoder.setBuffer(scales, offset: scalesOffset, index: 2)
            encoder.setBuffer(biases, offset: biasesOffset, index: 3)
            encoder.setBuffer(rowSummariesBuffer, offset: 0, index: 4)
            var dValue = d
            var vocabValue = vocab
            encoder.setBytes(&dValue, length: MemoryLayout<UInt32>.size, index: 5)
            encoder.setBytes(&vocabValue, length: MemoryLayout<UInt32>.size, index: 6)

            let threadgroupSize = MTLSize(
                width: 32 * Self.rowsPerThreadgroup,
                height: 1,
                depth: 1)
            encoder.dispatchThreadgroups(
                MTLSize(width: rowGroups, height: 1, depth: 1),
                threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(rowReducer)
            encoder.setBuffer(rowSummariesBuffer, offset: 0, index: 0)
            encoder.setBuffer(outToken, offset: outTokenOffset, index: 1)
            var rowGroupCount = UInt32(rowGroups)
            encoder.setBytes(&rowGroupCount, length: MemoryLayout<UInt32>.size, index: 2)

            let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
            encoder.dispatchThreads(threadgroupSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }
    }
}
