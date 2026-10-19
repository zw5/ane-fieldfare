import Metal

/// Final BF16 RMSNorm, INT4 affine lm-head projection, and greedy argmax.
/// The hot path writes one token ID without materializing vocab-sized logits.
final class LMHeadChainInt4 {
    static let rowsPerThreadgroup = 8
    /// Hidden rows one `encodeGreedyRows` call can score; matches the kernel.
    static let maxGreedyRows = 16

    private static let rowSummaryStride = 2
    private static let realDecodeD: UInt32 = 2816
    private static let realDecodeVocab: UInt32 = 262144
    private static let realDecodeHeadConstants: [MetalFunctionConstant] = [
        MetalFunctionConstant(index: 10, value: .uint32(realDecodeD)),
        MetalFunctionConstant(index: 11, value: .uint32(realDecodeVocab)),
        MetalFunctionConstant(index: 13, value: .bool(true)),
    ]

    private let rms: RMSNorm
    private let rowGreedy: any MTLComputePipelineState
    private let rowGreedySpecialized: any MTLComputePipelineState
    private let rowReducer: any MTLComputePipelineState
    private let rowsProjection: DequantInt4GEMV
    private let rowsArgmax: any MTLComputePipelineState
    private let xNormedRowsBuffer: any MTLBuffer
    private let rowsLogitsBuffer: any MTLBuffer
    /// Per-row greedy-token probability from the last `encodeGreedyRows`.
    let rowsConfidence: any MTLBuffer
    private let xNormedBuffer: any MTLBuffer
    private let rowSummariesBuffer: any MTLBuffer
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
        self.rowsProjection = try DequantInt4GEMV(context: context)
        self.rowsArgmax = try context.pipeline("argmax_rows_f32")
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
              let rowsLogits = context.device.makeBuffer(
                  length: Self.maxGreedyRows * maxVocab * MemoryLayout<Float>.size,
                  options: .storageModePrivate) else {
            throw MetalError.noDevice
        }
        self.xNormedRowsBuffer = xNormedRows
        self.rowsLogitsBuffer = rowsLogits
        guard let confidence = context.device.makeBuffer(
                  length: Self.maxGreedyRows * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        self.rowsConfidence = confidence
    }

    /// Greedy tokens for `rowCount` consecutive FP16 hidden rows starting at
    /// `hiddenOffset`, written as `UInt32` to `outTokens`. Reads the INT4 table
    /// once for all rows.
    func encodeGreedyRows(commandBuffer: any MTLCommandBuffer,
                          hidden: any MTLBuffer,
                          hiddenOffset: Int = 0,
                          rowCount: Int,
                          normWeight: any MTLBuffer,
                          normOffset: Int = 0,
                          weights: any MTLBuffer,
                          weightsOffset: Int = 0,
                          scales: any MTLBuffer,
                          scalesOffset: Int = 0,
                          biases: any MTLBuffer,
                          biasesOffset: Int = 0,
                          outTokens: any MTLBuffer,
                          outTokensOffset: Int = 0,
                          d: UInt32,
                          vocab: UInt32,
                          rmsEps: Float = 1e-6,
                          softcap: Float = 0) throws {
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
        // FP32 logits from the multi-row GEMV, which streams the table once
        // and keeps every row's accumulators in registers, then an argmax per
        // row. The fused tiled kernel this replaced cost ~3 ms per row.
        try rowsProjection.encodeRows(commandBuffer: commandBuffer,
                                  weights: weights, weightsOffset: weightsOffset,
                                  scales: scales, scalesOffset: scalesOffset,
                                  biases: biases, biasesOffset: biasesOffset,
                                  x: xNormedRowsBuffer, y: rowsLogitsBuffer,
                                  m: vocab, n: d, rows: rowCount,
                                  xStrideElements: Int(d), yStrideElements: Int(vocab),
                                  types: .f16ToF32)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            let vocabValue = vocab
            encoder.setComputePipelineState(rowsArgmax)
            encoder.setBuffer(rowsLogitsBuffer, offset: 0, index: 0)
            encoder.setBuffer(outTokens, offset: outTokensOffset, index: 1)
            encoder.setValue(vocabValue, index: 2)
            encoder.setBuffer(rowsConfidence, offset: 0, index: 3)
            let cap = softcap
            encoder.setValue(cap, index: 4)
            encoder.dispatchThreadgroups(
                MTLSize(width: rowCount, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
            encoder.endEncoding()
        }
    }

    func encodeGreedyDecode(commandBuffer: any MTLCommandBuffer,
                            hidden: any MTLBuffer,
                            hiddenOffset: Int = 0,
                            normWeight: any MTLBuffer,
                            normOffset: Int = 0,
                            weights: any MTLBuffer,
                            weightsOffset: Int = 0,
                            scales: any MTLBuffer,
                            scalesOffset: Int = 0,
                            biases: any MTLBuffer,
                            biasesOffset: Int = 0,
                            outToken: any MTLBuffer,
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
            let dValue = d
            let vocabValue = vocab
            encoder.setValue(dValue, index: 5)
            encoder.setValue(vocabValue, index: 6)

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
            let rowGroupCount = UInt32(rowGroups)
            encoder.setValue(rowGroupCount, index: 2)

            let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
            encoder.dispatchThreads(threadgroupSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }
    }
}
