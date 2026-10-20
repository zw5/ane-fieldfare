import Metal

/// Final BF16 RMSNorm, INT4 affine lm-head projection, and greedy argmax.
/// The hot path writes one token ID without materializing vocab-sized logits.
final class LMHeadChainInt4 {
    static let rowsPerThreadgroup = 8
    /// Hidden rows one `encodeGreedyRows` call can score; matches the kernel.
    static let maxGreedyRows = 16

    private static let rowSummaryStride = 2
    static let realDecodeD: UInt32 = 2816
    private static let realDecodeVocab: UInt32 = 262144
    private static let realDecodeHeadConstants: [MetalFunctionConstant] = [
        MetalFunctionConstant(index: 10, value: .uint32(realDecodeD)),
        MetalFunctionConstant(index: 11, value: .uint32(realDecodeVocab)),
    ]

    /// Rows one `encodeGreedyCheck` call checks; matches the kernel.
    static let maxCheckRows = 4
    /// Weight rows per SIMD group in the check kernel; matches it.
    private static let checkRowsPerSimd = 2

    private let rms: RMSNorm
    private let rowGreedy: any MTLComputePipelineState
    private let rowGreedySpecialized: any MTLComputePipelineState
    /// The single-row head over a `HotVocabulary` list, D fixed.
    private let hotGreedy: any MTLComputePipelineState
    /// `lm_head_greedy_int4_check_chunk_raw` by row count, specialized to the
    /// real head's shape (index 0 is 1 row).
    private let checkSpecialized: [any MTLComputePipelineState]
    private let checkGeneric: [any MTLComputePipelineState]
    private let checkSummariesBuffer: any MTLBuffer
    private let rowReducer: any MTLComputePipelineState
    private let rowsProjection: DequantInt4GEMV
    private let rowsArgmax: any MTLComputePipelineState
    private let rowsSample: any MTLComputePipelineState
    /// A zero `UInt32`, the key offset of a draw whose positions are known.
    private let zeroKeyOffset: any MTLBuffer
    private let xNormedRowsBuffer: any MTLBuffer
    private let rowsLogitsBuffer: any MTLBuffer
    /// Per-row greedy-token probability from the last `encodeGreedyRows`.
    let rowsConfidence: any MTLBuffer
    private let xNormedBuffer: any MTLBuffer
    private let rowSummariesBuffer: any MTLBuffer
    private let maxD: Int
    private let maxVocab: Int

    init(context: MetalContext, maxD: Int, maxVocab: Int) throws {
        self.rms = try RMSNorm(context: context)
        self.rowGreedy = try context.pipeline("lm_head_greedy_int4_rows_chunk_raw")
        self.rowGreedySpecialized = try context.pipeline(
            "lm_head_greedy_int4_rows_chunk_raw",
            constants: Self.realDecodeHeadConstants)
        self.rowReducer = try context.pipeline("lm_head_greedy_int4_rows_reduce")
        self.hotGreedy = try context.pipeline(
            "lm_head_greedy_int4_listed_chunk_raw",
            constants: [MetalFunctionConstant(index: 10, value: .uint32(Self.realDecodeD))])
        self.checkSpecialized = try (1...Self.maxCheckRows).map {
            try context.pipeline("lm_head_greedy_int4_check_chunk_raw",
                                 constants: Self.realDecodeHeadConstants
                                     + [MetalFunctionConstant(index: 16, value: .uint32(UInt32($0)))])
        }
        self.checkGeneric = try (1...Self.maxCheckRows).map {
            try context.pipeline("lm_head_greedy_int4_check_chunk_raw",
                                 constants: [MetalFunctionConstant(index: 16, value: .uint32(UInt32($0)))])
        }
        self.rowsProjection = try DequantInt4GEMV(context: context)
        self.rowsArgmax = try context.pipeline("argmax_rows_f32")
        self.rowsSample = try context.pipeline("sample_rows_f32")
        let zero = try context.device.newBuffer(length: 16, options: .storageModeShared)
        unsafe memset(zero.contents(), 0, 16)
        self.zeroKeyOffset = zero
        self.maxD = maxD
        self.maxVocab = maxVocab

        let rowGroups = (maxVocab + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup
        let xLength = max(maxD, 1) * MemoryLayout<Float16>.size
        let summaryLength = rowGroups * Self.rowSummaryStride * MemoryLayout<Float>.size
        self.xNormedBuffer = try context.device.newBuffer(length: xLength, options: .storageModePrivate)
        self.rowSummariesBuffer = try context.device.newBuffer(length: summaryLength, options: .storageModePrivate)
        self.checkSummariesBuffer = try context.device.newBuffer(length: Self.maxCheckRows * summaryLength,
                                                                 options: .storageModePrivate)
        self.xNormedRowsBuffer = try context.device.newBuffer(length: Self.maxGreedyRows * xLength,
                                                              options: .storageModePrivate)
        self.rowsLogitsBuffer = try context.device.newBuffer(
            length: Self.maxGreedyRows * maxVocab * MemoryLayout<Float>.size, options: .storageModePrivate)
        self.rowsConfidence = try context.device.newBuffer(length: Self.maxGreedyRows * MemoryLayout<Float>.size,
                                                           options: .storageModeShared)
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
                          softcap: Float = 0,
                          sampling: RowSampling? = nil) throws {
        precondition(rowCount > 0 && rowCount <= Self.maxGreedyRows,
                     "rowCount=\(rowCount) outside 1...\(Self.maxGreedyRows)")
        precondition(Int(d) <= maxD && Int(vocab) <= maxVocab)
        precondition(Int(d) % 256 == 0, "d must be a multiple of 256")
        let rowBytes = Int(d) * MemoryLayout<Float16>.stride
        for row in 0..<rowCount {
            try rms.encodeBF16W(commandBuffer: commandBuffer,
                                x: hidden, xOffset: hiddenOffset + row * rowBytes,
                                weight: normWeight, weightOffset: normOffset,
                                out: xNormedRowsBuffer, outOffset: row * rowBytes,
                                d: d, eps: rmsEps)
        }
        // FP32 logits from the multi-row GEMV, which streams the table once
        // and keeps every row's accumulators in registers, then an argmax per
        // row.
        try rowsProjection.encodeRows(commandBuffer: commandBuffer,
                                  weights: weights, weightsOffset: weightsOffset,
                                  scales: scales, scalesOffset: scalesOffset,
                                  biases: biases, biasesOffset: biasesOffset,
                                  x: xNormedRowsBuffer, y: rowsLogitsBuffer,
                                  m: vocab, n: d, rows: rowCount,
                                  xStrideElements: Int(d), yStrideElements: Int(vocab),
                                  types: .f16ToF32)
        try encodeChoice(commandBuffer: commandBuffer, rowCount: rowCount, count: vocab, ids: nil,
                         outTokens: outTokens, outTokensOffset: outTokensOffset, softcap: softcap,
                         sampling: sampling)
    }

    /// `encodeGreedyRows` over `table`'s rows only: each row's greedy token
    /// among them, and its probability among them in `rowsConfidence`. An
    /// approximation for a drafter, which verify corrects.
    func encodeHotGreedyRows(commandBuffer: any MTLCommandBuffer,
                             hidden: any MTLBuffer,
                             rowCount: Int,
                             normWeight: any MTLBuffer,
                             normOffset: Int = 0,
                             table: HotVocabulary,
                             weights: any MTLBuffer,
                             weightsOffset: Int = 0,
                             scales: any MTLBuffer,
                             scalesOffset: Int = 0,
                             biases: any MTLBuffer,
                             biasesOffset: Int = 0,
                             outIndices: any MTLBuffer,
                             d: UInt32,
                             rmsEps: Float = 1e-6,
                             softcap: Float = 0,
                             sampling: RowSampling? = nil) throws {
        precondition(rowCount > 0 && rowCount <= Self.maxGreedyRows && table.count > 0)
        precondition(Int(d) <= maxD && Int(d) % 256 == 0)
        let rowBytes = Int(d) * MemoryLayout<Float16>.stride
        for row in 0..<rowCount {
            try rms.encodeBF16W(commandBuffer: commandBuffer,
                                x: hidden, xOffset: row * rowBytes,
                                weight: normWeight, weightOffset: normOffset,
                                out: xNormedRowsBuffer, outOffset: row * rowBytes,
                                d: d, eps: rmsEps)
        }
        try rowsProjection.encodeRowsListed(commandBuffer: commandBuffer,
                                            weights: weights, weightsOffset: weightsOffset,
                                            scales: scales, scalesOffset: scalesOffset,
                                            biases: biases, biasesOffset: biasesOffset,
                                            ids: table.ids, x: xNormedRowsBuffer, y: rowsLogitsBuffer,
                                            m: UInt32(table.count), n: d, rows: rowCount,
                                            xStrideElements: Int(d), yStrideElements: table.count)
        try encodeChoice(commandBuffer: commandBuffer, rowCount: rowCount, count: UInt32(table.count),
                         ids: table.ids, outTokens: outIndices, outTokensOffset: 0, softcap: softcap,
                         sampling: sampling)
    }

    /// How a row of the FP32 logits picks its token: the argmax, or a draw
    /// under `rule` (`sample_rows_f32`) for the token at position `keyBase`
    /// + row, plus the `UInt32` at `keyOffset` when the base lives on the GPU.
    struct RowSampling {
        let rule: TokenSampling
        let keyBase: Int
        var keyOffset: (any MTLBuffer)?
    }

    private func encodeChoice(commandBuffer: any MTLCommandBuffer, rowCount: Int, count: UInt32,
                              ids: (any MTLBuffer)?, outTokens: any MTLBuffer, outTokensOffset: Int,
                              softcap: Float, sampling: RowSampling?) throws {
        let encoder = try commandBuffer.makeTracedComputeEncoder()
        encoder.setBuffer(rowsLogitsBuffer, offset: 0, index: 0)
        encoder.setBuffer(outTokens, offset: outTokensOffset, index: 1)
        if let sampling, sampling.rule.temperature > 0 {
            struct Params {
                var temperature: Float, topP: Float, topK: UInt32, count: UInt32, keyBase: UInt32
                var listed: UInt32, softcap: Float, seedLo: UInt32, seedHi: UInt32
            }
            let rule = sampling.rule
            let params = Params(temperature: rule.temperature, topP: rule.topP, topK: UInt32(rule.topK ?? 0),
                                count: count, keyBase: UInt32(sampling.keyBase), listed: ids == nil ? 0 : 1,
                                softcap: softcap, seedLo: UInt32(truncatingIfNeeded: rule.seed),
                                seedHi: UInt32(truncatingIfNeeded: rule.seed >> 32))
            encoder.setComputePipelineState(rowsSample)
            encoder.setBuffer(rowsConfidence, offset: 0, index: 2)
            encoder.setValue(params, index: 3)
            encoder.setBuffer(ids ?? zeroKeyOffset, offset: 0, index: 4)
            encoder.setBuffer(sampling.keyOffset ?? zeroKeyOffset, offset: 0, index: 5)
        } else {
            encoder.setComputePipelineState(rowsArgmax)
            encoder.setValue(count, index: 2)
            encoder.setBuffer(rowsConfidence, offset: 0, index: 3)
            encoder.setValue(softcap, index: 4)
        }
        encoder.dispatchThreadgroups(
            MTLSize(width: rowCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
        encoder.endEncoding()
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
                            rmsEps: Float = 1e-6) throws {
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
        try rms.encodeBF16W(commandBuffer: commandBuffer,
                            x: hidden,
                            xOffset: hiddenOffset,
                            weight: normWeight,
                            weightOffset: normOffset,
                            out: xNormedBuffer,
                            d: d,
                            eps: rmsEps)

        let rowsEncoder = try commandBuffer.makeTracedComputeEncoder()
        let specialized = d == Self.realDecodeD && vocab == Self.realDecodeVocab
        rowsEncoder.setComputePipelineState(specialized ? rowGreedySpecialized : rowGreedy)
        rowsEncoder.setBuffer(xNormedBuffer, offset: 0, index: 0)
        rowsEncoder.setBuffer(weights, offset: weightsOffset, index: 1)
        rowsEncoder.setBuffer(scales, offset: scalesOffset, index: 2)
        rowsEncoder.setBuffer(biases, offset: biasesOffset, index: 3)
        rowsEncoder.setBuffer(rowSummariesBuffer, offset: 0, index: 4)
        rowsEncoder.setValue(d, index: 5)
        rowsEncoder.setValue(vocab, index: 6)
        rowsEncoder.dispatchThreadgroups(
            MTLSize(width: rowGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * Self.rowsPerThreadgroup, height: 1, depth: 1))
        rowsEncoder.endEncoding()

        let reduceEncoder = try commandBuffer.makeTracedComputeEncoder()
        reduceEncoder.setComputePipelineState(rowReducer)
        reduceEncoder.setBuffer(rowSummariesBuffer, offset: 0, index: 0)
        reduceEncoder.setBuffer(outToken, offset: outTokenOffset, index: 1)
        reduceEncoder.setValue(UInt32(rowGroups), index: 2)
        let reduceSize = MTLSize(width: 256, height: 1, depth: 1)
        reduceEncoder.dispatchThreads(reduceSize, threadsPerThreadgroup: reduceSize)
        reduceEncoder.endEncoding()
    }

    /// Normalizes `hidden` into row `row` of `normedRows` and writes the
    /// greedy token among `table`'s rows of the head to `outToken`. Each row's
    /// logit is computed as `encodeGreedyDecode` computes it. The token is a
    /// guess until `encodeGreedyCheck` confirms it on the saved row.
    func encodeHotGreedyDecode(commandBuffer: any MTLCommandBuffer,
                               hidden: any MTLBuffer,
                               hiddenOffset: Int = 0,
                               normWeight: any MTLBuffer,
                               normOffset: Int = 0,
                               table: HotVocabulary,
                               weights: any MTLBuffer,
                               weightsOffset: Int = 0,
                               scales: any MTLBuffer,
                               scalesOffset: Int = 0,
                               biases: any MTLBuffer,
                               biasesOffset: Int = 0,
                               normedRows: any MTLBuffer,
                               row: Int,
                               outToken: any MTLBuffer,
                               d: UInt32,
                               rmsEps: Float = 1e-6) throws {
        precondition(d == Self.realDecodeD, "the hot head is specialized to D=\(Self.realDecodeD)")
        precondition(table.count > 0 && weightsOffset % 2 == 0)
        try encodeNormRow(commandBuffer: commandBuffer, hidden: hidden, hiddenOffset: hiddenOffset,
                          normWeight: normWeight, normOffset: normOffset, normedRows: normedRows, row: row,
                          d: d, rmsEps: rmsEps)
        let rowOffset = row * Int(d) * MemoryLayout<Float16>.stride
        let rowGroups = (table.count + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup
        let encoder = try commandBuffer.makeTracedComputeEncoder()
        encoder.setComputePipelineState(hotGreedy)
        encoder.setBuffer(normedRows, offset: rowOffset, index: 0)
        encoder.setBuffer(weights, offset: weightsOffset, index: 1)
        encoder.setBuffer(scales, offset: scalesOffset, index: 2)
        encoder.setBuffer(biases, offset: biasesOffset, index: 3)
        encoder.setBuffer(rowSummariesBuffer, offset: 0, index: 4)
        let count = UInt32(table.count)
        encoder.setValue(d, index: 5)
        encoder.setValue(count, index: 6)
        encoder.setBuffer(table.ids, offset: 0, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: rowGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * Self.rowsPerThreadgroup, height: 1, depth: 1))
        encoder.setComputePipelineState(rowReducer)
        encoder.setBuffer(rowSummariesBuffer, offset: 0, index: 0)
        encoder.setBuffer(outToken, offset: 0, index: 1)
        let groups = UInt32(rowGroups)
        encoder.setValue(groups, index: 2)
        encoder.dispatchThreads(MTLSize(width: 256, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// The head's final norm of `hidden` into row `row` of `normedRows`.
    func encodeNormRow(commandBuffer: any MTLCommandBuffer, hidden: any MTLBuffer, hiddenOffset: Int = 0,
                       normWeight: any MTLBuffer, normOffset: Int = 0, normedRows: any MTLBuffer, row: Int,
                       d: UInt32, rmsEps: Float = 1e-6) throws {
        precondition(row >= 0 && row < Self.maxCheckRows)
        try rms.encodeBF16W(commandBuffer: commandBuffer, x: hidden, xOffset: hiddenOffset,
                            weight: normWeight, weightOffset: normOffset,
                            out: normedRows, outOffset: row * Int(d) * MemoryLayout<Float16>.stride,
                            d: d, eps: rmsEps)
    }

    /// The full head's greedy token for each of `rowCount` already
    /// normalized rows, written as `UInt32` to `outTokens`. Reads the INT4
    /// table once; row r's token is bitwise `encodeGreedyDecode`'s for the
    /// hidden row it was normalized from.
    func encodeGreedyCheck(commandBuffer: any MTLCommandBuffer,
                           normedRows: any MTLBuffer,
                           rowCount: Int,
                           weights: any MTLBuffer,
                           weightsOffset: Int = 0,
                           scales: any MTLBuffer,
                           scalesOffset: Int = 0,
                           biases: any MTLBuffer,
                           biasesOffset: Int = 0,
                           outTokens: any MTLBuffer,
                           d: UInt32,
                           vocab: UInt32) throws {
        precondition(rowCount > 0 && rowCount <= Self.maxCheckRows)
        precondition(Int(d) <= maxD && Int(vocab) <= maxVocab && Int(d) % 256 == 0)
        precondition(weightsOffset % 2 == 0)
        let rowsPerGroup = Self.rowsPerThreadgroup * Self.checkRowsPerSimd
        let rowGroups = (Int(vocab) + rowsPerGroup - 1) / rowsPerGroup
        let specialized = d == Self.realDecodeD && vocab == Self.realDecodeVocab
        let encoder = try commandBuffer.makeTracedComputeEncoder()
        encoder.setComputePipelineState((specialized ? checkSpecialized : checkGeneric)[rowCount - 1])
        encoder.setBuffer(normedRows, offset: 0, index: 0)
        encoder.setBuffer(weights, offset: weightsOffset, index: 1)
        encoder.setBuffer(scales, offset: scalesOffset, index: 2)
        encoder.setBuffer(biases, offset: biasesOffset, index: 3)
        encoder.setBuffer(checkSummariesBuffer, offset: 0, index: 4)
        encoder.setValue(d, index: 5)
        encoder.setValue(vocab, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: rowGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * Self.rowsPerThreadgroup, height: 1, depth: 1))
        encoder.setComputePipelineState(rowReducer)
        let groups = UInt32(rowGroups)
        encoder.setValue(groups, index: 2)
        for row in 0..<rowCount {
            encoder.setBuffer(checkSummariesBuffer,
                              offset: row * rowGroups * Self.rowSummaryStride * MemoryLayout<Float>.size,
                              index: 0)
            encoder.setBuffer(outTokens, offset: row * MemoryLayout<UInt32>.size, index: 1)
            encoder.dispatchThreads(MTLSize(width: 256, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        encoder.endEncoding()
    }
}
