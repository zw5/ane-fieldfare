import Metal

/// MLX-affine INT4 matrix-vector multiplication.
/// Eight SIMD groups process eight output rows per threadgroup.
final class DequantInt4GEMV {
    private struct Shape: Hashable {
        var m: UInt32
        var n: UInt32
    }

    private static let rowsPerThreadgroup = 8
    private static let realDecodeShapes: [Shape] = [
        Shape(m: 4096, n: 2816),
        Shape(m: 2048, n: 2816),
        Shape(m: 2816, n: 4096),
        Shape(m: 8192, n: 2816),
        Shape(m: 1024, n: 2816),
        Shape(m: 2816, n: 8192),
        // Gemma 4 26B-A4B shared FFN gate/up and down.
        Shape(m: 2112, n: 2816),
        Shape(m: 2816, n: 2112),
    ]

    /// Input rows one `encodeRows` dispatch handles; matches the kernel.
    static let maxBatchRows = 16

    private let pipeline: any MTLComputePipelineState
    private let context: MetalContext
    private let specializedPipelines: [Shape: any MTLComputePipelineState]

    init(context: MetalContext) throws {
        self.pipeline = try context.pipeline(
            "dequant_int4_gemv_simd",
            constants: [],
            maxTotalThreadsPerThreadgroup: 512)

        var specializedPipelines: [Shape: any MTLComputePipelineState] = [:]
        for shape in Self.realDecodeShapes {
            specializedPipelines[shape] = try context.pipeline(
                "dequant_int4_gemv_simd",
                constants: [
                    MetalFunctionConstant(index: 20, value: .uint32(shape.m)),
                    MetalFunctionConstant(index: 21, value: .uint32(shape.n)),
                    MetalFunctionConstant(index: 22, value: .bool(true)),
                ],
                maxTotalThreadsPerThreadgroup: 512)
        }
        self.specializedPipelines = specializedPipelines
        self.context = context
    }

    /// Activation and output element types for `encodeRows`.
    enum RowsTypes: String {
        case f16 = "dequant_int4_gemv_rows"
        case bf16 = "dequant_int4_gemv_rows_bf16"
        /// FP16 input, FP32 output, for logits.
        case f16ToF32 = "dequant_int4_gemv_rows_f32"
    }

    /// Weight rows per SIMD group for `rows` input rows. Two amortize the
    /// input traffic at 3–4 rows; past that the extra accumulators cost
    /// occupancy (measured: twice as slow at 8 rows). The ALU kernel only:
    /// 5+ rows normally take the matrix kernel.
    static func weightRowsPerSIMD(rows: Int) -> Int {
        (3...4).contains(rows) ? 2 : 1
    }

    /// `rows` input vectors at `xStrideElements` apart, each projected like
    /// `encode`, with the weights read once. Pipelines are specialized per row
    /// count and width. Below `matrixMinRows` rows, row r's result is bitwise
    /// the single-row GEMV's; from there the SIMD-group matrix kernel runs and
    /// matches it to FP32 rounding only.
    func encodeRows(commandBuffer: any MTLCommandBuffer,
                    weights: any MTLBuffer, weightsOffset: Int,
                    scales: any MTLBuffer, scalesOffset: Int,
                    biases: any MTLBuffer, biasesOffset: Int,
                    x: any MTLBuffer, xOffset: Int = 0,
                    y: any MTLBuffer, yOffset: Int = 0,
                    m: UInt32, n: UInt32, rows: Int,
                    xStrideElements: Int, yStrideElements: Int,
                    types: RowsTypes = .f16) throws {
        precondition(rows > 0 && rows <= Self.maxBatchRows, "rows=\(rows) outside 1...\(Self.maxBatchRows)")
        precondition(n % UInt32(Quantization.groupSize) == 0 && weightsOffset % 2 == 0)
        if rows >= Self.matrixMinRows {
            try encodeRowsMatrix(commandBuffer: commandBuffer,
                             weights: weights, weightsOffset: weightsOffset,
                             scales: scales, scalesOffset: scalesOffset,
                             biases: biases, biasesOffset: biasesOffset,
                             x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                             m: m, n: n, rows: rows,
                             xStrideElements: xStrideElements, yStrideElements: yStrideElements,
                             types: types)
            return
        }
        let weightRows = Self.weightRowsPerSIMD(rows: rows)
        let pipeline = try context.pipeline(
            types.rawValue,
            constants: [
                MetalFunctionConstant(index: 27, value: .uint32(UInt32(rows))),
                MetalFunctionConstant(index: 28, value: .uint32(n)),
                MetalFunctionConstant(index: 29, value: .bool(true)),
                MetalFunctionConstant(index: 32, value: .uint32(UInt32(weightRows))),
            ],
            maxTotalThreadsPerThreadgroup: 256)
        guard let encoder = commandBuffer.makeTracedComputeEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(y, offset: yOffset, index: 4)
        let values = [m, n, UInt32(rows), UInt32(xStrideElements), UInt32(yStrideElements)]
        for i in values.indices { encoder.setValue(values[i], index: 5 + i) }
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(m) + 8 * weightRows - 1) / (8 * weightRows), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// From this many input rows `encodeRows` uses the SIMD-group matrix
    /// kernel: the ALU kernel is bandwidth-bound through 4 rows and
    /// ALU-bound past them.
    static let matrixMinRows = 5

    /// `encodeRows` over the weight rows listed in `ids` (`m` of them): output
    /// o of each row is weight row `ids[o]`'s. Always the SIMD-group matrix
    /// kernel, FP32 out.
    func encodeRowsListed(commandBuffer: any MTLCommandBuffer,
                          weights: any MTLBuffer, weightsOffset: Int,
                          scales: any MTLBuffer, scalesOffset: Int,
                          biases: any MTLBuffer, biasesOffset: Int,
                          ids: any MTLBuffer, x: any MTLBuffer, y: any MTLBuffer,
                          m: UInt32, n: UInt32, rows: Int,
                          xStrideElements: Int, yStrideElements: Int) throws {
        precondition(rows > 0 && rows <= Self.maxBatchRows, "rows=\(rows) outside 1...\(Self.maxBatchRows)")
        precondition(n % UInt32(Quantization.groupSize) == 0 && weightsOffset % 2 == 0)
        try encodeRowsMatrix(commandBuffer: commandBuffer,
                             weights: weights, weightsOffset: weightsOffset,
                             scales: scales, scalesOffset: scalesOffset,
                             biases: biases, biasesOffset: biasesOffset,
                             x: x, xOffset: 0, y: y, yOffset: 0,
                             m: m, n: n, rows: rows,
                             xStrideElements: xStrideElements, yStrideElements: yStrideElements,
                             types: .f16ToF32, ids: ids)
    }

    private func encodeRowsMatrix(commandBuffer: any MTLCommandBuffer,
                                  weights: any MTLBuffer, weightsOffset: Int,
                                  scales: any MTLBuffer, scalesOffset: Int,
                                  biases: any MTLBuffer, biasesOffset: Int,
                                  x: any MTLBuffer, xOffset: Int, y: any MTLBuffer, yOffset: Int,
                                  m: UInt32, n: UInt32, rows: Int,
                                  xStrideElements: Int, yStrideElements: Int, types: RowsTypes,
                                  ids: (any MTLBuffer)? = nil) throws {
        let name = switch types {
        case .f16: "dequant_int4_gemv_rows_mma"
        case .f16ToF32: "dequant_int4_gemv_rows_mma_f32"
        case .bf16: "dequant_int4_gemv_rows_mma_bf16"
        }
        let pipeline = try context.pipeline(
            name,
            constants: [
                MetalFunctionConstant(index: 27, value: .uint32(UInt32(rows))),
                MetalFunctionConstant(index: 28, value: .uint32(n)),
                MetalFunctionConstant(index: 29, value: .bool(true)),
                MetalFunctionConstant(index: 33, value: .bool(ids != nil)),
            ],
            maxTotalThreadsPerThreadgroup: 128)
        guard let encoder = commandBuffer.makeTracedComputeEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        if let ids { encoder.setBuffer(ids, offset: 0, index: 10) }
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(y, offset: yOffset, index: 4)
        let values = [m, n, UInt32(rows), UInt32(xStrideElements), UInt32(yStrideElements)]
        for i in values.indices { encoder.setValue(values[i], index: 5 + i) }
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(m) + 31) / 32, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encode(commandBuffer: any MTLCommandBuffer,
                weights: any MTLBuffer,
                weightsOffset: Int = 0,
                scales: any MTLBuffer,
                scalesOffset: Int = 0,
                biases: any MTLBuffer,
                biasesOffset: Int = 0,
                x: any MTLBuffer,
                xOffset: Int = 0,
                y: any MTLBuffer,
                yOffset: Int = 0,
                m: UInt32,
                n: UInt32) {
        precondition(n % UInt32(Quantization.groupSize) == 0,
                     "N must be a multiple of \(Quantization.groupSize)")
        // The kernel reads packed weights through a `ushort*`; the repacker
        // guarantees two-byte sub-tensor alignment but not four-byte alignment.
        precondition(weightsOffset % 2 == 0,
                     "dequant_int4_gemv_simd needs a 2-aligned weightsOffset, got \(weightsOffset)")
        guard let encoder = commandBuffer.makeTracedComputeEncoder() else { return }
        encoder.setComputePipelineState(
            specializedPipelines[Shape(m: m, n: n)] ?? pipeline)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(y, offset: yOffset, index: 4)
        let mValue = m
        let nValue = n
        encoder.setValue(mValue, index: 5)
        encoder.setValue(nValue, index: 6)

        let threadgroupSize = MTLSize(
            width: 32 * Self.rowsPerThreadgroup,
            height: 1,
            depth: 1)
        let threadgroupCount = MTLSize(
            width: (Int(m) + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup,
            height: 1,
            depth: 1)
        encoder.dispatchThreadgroups(threadgroupCount,
                                     threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }
}
