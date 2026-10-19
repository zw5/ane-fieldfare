import Foundation
import Metal

final class PrefillSharedExpert {
    /// Chunks at least this long use batched tensor-op GEMMs; shorter ones
    /// keep the per-token GEMV loop.
    static let batchedMinimumTokens = 32

    private let shared: SharedExpertRuntime
    private let context: MetalContext
    private let batched: MPPPrefillInt4QMM?
    private let geluMulPSO: MTLComputePipelineState?
    /// Multi-row GEMV for small INT4 batches such as speculative verification.
    private let multiRow: DequantInt4GEMV?
    private var batchScratch: (gate: MTLBuffer, up: MTLBuffer, act: MTLBuffer, tokens: Int, intermediate: Int)?

    /// `batched` selects the INT4 tensor-op GEMM path for 4-bit weights. It
    /// changes accumulation order relative to the per-token GEMV reference.
    init(context: MetalContext, weightBits: Int = 8, batched: Bool = false) throws {
        self.shared = try SharedExpertRuntime(context: context, weightBits: weightBits)
        self.context = context
        if batched && weightBits == 4 {
            let qmm = MPPPrefillInt4QMM(context: context)
            self.batched = qmm.isAvailable ? qmm : nil
            self.geluMulPSO = try context.pipeline("gelu_mul_fp16")
            self.multiRow = try DequantInt4GEMV(context: context)
        } else {
            self.batched = nil
            self.geluMulPSO = nil
            self.multiRow = nil
        }
    }

    var usesBatchedGEMM: Bool { batched != nil }

    private func scratch(tokens: Int, intermediate: Int) throws -> (gate: MTLBuffer, up: MTLBuffer, act: MTLBuffer) {
        if let existing = batchScratch, existing.tokens >= tokens, existing.intermediate == intermediate {
            return (existing.gate, existing.up, existing.act)
        }
        let bytes = tokens * intermediate * MemoryLayout<Float16>.stride
        func make(_ label: String) throws -> MTLBuffer {
            guard let buffer = context.device.makeBuffer(length: bytes, options: .storageModePrivate) else {
                throw SharedExpertError.scratchTooSmall("cannot allocate \(bytes) bytes for \(label)")
            }
            buffer.label = label
            return buffer
        }
        let made = (gate: try make("prefill.sharedBatchGate"), up: try make("prefill.sharedBatchUp"),
                    act: try make("prefill.sharedBatchAct"))
        batchScratch = (made.gate, made.up, made.act, tokens, intermediate)
        return made
    }

    /// Returns false when the batched path does not apply and the caller's
    /// per-token loop must run.
    private func encodeBatched(commandBuffer cb: MTLCommandBuffer, x: MTLBuffer, xOffset: Int,
                               y: MTLBuffer, yOffset: Int,
                               gate: SharedExpertInt8Proj, up: SharedExpertInt8Proj, down: SharedExpertInt8Proj,
                               queryCount: Int, d: Int, intermediate: Int,
                               xStrideElements: Int, yStrideElements: Int,
                               smallBatch: Bool) throws -> Bool {
        let useMultiRow = smallBatch && queryCount <= DequantInt4GEMV.maxBatchRows && multiRow != nil
        guard let batched, let geluMulPSO, useMultiRow || queryCount >= Self.batchedMinimumTokens,
              xStrideElements == d, yStrideElements == d else { return false }
        let buffers = try scratch(tokens: queryCount, intermediate: intermediate)
        func project(_ p: SharedExpertInt8Proj, x: MTLBuffer, xOffset: Int, y: MTLBuffer, yOffset: Int) throws -> Bool {
            if useMultiRow, let multiRow {
                try multiRow.encodeRows(commandBuffer: cb, weights: p.weights, weightsOffset: p.weightsOffset,
                                    scales: p.scales, scalesOffset: p.scalesOffset,
                                    biases: p.biases, biasesOffset: p.biasesOffset,
                                    x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                                    m: p.rows, n: p.cols, rows: queryCount,
                                    xStrideElements: Int(p.cols), yStrideElements: Int(p.rows))
                return true
            }
            return batched.encode(commandBuffer: cb, weights: p.weights, weightsOffset: p.weightsOffset,
                           scales: p.scales, scalesOffset: p.scalesOffset,
                           biases: p.biases, biasesOffset: p.biasesOffset,
                           x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                           m: queryCount, n: Int(p.rows), k: Int(p.cols)).path != .fallback
        }
        // A fallback on the first projection leaves nothing encoded; any
        // later one is a contract violation for these fixed shapes.
        guard try project(gate, x: x, xOffset: xOffset, y: buffers.gate, yOffset: 0) else { return false }
        guard try project(up, x: x, xOffset: xOffset, y: buffers.up, yOffset: 0),
              let encoder = cb.makeComputeCommandEncoder() else {
            throw SharedExpertError.dimensionMismatch("batched shared expert up projection fell back")
        }
        encoder.setComputePipelineState(geluMulPSO)
        encoder.setBuffer(buffers.gate, offset: 0, index: 0)
        encoder.setBuffer(buffers.up, offset: 0, index: 1)
        encoder.setBuffer(buffers.act, offset: 0, index: 2)
        var count = UInt32(queryCount * intermediate)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreads(MTLSize(width: Int(count), height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(geluMulPSO.maxTotalThreadsPerThreadgroup, 256),
                                                               height: 1, depth: 1))
        encoder.endEncoding()
        guard try project(down, x: buffers.act, xOffset: 0, y: y, yOffset: yOffset) else {
            throw SharedExpertError.dimensionMismatch("batched shared expert down projection fell back")
        }
        return true
    }

    func encodeBlock(commandBuffer cb: MTLCommandBuffer,
                            x: MTLBuffer,
                            xOffset: Int = 0,
                            y: MTLBuffer,
                            yOffset: Int = 0,
                            gate: SharedExpertInt8Proj,
                            up: SharedExpertInt8Proj,
                            down: SharedExpertInt8Proj,
                            scratchGate: MTLBuffer,
                            scratchGateOffset: Int = 0,
                            scratchUp: MTLBuffer,
                            scratchUpOffset: Int = 0,
                            scratchAct: MTLBuffer,
                            scratchActOffset: Int = 0,
                            queryCount: Int,
                            d: Int,
                            intermediate: Int,
                            xStrideElements: Int,
                            yStrideElements: Int,
                            smallBatch: Bool = false) throws {
        precondition(queryCount >= 0, "queryCount must be non-negative")
        precondition(d > 0, "d must be positive")
        precondition(intermediate > 0, "intermediate must be positive")
        precondition(xStrideElements >= d, "x stride is too small")
        precondition(yStrideElements >= d, "y stride is too small")
        guard gate.rows == UInt32(intermediate), gate.cols == UInt32(d),
              up.rows == UInt32(intermediate), up.cols == UInt32(d),
              down.rows == UInt32(d), down.cols == UInt32(intermediate) else {
            throw SharedExpertInt8Error.dimensionMismatch(
                "expected gate/up=(\(intermediate),\(d)) down=(\(d),\(intermediate))")
        }

        if try encodeBatched(commandBuffer: cb, x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                             gate: gate, up: up, down: down, queryCount: queryCount, d: d,
                             intermediate: intermediate, xStrideElements: xStrideElements,
                             yStrideElements: yStrideElements, smallBatch: smallBatch) {
            return
        }
        let halfBytes = MemoryLayout<Float16>.stride
        for row in 0..<queryCount {
            try shared.encode(commandBuffer: cb,
                              x: x,
                              xOffset: xOffset + row * xStrideElements * halfBytes,
                              gate: gate,
                              up: up,
                              down: down,
                              y: y,
                              yOffset: yOffset + row * yStrideElements * halfBytes,
                              scratchGate: scratchGate,
                              scratchGateOffset: scratchGateOffset,
                              scratchUp: scratchUp,
                              scratchUpOffset: scratchUpOffset,
                              scratchAct: scratchAct,
                              scratchActOffset: scratchActOffset)
        }
    }
}
