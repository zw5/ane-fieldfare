import Foundation
public import Metal

/// One projection of the Gemma 4 shared (dense) MLP: MLX affine weights
/// (group=64, BF16 scale + BF16 bias), 8-bit or 4-bit. 26B-A4B shapes:
///   gate: [F=2112, D=2816]   up:   [F=2112, D=2816]   down: [D=2816, F=2112]
public struct SharedExpertProjection {
    public let weights: any MTLBuffer
    public let scales:  any MTLBuffer
    public let biases:  any MTLBuffer
    public let weightsOffset: Int
    public let scalesOffset:  Int
    public let biasesOffset:  Int
    public let rows: UInt32
    public let cols: UInt32

    public init(weights: any MTLBuffer, scales: any MTLBuffer, biases: any MTLBuffer,
                weightsOffset: Int = 0, scalesOffset: Int = 0, biasesOffset: Int = 0,
                rows: UInt32, cols: UInt32) {
        self.weights       = weights
        self.scales        = scales
        self.biases        = biases
        self.weightsOffset = weightsOffset
        self.scalesOffset  = scalesOffset
        self.biasesOffset  = biasesOffset
        self.rows          = rows
        self.cols          = cols
    }
}

/// Standalone 8-bit dense MLP — the "shared expert" branch of Gemma 4's
/// parallel MoE. Runs on the same hidden as the routed
/// branch, in parallel; their outputs are summed downstream.
///
///     y = down(gelu_pytorch_tanh(gate(x)) * up(x))
///
/// The gate, up, and activation work is fused. The caller provides one FP16
/// activation buffer of length F, which can be reused across layers.
final class SharedExpertInt8 {
    private let int8: DequantInt8GEMV
    private let fusedGateUpActPSO: any MTLComputePipelineState
    private let specializedFusedGateUpActPSO: any MTLComputePipelineState

    init(context: MetalContext) throws {
        self.int8 = try DequantInt8GEMV(context: context)
        self.fusedGateUpActPSO = try context.pipeline("shared_int8_gate_up_act_simd")
        self.specializedFusedGateUpActPSO = try context.pipeline(
            "shared_int8_gate_up_act_simd",
            constants: [
                MetalFunctionConstant(index: 70, value: .uint32(2112)),
                MetalFunctionConstant(index: 71, value: .uint32(2816)),
                MetalFunctionConstant(index: 72, value: .bool(true)),
            ])
    }

    func encode(commandBuffer cb: any MTLCommandBuffer,
                       x: any MTLBuffer, xOffset: Int = 0,
                       gate: SharedExpertProjection,
                       up:   SharedExpertProjection,
                       down: SharedExpertProjection,
                       y: any MTLBuffer, yOffset: Int = 0,
                       scratchAct:  any MTLBuffer, scratchActOffset:  Int = 0) throws {
        guard gate.rows == up.rows, gate.cols == up.cols else {
            throw SharedExpertError.dimensionMismatch(
                "gate/up shapes differ: gate=(\(gate.rows),\(gate.cols)) up=(\(up.rows),\(up.cols))")
        }
        guard down.rows == gate.cols, down.cols == gate.rows else {
            throw SharedExpertError.dimensionMismatch(
                "down expected (\(gate.cols),\(gate.rows)), got (\(down.rows),\(down.cols))")
        }
        let needBytes = Int(gate.rows) * MemoryLayout<Float16>.size
        guard scratchActOffset >= 0, scratchActOffset + needBytes <= scratchAct.length else {
            throw SharedExpertError.scratchTooSmall(
                "scratchAct offset \(scratchActOffset) + needed \(needBytes) exceeds length \(scratchAct.length)")
        }
        let inputBytes = Int(gate.cols) * MemoryLayout<Float16>.size
        let outputBytes = Int(down.rows) * MemoryLayout<Float16>.size
        guard xOffset >= 0, xOffset + inputBytes <= x.length else {
            throw SharedExpertError.scratchTooSmall(
                "x offset \(xOffset) + needed \(inputBytes) exceeds length \(x.length)")
        }
        guard yOffset >= 0, yOffset + outputBytes <= y.length else {
            throw SharedExpertError.scratchTooSmall(
                "y offset \(yOffset) + needed \(outputBytes) exceeds length \(y.length)")
        }

        try encodeFusedGateUpAct(commandBuffer: cb, x: x, xOffset: xOffset, gate: gate, up: up,
                                 scratchAct: scratchAct, scratchActOffset: scratchActOffset)
        try int8.encode(commandBuffer: cb,
                        weights: down.weights, weightsOffset: down.weightsOffset,
                        scales:  down.scales,  scalesOffset:  down.scalesOffset,
                        biases:  down.biases,  biasesOffset:  down.biasesOffset,
                        x: scratchAct, xOffset: scratchActOffset,
                        y: y, yOffset: yOffset,
                        m: down.rows, n: down.cols)
    }

    private func encodeFusedGateUpAct(commandBuffer cb: any MTLCommandBuffer,
                                      x: any MTLBuffer,
                                      xOffset: Int,
                                      gate: SharedExpertProjection,
                                      up: SharedExpertProjection,
                                      scratchAct: any MTLBuffer,
                                      scratchActOffset: Int) throws {
        let enc = try cb.makeTracedComputeEncoder()
        enc.setComputePipelineState(
            gate.rows == 2112 && gate.cols == 2816 ? specializedFusedGateUpActPSO : fusedGateUpActPSO)
        enc.setBuffer(gate.weights, offset: gate.weightsOffset, index: 0)
        enc.setBuffer(gate.scales,  offset: gate.scalesOffset,  index: 1)
        enc.setBuffer(gate.biases,  offset: gate.biasesOffset,  index: 2)
        enc.setBuffer(up.weights,   offset: up.weightsOffset,   index: 3)
        enc.setBuffer(up.scales,    offset: up.scalesOffset,    index: 4)
        enc.setBuffer(up.biases,    offset: up.biasesOffset,    index: 5)
        enc.setBuffer(x, offset: xOffset, index: 6)
        enc.setBuffer(scratchAct, offset: scratchActOffset, index: 7)
        let rows = gate.rows
        let cols = gate.cols
        enc.setValue(rows, index: 8)
        enc.setValue(cols, index: 9)

        let rowsPerTG = 8
        let groups = (Int(gate.rows) + rowsPerTG - 1) / rowsPerTG
        enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32 * rowsPerTG,
                                                                height: 1,
                                                                depth: 1))
        enc.endEncoding()
    }
}
