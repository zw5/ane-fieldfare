import Foundation
public import Metal

public enum SharedExpertError: Error, CustomStringConvertible {
    case unsupportedWeightBits(Int)
    case dimensionMismatch(String)
    case scratchTooSmall(String)

    public var description: String {
        switch self {
        case .unsupportedWeightBits(let bits):
            return "SharedExpert unsupported weight bits: \(bits)"
        case .dimensionMismatch(let detail):
            return "SharedExpert dimension mismatch: \(detail)"
        case .scratchTooSmall(let detail):
            return "SharedExpert scratch too small: \(detail)"
        }
    }
}

public final class SharedExpertInt4 {
    private let int4: DequantInt4GEMV
    private let geluMulPSO: any MTLComputePipelineState
    private let context: MetalContext

    public init(context: MetalContext) throws {
        self.int4 = try DequantInt4GEMV(context: context)
        self.geluMulPSO = try context.pipeline("gelu_mul_fp16")
        self.context = context
    }

    /// Gate, up and GELU in one dispatch, specialized for (`d`, `f`).
    private func gateUpActPipeline(d: UInt32, f: UInt32) throws -> any MTLComputePipelineState {
        try context.pipeline("shared_gate_up_act_int4", constants: [
            MetalFunctionConstant(index: 0, value: .uint32(d)),
            MetalFunctionConstant(index: 1, value: .uint32(f)),
            MetalFunctionConstant(index: 3, value: .bool(true)),
        ])
    }

    public func encode(commandBuffer cb: any MTLCommandBuffer,
                       x: any MTLBuffer, xOffset: Int = 0,
                       gate: SharedExpertProjection,
                       up: SharedExpertProjection,
                       down: SharedExpertProjection,
                       y: any MTLBuffer, yOffset: Int = 0,
                       scratchGate: any MTLBuffer, scratchGateOffset: Int = 0,
                       scratchUp: any MTLBuffer, scratchUpOffset: Int = 0,
                       scratchAct: any MTLBuffer, scratchActOffset: Int = 0) throws {
        guard gate.rows == up.rows, gate.cols == up.cols,
              down.rows == gate.cols, down.cols == gate.rows else {
            throw SharedExpertError.dimensionMismatch(
                "gate=(\(gate.rows),\(gate.cols)) up=(\(up.rows),\(up.cols)) down=(\(down.rows),\(down.cols))")
        }
        let intermediate = Int(gate.rows)
        let required = intermediate * MemoryLayout<Float16>.stride
        guard scratchGateOffset >= 0, scratchGateOffset + required <= scratchGate.length,
              scratchUpOffset >= 0, scratchUpOffset + required <= scratchUp.length,
              scratchActOffset >= 0, scratchActOffset + required <= scratchAct.length else {
            throw SharedExpertError.scratchTooSmall("need \(required) bytes per intermediate buffer")
        }
        let inputBytes = Int(gate.cols) * MemoryLayout<Float16>.stride
        let outputBytes = Int(down.rows) * MemoryLayout<Float16>.stride
        guard xOffset >= 0, xOffset + inputBytes <= x.length else {
            throw SharedExpertError.scratchTooSmall("input range exceeds x buffer")
        }
        guard yOffset >= 0, yOffset + outputBytes <= y.length else {
            throw SharedExpertError.scratchTooSmall("output range exceeds y buffer")
        }

        if gate.weightsOffset % 2 == 0, up.weightsOffset % 2 == 0,
           gate.cols % UInt32(Quantization.groupSize) == 0,
           let encoder = cb.makeTracedComputeEncoder() {
            let d = gate.cols, f = gate.rows
            encoder.setComputePipelineState(try gateUpActPipeline(d: d, f: f))
            encoder.setBuffer(gate.weights, offset: gate.weightsOffset, index: 0)
            encoder.setBuffer(gate.scales, offset: gate.scalesOffset, index: 1)
            encoder.setBuffer(gate.biases, offset: gate.biasesOffset, index: 2)
            encoder.setBuffer(up.weights, offset: up.weightsOffset, index: 3)
            encoder.setBuffer(up.scales, offset: up.scalesOffset, index: 4)
            encoder.setBuffer(up.biases, offset: up.biasesOffset, index: 5)
            encoder.setBuffer(x, offset: xOffset, index: 6)
            encoder.setBuffer(scratchAct, offset: scratchActOffset, index: 7)
            encoder.setValue(d, index: 8)
            encoder.setValue(f, index: 9)
            encoder.dispatchThreadgroups(MTLSize(width: (intermediate + 7) / 8, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            int4.encode(commandBuffer: cb,
                        weights: down.weights, weightsOffset: down.weightsOffset,
                        scales: down.scales, scalesOffset: down.scalesOffset,
                        biases: down.biases, biasesOffset: down.biasesOffset,
                        x: scratchAct, xOffset: scratchActOffset,
                        y: y, yOffset: yOffset,
                        m: down.rows, n: down.cols)
            return
        }

        int4.encode(commandBuffer: cb,
                    weights: gate.weights, weightsOffset: gate.weightsOffset,
                    scales: gate.scales, scalesOffset: gate.scalesOffset,
                    biases: gate.biases, biasesOffset: gate.biasesOffset,
                    x: x, xOffset: xOffset,
                    y: scratchGate, yOffset: scratchGateOffset,
                    m: gate.rows, n: gate.cols)
        int4.encode(commandBuffer: cb,
                    weights: up.weights, weightsOffset: up.weightsOffset,
                    scales: up.scales, scalesOffset: up.scalesOffset,
                    biases: up.biases, biasesOffset: up.biasesOffset,
                    x: x, xOffset: xOffset,
                    y: scratchUp, yOffset: scratchUpOffset,
                    m: up.rows, n: up.cols)

        guard let encoder = cb.makeTracedComputeEncoder() else { return }
        encoder.setComputePipelineState(geluMulPSO)
        encoder.setBuffer(scratchGate, offset: scratchGateOffset, index: 0)
        encoder.setBuffer(scratchUp, offset: scratchUpOffset, index: 1)
        encoder.setBuffer(scratchAct, offset: scratchActOffset, index: 2)
        let count = UInt32(intermediate)
        encoder.setValue(count, index: 3)
        let width = min(geluMulPSO.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(MTLSize(width: intermediate, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()

        int4.encode(commandBuffer: cb,
                    weights: down.weights, weightsOffset: down.weightsOffset,
                    scales: down.scales, scalesOffset: down.scalesOffset,
                    biases: down.biases, biasesOffset: down.biasesOffset,
                    x: scratchAct, xOffset: scratchActOffset,
                    y: y, yOffset: yOffset,
                    m: down.rows, n: down.cols)
    }
}

public final class SharedExpertRuntime {
    private enum Implementation {
        case int4(SharedExpertInt4)
        case int8(SharedExpertInt8)
    }

    private let implementation: Implementation
    public let weightBits: Int

    public init(context: MetalContext, weightBits: Int) throws {
        self.weightBits = weightBits
        switch weightBits {
        case 4: self.implementation = .int4(try SharedExpertInt4(context: context))
        case 8: self.implementation = .int8(try SharedExpertInt8(context: context))
        default: throw SharedExpertError.unsupportedWeightBits(weightBits)
        }
    }

    public func encode(commandBuffer: any MTLCommandBuffer,
                       x: any MTLBuffer, xOffset: Int = 0,
                       gate: SharedExpertProjection,
                       up: SharedExpertProjection,
                       down: SharedExpertProjection,
                       y: any MTLBuffer, yOffset: Int = 0,
                       scratchGate: any MTLBuffer, scratchGateOffset: Int = 0,
                       scratchUp: any MTLBuffer, scratchUpOffset: Int = 0,
                       scratchAct: any MTLBuffer, scratchActOffset: Int = 0) throws {
        switch implementation {
        case .int4(let runtime):
            try runtime.encode(commandBuffer: commandBuffer, x: x, xOffset: xOffset,
                               gate: gate, up: up, down: down, y: y, yOffset: yOffset,
                               scratchGate: scratchGate, scratchGateOffset: scratchGateOffset,
                               scratchUp: scratchUp, scratchUpOffset: scratchUpOffset,
                               scratchAct: scratchAct, scratchActOffset: scratchActOffset)
        case .int8(let runtime):
            try runtime.encode(commandBuffer: commandBuffer, x: x, xOffset: xOffset,
                               gate: gate, up: up, down: down, y: y, yOffset: yOffset,
                               scratchAct: scratchAct, scratchActOffset: scratchActOffset)
        }
    }
}
