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

/// INT4 shared MLP: gate, up and GELU in one dispatch
/// (`shared_gate_up_act_int4`), then the down GEMV.
public final class SharedExpertInt4 {
    private let int4: DequantInt4GEMV
    private let context: MetalContext

    public init(context: MetalContext) throws {
        self.int4 = try DequantInt4GEMV(context: context)
        self.context = context
    }

    public func encode(commandBuffer cb: any MTLCommandBuffer,
                       x: any MTLBuffer, xOffset: Int = 0,
                       gate: SharedExpertProjection,
                       up: SharedExpertProjection,
                       down: SharedExpertProjection,
                       y: any MTLBuffer, yOffset: Int = 0,
                       scratchAct: any MTLBuffer, scratchActOffset: Int = 0) throws {
        guard gate.rows == up.rows, gate.cols == up.cols,
              down.rows == gate.cols, down.cols == gate.rows else {
            throw SharedExpertError.dimensionMismatch(
                "gate=(\(gate.rows),\(gate.cols)) up=(\(up.rows),\(up.cols)) down=(\(down.rows),\(down.cols))")
        }
        let intermediate = Int(gate.rows)
        let required = intermediate * MemoryLayout<Float16>.stride
        guard scratchActOffset >= 0, scratchActOffset + required <= scratchAct.length else {
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
        // The kernel reads packed weights through a `ushort*`.
        precondition(gate.weightsOffset % 2 == 0 && up.weightsOffset % 2 == 0
                     && gate.cols % UInt32(Quantization.groupSize) == 0,
                     "shared_gate_up_act_int4 needs 2-aligned weights and D a multiple of \(Quantization.groupSize)")
        let d = gate.cols, f = gate.rows
        let pipeline = try context.pipeline("shared_gate_up_act_int4", constants: [
            MetalFunctionConstant(index: 0, value: .uint32(d)),
            MetalFunctionConstant(index: 1, value: .uint32(f)),
            MetalFunctionConstant(index: 3, value: .bool(true)),
        ])
        let encoder = try cb.makeTracedComputeEncoder()
        encoder.setComputePipelineState(pipeline)
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
        try int4.encode(commandBuffer: cb,
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

    public init(context: MetalContext, weightBits: Int) throws {
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
                       scratchAct: any MTLBuffer, scratchActOffset: Int = 0) throws {
        switch implementation {
        case .int4(let runtime):
            try runtime.encode(commandBuffer: commandBuffer, x: x, xOffset: xOffset,
                               gate: gate, up: up, down: down, y: y, yOffset: yOffset,
                               scratchAct: scratchAct, scratchActOffset: scratchActOffset)
        case .int8(let runtime):
            try runtime.encode(commandBuffer: commandBuffer, x: x, xOffset: xOffset,
                               gate: gate, up: up, down: down, y: y, yOffset: yOffset,
                               scratchAct: scratchAct, scratchActOffset: scratchActOffset)
        }
    }
}
