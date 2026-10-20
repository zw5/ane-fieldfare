import Foundation
public import Metal

public enum SharedExpertError: Error, CustomStringConvertible {
    case dimensionMismatch(String)
    case scratchTooSmall(String)

    public var description: String {
        switch self {
        case .dimensionMismatch(let detail):
            return "SharedExpert dimension mismatch: \(detail)"
        case .scratchTooSmall(let detail):
            return "SharedExpert scratch too small: \(detail)"
        }
    }
}

/// One Q4_0 projection of the shared MLP: its weights and scales, which
/// usually live in the resident buffer, and its `[rows, cols]` shape.
public struct SharedExpertProjection {
    public let weights: any MTLBuffer
    public let scales:  any MTLBuffer
    public let weightsOffset: Int
    public let scalesOffset:  Int
    public let rows: UInt32
    public let cols: UInt32

    public init(weights: any MTLBuffer, scales: any MTLBuffer,
                weightsOffset: Int = 0, scalesOffset: Int = 0,
                rows: UInt32, cols: UInt32) {
        self.weights       = weights
        self.scales        = scales
        self.weightsOffset = weightsOffset
        self.scalesOffset  = scalesOffset
        self.rows          = rows
        self.cols          = cols
    }
}

/// Gemma 4's shared expert, the dense branch beside the routed experts:
/// `y = down(gelu_pytorch_tanh(gate(x)) * up(x))`. Gate, up and GELU run in
/// one dispatch (`shared_gate_up_act_int4`), then the down GEMV.
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
        guard let encoder = cb.makeTracedComputeEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(gate.weights, offset: gate.weightsOffset, index: 0)
        encoder.setBuffer(gate.scales, offset: gate.scalesOffset, index: 1)
        encoder.setBuffer(up.weights, offset: up.weightsOffset, index: 2)
        encoder.setBuffer(up.scales, offset: up.scalesOffset, index: 3)
        encoder.setBuffer(x, offset: xOffset, index: 4)
        encoder.setBuffer(scratchAct, offset: scratchActOffset, index: 5)
        encoder.setValue(d, index: 6)
        encoder.setValue(f, index: 7)
        encoder.dispatchThreadgroups(MTLSize(width: (intermediate + 7) / 8, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        int4.encode(commandBuffer: cb,
                    weights: down.weights, weightsOffset: down.weightsOffset,
                    scales: down.scales, scalesOffset: down.scalesOffset,
                    x: scratchAct, xOffset: scratchActOffset,
                    y: y, yOffset: yOffset,
                    m: down.rows, n: down.cols)
    }
}
