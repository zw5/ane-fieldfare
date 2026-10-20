import Foundation
public import Metal

public final class MPPPrefillInt4QMM {
    public enum Variant: String, Sendable {
        case control
        case apple10BF16 = "apple10-bf16"
    }

    public enum Path: String, Sendable {
        case affineThreadgroupF16 = "affine-threadgroup-f16"
        case affineThreadgroupBF16Apple10 = "affine-threadgroup-bf16-apple10"
        case fallback
    }

    private static let tileM = 64
    private static let tileN = 32
    private static let tileK = Quantization.groupSize

    private let pso: any MTLComputePipelineState
    private let variant: Variant

    /// `apple10BF16` (the vision projector) needs an Apple10 GPU.
    public init(context: MetalContext, variant: Variant = .control) throws {
        let name = switch variant {
        case .control: "mpp_prefill_affine_threadgroup_f16"
        case .apple10BF16: "mpp_prefill_affine_threadgroup_bf16_apple10"
        }
        if variant == .apple10BF16, !context.device.supportsFamily(.apple10) {
            throw MetalError.missingFunction("\(name) needs an Apple10 GPU")
        }
        let library = try MetalContext.privateLibrary(device: context.device, module: "tensorops")
        guard let function = library.makeFunction(name: name) else {
            throw MetalError.missingFunction(name)
        }
        self.pso = try context.device.makeComputePipelineState(function: function)
        self.variant = variant
    }

    @discardableResult
    public func encode(commandBuffer: any MTLCommandBuffer,
                       weights: any MTLBuffer, weightsOffset: Int = 0,
                       scales: any MTLBuffer, scalesOffset: Int = 0,
                       biases: any MTLBuffer, biasesOffset: Int = 0,
                       x: any MTLBuffer,
                       y: any MTLBuffer,
                       m: Int,
                       n: Int,
                       k: Int) throws -> Path {
        // The Apple10 kernel takes any M: it bound-checks every store against
        // M and the dispatch rounds up to a whole tile.
        let useApple10 = variant == .apple10BF16
        guard m > 0, n > 0, k > 0, k.isMultiple(of: useApple10 ? 128 : Self.tileK),
              weightsOffset >= 0,
              scalesOffset.isMultiple(of: MemoryLayout<UInt16>.stride),
              biasesOffset.isMultiple(of: MemoryLayout<UInt16>.stride) else {
            return .fallback
        }
        let encoder = try commandBuffer.makeTracedComputeEncoder()

        encoder.setComputePipelineState(pso)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(x, offset: 0, index: 3)
        encoder.setBuffer(y, offset: 0, index: 4)
        encoder.setValue(UInt32(m), index: 5)
        encoder.setValue(UInt32(n), index: 6)
        encoder.setValue(UInt32(k), index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: (n + (useApple10 ? 64 : Self.tileN) - 1)
                        / (useApple10 ? 64 : Self.tileN),
                    height: (m + Self.tileM - 1) / Self.tileM,
                    depth: 1),
            threadsPerThreadgroup: MTLSize(width: pso.threadExecutionWidth * 4,
                                           height: 1,
                                           depth: 1))
        encoder.endEncoding()
        return useApple10 ? .affineThreadgroupBF16Apple10 : .affineThreadgroupF16
    }
}
