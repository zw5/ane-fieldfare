import Foundation
public import Metal

public final class MPPPrefillInt4QMM {
    public enum Variant: String, Sendable {
        case control
        case apple10BF16 = "apple10-bf16"
    }

    private static let tileM = 64
    private static let tileN = 32
    private static let tileK = Quantization.groupSize

    private let pso: any MTLComputePipelineState
    /// The control product on 128x64 tiles with word-wide dequantization,
    /// for shapes it fits.
    private let widePSO: (any MTLComputePipelineState)?
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
        self.widePSO = try variant == .control
            ? library.makeFunction(name: "mpp_prefill_affine_wide_f16")
                .map { try context.device.makeComputePipelineState(function: $0) }
            : nil
        self.variant = variant
    }

    public func encode(commandBuffer: any MTLCommandBuffer,
                       weights: any MTLBuffer, weightsOffset: Int = 0,
                       scales: any MTLBuffer, scalesOffset: Int = 0,
                       biases: any MTLBuffer, biasesOffset: Int = 0,
                       x: any MTLBuffer,
                       y: any MTLBuffer,
                       m: Int,
                       n: Int,
                       k: Int) throws {
        // The Apple10 kernel takes any M: it bound-checks every store against
        // M and the dispatch rounds up to a whole tile.
        let useApple10 = variant == .apple10BF16
        precondition(m > 0 && n > 0 && k.isMultiple(of: useApple10 ? 128 : Self.tileK) && k > 0
                     && weightsOffset >= 0
                     && scalesOffset.isMultiple(of: MemoryLayout<UInt16>.stride)
                     && biasesOffset.isMultiple(of: MemoryLayout<UInt16>.stride),
                     "INT4 QMM shape m=\(m) n=\(n) k=\(k) or offsets unsupported")
        let encoder = try commandBuffer.makeTracedComputeEncoder()
        let wide = widePSO.flatMap { pso in
            n.isMultiple(of: 64) && k.isMultiple(of: 64) && weightsOffset.isMultiple(of: 4) ? pso : nil
        }

        encoder.setComputePipelineState(wide ?? pso)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(x, offset: 0, index: 3)
        encoder.setBuffer(y, offset: 0, index: 4)
        encoder.setValue(UInt32(m), index: 5)
        encoder.setValue(UInt32(n), index: 6)
        encoder.setValue(UInt32(k), index: 7)
        let tileN = useApple10 || wide != nil ? 64 : Self.tileN
        let tileM = wide != nil ? 128 : Self.tileM
        encoder.dispatchThreadgroups(
            MTLSize(width: (n + tileN - 1) / tileN,
                    height: (m + tileM - 1) / tileM,
                    depth: 1),
            threadsPerThreadgroup: MTLSize(width: pso.threadExecutionWidth * 4,
                                           height: 1,
                                           depth: 1))
        encoder.endEncoding()
    }
}
