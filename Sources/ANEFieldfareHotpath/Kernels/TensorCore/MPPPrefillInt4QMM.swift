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

    private let context: MetalContext
    private let controlPSO: any MTLComputePipelineState?
    private let apple10BF16PSO: any MTLComputePipelineState?
    private let variant: Variant
    public let unavailableReason: String?

    public init(context: MetalContext, variant: Variant = .control) {
        self.context = context
        self.variant = variant
        var controlPSO: any MTLComputePipelineState?
        var apple10BF16PSO: any MTLComputePipelineState?
        var unavailableReason: String?
        let apple10FunctionName: String? = variant == .apple10BF16
            ? "mpp_prefill_affine_threadgroup_bf16_apple10_v1" : nil
        do {
            if let apple10FunctionName {
                guard context.device.supportsFamily(.apple10) else {
                    throw NSError(
                        domain: "MPPPrefillInt4QMM",
                        code: 2,
                        userInfo: [NSLocalizedDescriptionKey:
                            "\(apple10FunctionName) needs an Apple10 GPU family device"])
                }
            }
            let library = try MetalContext.privateLibrary(device: context.device, module: "tensorops")
            guard let controlFunction = library.makeFunction(
                name: "mpp_prefill_affine_threadgroup_f16") else {
                throw NSError(
                    domain: "MPPPrefillInt4QMM",
                    code: 1,
                    userInfo: [NSLocalizedDescriptionKey:
                        "mpp_prefill_affine_threadgroup_f16 symbol unavailable"])
            }
            controlPSO = try context.device.makeComputePipelineState(
                function: controlFunction)
            if let apple10FunctionName {
                guard let apple10Function = library.makeFunction(
                    name: apple10FunctionName) else {
                    throw NSError(
                        domain: "MPPPrefillInt4QMM",
                        code: 2,
                        userInfo: [NSLocalizedDescriptionKey:
                            "\(apple10FunctionName) symbol unavailable"])
                }
                apple10BF16PSO = try context.device.makeComputePipelineState(
                    function: apple10Function)
            }
        } catch {
            unavailableReason = String(describing: error)
        }
        self.controlPSO = controlPSO
        self.apple10BF16PSO = apple10BF16PSO
        self.unavailableReason = unavailableReason
    }

    public var isAvailable: Bool {
        controlPSO != nil && (
            variant == .control
                || variant == .apple10BF16 && apple10BF16PSO != nil)
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
                       k: Int) -> Path {
        let supportedShape = m > 0 && n > 0 && k > 0
            && k.isMultiple(of: Self.tileK)
        // The BF16 variant (the vision projector) has no control kernel, so it
        // takes the Apple10 kernel at any M: it bound-checks every store
        // against M and the dispatch rounds up to a whole tile.
        let useApple10 = variant == .apple10BF16
            && context.device.supportsFamily(.apple10)
            && k.isMultiple(of: 128)
            && apple10BF16PSO != nil
        let selectedPSO = variant == .apple10BF16
            ? (useApple10 ? apple10BF16PSO : nil)
            : controlPSO
        guard supportedShape,
              weightsOffset >= 0,
              scalesOffset.isMultiple(of: MemoryLayout<UInt16>.stride),
              biasesOffset.isMultiple(of: MemoryLayout<UInt16>.stride),
              let pso = selectedPSO,
              let encoder = commandBuffer.makeTracedComputeEncoder() else {
            return .fallback
        }

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
