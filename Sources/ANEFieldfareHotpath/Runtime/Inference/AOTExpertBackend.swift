import ANEFieldfareAOTExpertRuntime
import Foundation

enum AOTExpertBackendError: Error, CustomStringConvertible {
    case unavailable(String)
    case incompatibleShape(String)
    case dispatchFailed(String)

    var description: String {
        switch self {
        case .unavailable(let detail): return "AOT expert provider unavailable: \(detail)"
        case .incompatibleShape(let detail): return "AOT expert shape rejected: \(detail)"
        case .dispatchFailed(let detail): return "AOT expert dispatch failed closed: \(detail)"
        }
    }
}

/// Opt-in bridge from the production Swift decode rail to compiler-authored
/// named expert procedures. The C runtime owns the 16 request banks, A/B ring,
/// generation checks, procedure cache and fail-closed lifecycle.
final class AOTExpertBackend: @unchecked Sendable {
    private var runtime: OpaquePointer?
    private let hiddenSize: Int
    private let topK: Int
    private var input: [Float]
    private var outputs: [Float]

    init(root: String, config: ArchConfig) throws {
        guard config.hiddenSize == Int(ANEFieldfareGemma4HiddenSize),
              config.moeIntermediateSize == Int(ANEFieldfareGemma4ExpertIntermediateSize),
              config.numExperts == Int(ANEFieldfareGemma4ExpertCount),
              config.topKExperts == Int(ANEFieldfareGemma4TopK) else {
            throw AOTExpertBackendError.incompatibleShape(
                "H=\(config.hiddenSize) F=\(config.moeIntermediateSize) " +
                "E=\(config.numExperts) K=\(config.topKExperts)")
        }
        hiddenSize = config.hiddenSize
        topK = config.topKExperts
        input = .init(repeating: 0, count: hiddenSize)
        outputs = .init(repeating: 0, count: hiddenSize * topK)
        let shape = ANEFieldfareAOTExpertShape(
            hiddenSize: UInt32(config.hiddenSize),
            intermediateSize: UInt32(config.moeIntermediateSize),
            expertCount: UInt32(config.numExperts),
            topK: UInt32(config.topKExperts),
            layerCount: UInt32(config.numLayers),
            requestBanks: UInt32(ANEFieldfareAOTExpertRequiredRequestBanks),
            capabilityFlags: UInt32(ANEFieldfareAOTCapabilityCompilerNamedProcedures |
                                    ANEFieldfareAOTCapabilityCompletionGenerations),
            procedureCacheCapacity: 16)
        var error: UnsafePointer<CChar>?
        runtime = root.withCString {
            ANEFieldfareAOTExpertRuntimeCreateFromArtifactRoot($0, shape, &error)
        }
        guard runtime != nil else {
            throw AOTExpertBackendError.unavailable(
                error.map { String(cString: $0) } ?? "construction failed")
        }
    }

    deinit { ANEFieldfareAOTExpertRuntimeDestroy(runtime) }

    func call(layer: Int, experts: [Int], weights: [Float16],
              input source: UnsafePointer<Float16>, output: UnsafeMutablePointer<Float16>) throws {
        guard experts.count == topK, weights.count == topK else {
            throw AOTExpertBackendError.dispatchFailed("top-8 metadata count mismatch")
        }
        for index in 0..<hiddenSize { input[index] = Float(source[index]) }
        let expertIDs = experts.map(UInt32.init)
        let ok = expertIDs.withUnsafeBufferPointer { expertPointer in
            input.withUnsafeBufferPointer { inputPointer in
                outputs.withUnsafeMutableBufferPointer { outputPointer in
                    ANEFieldfareAOTExpertRuntimeCallTop8Flat(
                        runtime, UInt32(layer), expertPointer.baseAddress,
                        inputPointer.baseAddress, outputPointer.baseAddress, 30_000_000_000)
                }
            }
        }
        guard ok else {
            let message = ANEFieldfareAOTExpertRuntimeLastError(runtime)
                .map { String(cString: $0) } ?? "unknown provider failure"
            throw AOTExpertBackendError.dispatchFailed(message)
        }
        for column in 0..<hiddenSize {
            var reduced: Float = 0
            for rank in 0..<topK {
                reduced += Float(weights[rank]) * outputs[rank * hiddenSize + column]
            }
            output[column] = Float16(reduced)
        }
    }
}
