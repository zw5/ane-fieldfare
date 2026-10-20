import Foundation
import Metal
import ANEFieldfareImportedFormat

/// Routed-expert access. Every expert of every layer has one fixed, zero-copy
/// buffer over the page cache, so binding never waits on storage; residency
/// only decides how soon the GPU can run. The forward loops ask which experts
/// are absent, bind everything at once, run what is resident, and read the
/// rest concurrently.
extension Model {
    public func routedExpertOffsets(layer: Int) -> MoEExpertOffsets {
        let expert = packedExpertsLayout.expert(layer: layer, expert: 0)
        func offset(_ role: String) -> UInt32 {
            guard let tensor = expert.subTensors[role],
                  let offset = UInt32(exactly: tensor.offset) else {
                preconditionFailure("invalid routed expert metadata for role \(role)")
            }
            return offset
        }
        return MoEExpertOffsets(
            gateWOff: offset("gate"),
            gateSOff: offset("gate_scales"),
            gateBOff: offset("gate_biases"),
            upWOff: offset("up"),
            upSOff: offset("up_scales"),
            upBOff: offset("up_biases"),
            downWOff: offset("down"),
            downSOff: offset("down_scales"),
            downBOff: offset("down_biases"))
    }

    public func routedExpertPhysicalOffsets(layer: Int) -> [UInt64] {
        packedExpertsLayout.layers[layer].experts.map(\.offset)
    }

    /// Bytes of one routed expert.
    public var routedExpertBytes: Int { Int(packedExpertsLayout.expertStride) }

    /// The experts' buffers, valid whether or not their pages are resident.
    public func routedExpertViews(layer: Int, experts: [Int]) throws -> [TensorView] {
        let mapped = try expertLayer(layer)
        return try experts.map { expert in
            guard expert >= 0, expert < mapped.expertCount else {
                throw MappedExpertLayerError.expertOutOfRange(expert)
            }
            return TensorView(
                buffer: mapped.buffers[expert],
                offset: 0,
                length: UInt64(mapped.expertBytes),
                scaleOffset: 0,
                scaleLength: 0,
                biasOffset: 0,
                biasLength: 0,
                shape: (UInt32(layer), UInt32(expert), 0, 0),
                dtype: GTurboFormatV1.DType.u32.rawValue)
        }
    }

    /// The subset of `experts`, in order, not resident in memory.
    public func nonResidentRoutedExperts(layer: Int, experts: [Int],
                                         probes: Int = MappedExpertLayer.residencyProbes) throws -> [Int] {
        let mapped = try expertLayer(layer)
        return experts.filter { !mapped.isResident($0, probes: probes) }
    }

    /// Reads `experts` into memory, concurrently, off the caller's thread.
    public func readRoutedExperts(layer: Int, experts: [Int]) async throws {
        try await Self.read(experts, of: expertLayer(layer))
    }

    /// Reads `experts` of `mapped` into memory on `BlockingIO`. Takes only the
    /// layer, so concurrent readers need not share the model.
    static func read(_ experts: [Int], of mapped: MappedExpertLayer) async throws {
        guard !experts.isEmpty else { return }
        try await BlockingIO.run { try mapped.read(experts) }
    }

    /// Asks storage for `experts` and returns at once.
    public func prefetchRoutedExperts(layer: Int, experts: [Int]) throws {
        try expertLayer(layer).prefetch(experts)
    }
}
