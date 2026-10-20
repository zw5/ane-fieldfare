import Foundation
import Metal
import ANEFieldfareModelFormat

/// Routed-expert access. Every expert of every layer has one fixed, zero-copy
/// buffer over the page cache, so binding never waits on storage; residency
/// only decides how soon the GPU can run. The forward loops ask which experts
/// are absent, bind everything at once, run what is resident, and read the
/// rest concurrently.
extension Model {
    /// Offsets of each Q4_0 region inside an expert blob; the same for every
    /// expert of every layer. The loader checked they fit UInt32.
    public func routedExpertOffsets(layer: Int) -> MoEExpertOffsets {
        let tensors = Dictionary(uniqueKeysWithValues: manifest.experts.tensors.map { ($0.name, $0) })
        func offsets(_ role: String) -> (UInt32, UInt32) {
            guard let tensor = tensors[role], let scales = tensor.scales,
                  let w = UInt32(exactly: tensor.weights.offset), let s = UInt32(exactly: scales.offset) else {
                preconditionFailure("invalid routed expert metadata for role \(role)")
            }
            return (w, s)
        }
        let gate = offsets("gate"), up = offsets("up"), down = offsets("down")
        return MoEExpertOffsets(gateWOff: gate.0, gateSOff: gate.1,
                                upWOff: up.0, upSOff: up.1,
                                downWOff: down.0, downSOff: down.1)
    }

    public func routedExpertPhysicalOffsets(layer: Int) -> [UInt64] {
        (0..<UInt64(manifest.experts.perLayer)).map { $0 * manifest.experts.stride }
    }

    /// Bytes of one routed expert.
    public var routedExpertBytes: Int { Int(manifest.experts.stride) }

    /// The experts' buffers, valid whether or not their pages are resident.
    public func routedExpertViews(layer: Int, experts: [Int]) throws -> [TensorView] {
        let mapped = try expertLayer(layer)
        return try experts.map { expert in
            guard expert >= 0, expert < mapped.expertCount else {
                throw ExpertStoreError.expertOutOfRange(expert)
            }
            return TensorView(
                buffer: mapped.buffers[expert],
                offset: 0,
                length: UInt64(mapped.expertBytes),
                scaleOffset: 0,
                scaleLength: 0,
                shape: (UInt32(layer), UInt32(expert), 0, 0))
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
        guard !experts.isEmpty else { return }
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
