import Foundation
import Metal
import ANEFieldfareImportedFormat

public struct RoutedExpertFetchPlan: Sendable {
    public let layer: Int
    public let cachePlan: ExpertCachePlan
    public let buffersStableAcrossFetch: Bool

    public var experts: [Int] { cachePlan.experts }
    public var misses: [Int] { cachePlan.misses }
    public var hits: Int { cachePlan.hits }
    public var assignedSlots: [Int] { cachePlan.assignedSlots }

    public init(layer: Int, cachePlan: ExpertCachePlan,
                buffersStableAcrossFetch: Bool = true) {
        self.layer = layer
        self.cachePlan = cachePlan
        self.buffersStableAcrossFetch = buffersStableAcrossFetch
    }
}

/// A plan against the exact wide-prefill transient carrier. Keeping this type
/// distinct from `RoutedExpertFetchPlan` prevents a candidate plan from being
/// accidentally executed by the persistent recurrent-decode cache.
public struct PrefillRoutedExpertFetchPlan: Sendable {
    public let layer: Int
    public let cachePlan: ExpertCachePlan
    public let expertStride: UInt64
    let carrierGeneration: UInt64

    public var experts: [Int] { cachePlan.experts }
    public var misses: [Int] { cachePlan.misses }
    public var missedExperts: [Int] { cachePlan.misses.map { cachePlan.experts[$0] } }
    public var hits: Int { cachePlan.hits }
    public var assignedSlots: [Int] { cachePlan.assignedSlots }
    public var requestedBytes: UInt64 { UInt64(cachePlan.misses.count) * expertStride }

    init(layer: Int,
         cachePlan: ExpertCachePlan,
         expertStride: UInt64,
         carrierGeneration: UInt64) {
        self.layer = layer
        self.cachePlan = cachePlan
        self.expertStride = expertStride
        self.carrierGeneration = carrierGeneration
    }
}

public struct PrefillRoutedExpertFetchResult: @unchecked Sendable {
    public let views: [TensorView]
    public let preadCalls: Int
    public let preadBytes: UInt64
    public let preadNanos: UInt64

    public init(views: [TensorView],
                preadCalls: Int,
                preadBytes: UInt64,
                preadNanos: UInt64) {
        self.views = views
        self.preadCalls = preadCalls
        self.preadBytes = preadBytes
        self.preadNanos = preadNanos
    }
}

extension Model {
    /// Workstream P's exact transient capacity: two disjoint top-8 banks.
    public static let prefillRoutedExpertSlotCount = 16
    public static let prefillRoutedExpertBankSize = 8

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

    public func adviseRoutedExperts(layer: Int,
                                    experts: [Int]) throws -> ExpertIOAdviceResult {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        return streamer.adviseExpertMisses(experts: experts)
    }

    public func routedExpertAdviceByteEstimate(layer: Int,
                                               missCount: Int) throws -> UInt64 {
        guard missCount > 0 else { return 0 }
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        return UInt64(missCount) * streamer.layout.expertStride
    }

    public func planRoutedExperts(layer: Int,
                                  experts: [Int],
                                  avoidingSlots: Set<Int> = []) throws -> RoutedExpertFetchPlan? {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        let validSlots = Set(avoidingSlots.filter { $0 >= 0 && $0 < streamer.slotCount })
        return RoutedExpertFetchPlan(
            layer: layer,
            cachePlan: streamer.planExpertsCached(experts: experts, avoidingSlots: validSlots),
            buffersStableAcrossFetch: !streamer.isWindowMapped)
    }

    public func planRoutedExpertsIfPossible(layer: Int,
                                            experts: [Int],
                                            avoidingSlots: Set<Int> = []) throws
        -> RoutedExpertFetchPlan? {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        let validSlots = Set(avoidingSlots.filter { $0 >= 0 && $0 < streamer.slotCount })
        guard let cachePlan = streamer.planExpertsCachedIfPossible(
            experts: experts,
            avoidingSlots: validSlots)
        else {
            return nil
        }
        return RoutedExpertFetchPlan(layer: layer, cachePlan: cachePlan,
                                     buffersStableAcrossFetch: !streamer.isWindowMapped)
    }

    public func routedExpertCacheSlotCount(layer _: Int) -> Int? {
        switch streamingMode {
        case .pread(let slotCount): return slotCount
        case .mapped: return packedExpertsLayout.expertsPerLayer
        case .adaptive(let total, _, _): return total / packedExpertsLayout.numLayers
        }
    }

    public func configureRoutedExpertCacheSlots(_ counts: [Int]) throws {
        guard case .adaptive(let total, let minimum, let maximum) = streamingMode,
              counts.count == packedExpertsLayout.numLayers,
              counts.reduce(0, +) == total,
              counts.allSatisfy({ $0 >= minimum && $0 <= maximum }) else {
            return
        }
        for layer in counts.indices { try ensureLayerOpened(layer) }
        streamersQueue.sync {
            // Release pages first, so reassignment cannot transiently exceed
            // the fixed physical budget.
            for layer in counts.indices {
                let current = streamersBox.streamers[layer]!.slotCount
                if counts[layer] < current {
                    streamersBox.streamers[layer]!.setActiveSlotCount(counts[layer])
                }
            }
            for layer in counts.indices {
                let current = streamersBox.streamers[layer]!.slotCount
                if counts[layer] > current {
                    streamersBox.streamers[layer]!.setActiveSlotCount(counts[layer])
                }
            }
        }
    }

    public func routedExpertBuffers(for plan: RoutedExpertFetchPlan) throws -> [TensorView] {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        return Self.makeExpertViews(
            streamer.expertCachePlanBuffers(plan.cachePlan),
            layer: plan.layer,
            experts: plan.experts)
    }

    public func adviseRoutedExperts(plan: RoutedExpertFetchPlan) throws -> ExpertIOAdviceResult {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        return streamer.adviseExpertCachePlanMisses(plan.cachePlan)
    }

    public func fetchRoutedExperts(plan: RoutedExpertFetchPlan) async throws -> [TensorView] {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let buffers = try streamer.executeExpertCachePlan(plan.cachePlan)
                    continuation.resume(returning: Self.makeExpertViews(
                        buffers,
                        layer: plan.layer,
                        experts: plan.experts))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    public func fetchRoutedExperts(layer: Int, experts: [Int]) async throws -> [TensorView] {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let buffers = try streamer.loadExpertsCached(experts: experts)
                    continuation.resume(returning: Self.makeExpertViews(
                        buffers,
                        layer: layer,
                        experts: experts))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// Return the one model-wide candidate carrier without changing the
    /// persistent decode cache. A layer transition releases the old 16 slots
    /// before allocating the new pair of banks. Transition during a fetch is
    /// rejected so two layer carriers cannot overlap.
    private func prefillPreadStreamer(
        layer: Int
    ) throws -> (streamer: PreadExpertStreamer, generation: UInt64) {
        try ensureLayerOpened(layer)
        return try streamersQueue.sync {
            () throws -> (streamer: PreadExpertStreamer, generation: UInt64) in
            if streamersBox.prefillPreadLayer == layer,
               let existing = streamersBox.prefillPreadStreamer {
                return (existing, streamersBox.prefillPreadGeneration)
            }
            guard streamersBox.prefillPreadFetches == 0 else {
                throw StreamerError.invalidIOSplitConfiguration(
                    "cannot change wide-prefill layer while an explicit read is active")
            }
            let decodeStreamer = streamersBox.streamers[layer]!
            // Drop the previous layer before allocating this layer's banks.
            streamersBox.prefillPreadStreamer = nil
            streamersBox.prefillPreadLayer = nil
            let transient = try PreadExpertStreamer(
                layout: decodeStreamer.layout,
                device: device,
                slotCount: Self.prefillRoutedExpertSlotCount,
                slotCapacity: Self.prefillRoutedExpertSlotCount,
                cachePolicy: .lru,
                directMapped: false,
                allowMappedCopySource: false,
                fileDescriptor: nil)
            guard !transient.isDirectMapped,
                  transient.slotCount == Self.prefillRoutedExpertSlotCount,
                  transient.slotCapacity == Self.prefillRoutedExpertSlotCount else {
                throw StreamerError.invalidIOSplitConfiguration(
                    "wide-prefill transient carrier is not exactly 16 explicit-pread slots")
            }
            streamersBox.prefillPreadGeneration &+= 1
            streamersBox.prefillPreadStreamer = transient
            streamersBox.prefillPreadLayer = layer
            return (transient, streamersBox.prefillPreadGeneration)
        }
    }

    public func prefillRoutedExpertCacheSlotCount(layer: Int) -> Int? {
        guard layer >= 0 && layer < packedExpertsLayout.numLayers else { return nil }
        return Self.prefillRoutedExpertSlotCount
    }

    /// Plan one router-authoritative exact top-8 group. `avoidingSlots` is the
    /// live bank owned by the consuming command; the planner must place this
    /// group entirely in the other bank or reject it.
    public func planPrefillRoutedExpertsIfPossible(
        layer: Int,
        experts: [Int],
        avoidingSlots: Set<Int> = []
    ) throws -> PrefillRoutedExpertFetchPlan? {
        guard layer >= 0 && layer < packedExpertsLayout.numLayers else {
            throw StreamerError.invalidIOSplitConfiguration(
                "wide-prefill layer \(layer) is out of range")
        }
        guard !experts.isEmpty,
              experts.count <= Self.prefillRoutedExpertBankSize,
              Set(experts).count == experts.count,
              experts.allSatisfy({ $0 >= 0 && $0 < packedExpertsLayout.expertsPerLayer }) else {
            throw StreamerError.invalidIOSplitConfiguration(
                "wide-prefill plan requires 1...8 unique in-range experts")
        }
        guard avoidingSlots.allSatisfy({
            $0 >= 0 && $0 < Self.prefillRoutedExpertSlotCount
        }) else {
            throw StreamerError.invalidIOSplitConfiguration(
                "wide-prefill avoiding slot is out of range")
        }
        let carrier = try prefillPreadStreamer(layer: layer)
        guard let cachePlan = carrier.streamer.planExpertsCachedIfPossible(
            experts: experts,
            avoidingSlots: avoidingSlots),
              cachePlan.assignedSlots.allSatisfy({ !avoidingSlots.contains($0) }) else {
            return nil
        }
        return PrefillRoutedExpertFetchPlan(
            layer: layer,
            cachePlan: cachePlan,
            expertStride: carrier.streamer.layout.expertStride,
            carrierGeneration: carrier.generation)
    }

    /// Bind a completed transient plan without performing I/O. Callers use
    /// this only after `fetchPrefillRoutedExperts(plan:)` has succeeded.
    public func prefillRoutedExpertBuffers(
        for plan: PrefillRoutedExpertFetchPlan
    ) throws -> [TensorView] {
        let streamer = try streamersQueue.sync { () throws -> PreadExpertStreamer in
            guard streamersBox.prefillPreadLayer == plan.layer,
                  streamersBox.prefillPreadGeneration == plan.carrierGeneration,
                  let streamer = streamersBox.prefillPreadStreamer else {
                throw StreamerError.invalidIOSplitConfiguration(
                    "wide-prefill plan belongs to a released layer carrier")
            }
            return streamer
        }
        return Self.makeExpertViews(
            streamer.expertCachePlanBuffers(plan.cachePlan),
            layer: plan.layer,
            experts: plan.experts)
    }

    /// Execute only explicit `pread` misses for the authoritative expert IDs.
    /// Short reads and offset failures propagate; there is no mapped fallback.
    public func fetchPrefillRoutedExperts(
        plan: PrefillRoutedExpertFetchPlan
    ) async throws -> PrefillRoutedExpertFetchResult {
        let streamer = try streamersQueue.sync { () throws -> PreadExpertStreamer in
            guard streamersBox.prefillPreadLayer == plan.layer,
                  streamersBox.prefillPreadGeneration == plan.carrierGeneration,
                  let streamer = streamersBox.prefillPreadStreamer else {
                throw StreamerError.invalidIOSplitConfiguration(
                    "wide-prefill plan belongs to a released layer carrier")
            }
            streamersBox.prefillPreadFetches += 1
            return streamer
        }
        guard !streamer.isDirectMapped else {
            streamersQueue.sync { streamersBox.prefillPreadFetches -= 1 }
            throw StreamerError.invalidIOSplitConfiguration(
                "wide-prefill candidate cannot execute against a mapped source")
        }
        nonisolated(unsafe) let model = self
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                defer {
                    model.streamersQueue.sync {
                        model.streamersBox.prefillPreadFetches -= 1
                    }
                }
                do {
                    let result = try streamer.executeExpertCachePlanMeasured(plan.cachePlan)
                    guard result.stats.bytes == plan.requestedBytes else {
                        throw StreamerError.sizeMismatch(
                            expected: plan.requestedBytes,
                            actual: result.stats.bytes)
                    }
                    continuation.resume(returning: PrefillRoutedExpertFetchResult(
                        views: Self.makeExpertViews(
                            result.buffers,
                            layer: plan.layer,
                            experts: plan.experts),
                        preadCalls: result.stats.calls,
                        preadBytes: result.stats.bytes,
                        preadNanos: result.stats.nanos))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// Exact wide-prefill carrier.  Unlike recurrent decode, a token tile
    /// selects almost the entire expert population.  Binding compact records
    /// directly avoids copying those records through the random-access
    /// cache while preserving every routed expert and routing weight.
    private func mappedPrefillStreamer(layer: Int) throws -> PreadExpertStreamer {
        try ensureLayerOpened(layer)
        return try streamersQueue.sync { () throws -> PreadExpertStreamer in
            if case .mapped = streamingMode {
                return streamersBox.streamers[layer]!
            }
            if let existing = streamersBox.prefillMappedStreamers[layer] {
                return existing
            }
            let decodeStreamer = streamersBox.streamers[layer]!
            let mapped = try PreadExpertStreamer(
                layout: decodeStreamer.layout,
                device: device,
                slotCount: packedExpertsLayout.expertsPerLayer,
                slotCapacity: packedExpertsLayout.expertsPerLayer,
                cachePolicy: expertCachePolicy,
                directMapped: true,
                fileDescriptor: nil)
            streamersBox.prefillMappedStreamers[layer] = mapped
            return mapped
        }
    }

    public func mappedPrefillRoutedExperts(layer: Int,
                                           experts: [Int]) throws -> [TensorView] {
        let streamer = try mappedPrefillStreamer(layer: layer)
        return Self.makeExpertViews(streamer.directMappedExpertBuffers(experts: experts),
                                    layer: layer,
                                    experts: experts)
    }

    /// Start one coalesced kernel read-ahead for the next compact layer while
    /// Metal consumes the current one.  Wide exact routing is dense enough in
    /// expert identity that this is useful work independent of the next
    /// layer's as-yet-unknown router output.
    public func beginPreparingMappedPrefill(layer: Int) {
        guard layer >= 0 && layer < packedExpertsLayout.numLayers else { return }
        nonisolated(unsafe) let model = self
        DispatchQueue.global(qos: .userInitiated).async {
            guard let streamer = try? model.mappedPrefillStreamer(layer: layer) else { return }
            _ = streamer.adviseExperts(experts: Array(0..<model.packedExpertsLayout.expertsPerLayer))
        }
    }

    private static func makeExpertViews(
        _ buffers: [(buffer: MTLBuffer, offset: UInt64, size: UInt64)],
        layer: Int,
        experts: [Int]
    ) -> [TensorView] {
        buffers.enumerated().map { index, entry in
            TensorView(
                buffer: entry.buffer,
                offset: entry.offset,
                length: entry.size,
                scaleOffset: 0,
                scaleLength: 0,
                biasOffset: 0,
                biasLength: 0,
                shape: (UInt32(layer), UInt32(experts[index]), 0, 0),
                dtype: GTurboFormatV1.DType.u32.rawValue)
        }
    }
}
