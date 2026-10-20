import Foundation
import Metal

enum PrefillGroupedRoutedMoEBufferIndex {
    static let hidden = 0
    static let sortedPairs = 1
    static let routePartials = 5
    static let actScratch = 7
    static let expertArgumentState = 9
    static let params = 10
}

struct PrefillGroupedRoutedMoEStreamedMetadataBuffers {
    let sortedPairs: any MTLBuffer
}

struct PrefillStreamedTileArgumentBuffer {
    let buffer: any MTLBuffer
}

struct PrefillGroupedRoutedMoEStreamedParams: Equatable, Sendable {
    var pairStart: UInt32
    var pairCount: UInt32
    var d: UInt32
    var routedIntermediate: UInt32
    var topK: UInt32
    var hiddenStrideElements: UInt32
    var liveExpertCount: UInt32
    var localExpert0: UInt32
    var localExpert1: UInt32
    var localExpert2: UInt32
    var localExpert3: UInt32
    var localExpert4: UInt32
    var localExpert5: UInt32
    var localExpert6: UInt32
    var localExpert7: UInt32
    var localExpert8: UInt32
    var localExpert9: UInt32
    var localExpert10: UInt32
    var localExpert11: UInt32
    var localExpert12: UInt32
    var localExpert13: UInt32
    var localExpert14: UInt32
    var localExpert15: UInt32
    var gateWOff: UInt32
    var gateSOff: UInt32
    var gateBOff: UInt32
    var upWOff: UInt32
    var upSOff: UInt32
    var upBOff: UInt32
    var downWOff: UInt32
    var downSOff: UInt32
    var downBOff: UInt32

    init(pairStart: UInt32,
                pairCount: UInt32,
                d: UInt32,
                routedIntermediate: UInt32,
                topK: UInt32,
                hiddenStrideElements: UInt32,
                binding: PrefillStreamedTileBinding,
                offsets: MoEExpertOffsets) {
        var ids = Array(repeating: UInt32.max, count: 16)
        for (index, expert) in binding.expertIDs.enumerated() {
            ids[index] = UInt32(expert)
        }
        self.pairStart = pairStart
        self.pairCount = pairCount
        self.d = d
        self.routedIntermediate = routedIntermediate
        self.topK = topK
        self.hiddenStrideElements = hiddenStrideElements
        self.liveExpertCount = UInt32(binding.expertIDs.count)
        self.localExpert0 = ids[0]
        self.localExpert1 = ids[1]
        self.localExpert2 = ids[2]
        self.localExpert3 = ids[3]
        self.localExpert4 = ids[4]
        self.localExpert5 = ids[5]
        self.localExpert6 = ids[6]
        self.localExpert7 = ids[7]
        self.localExpert8 = ids[8]
        self.localExpert9 = ids[9]
        self.localExpert10 = ids[10]
        self.localExpert11 = ids[11]
        self.localExpert12 = ids[12]
        self.localExpert13 = ids[13]
        self.localExpert14 = ids[14]
        self.localExpert15 = ids[15]
        self.gateWOff = offsets.gateWOff
        self.gateSOff = offsets.gateSOff
        self.gateBOff = offsets.gateBOff
        self.upWOff = offsets.upWOff
        self.upSOff = offsets.upSOff
        self.upBOff = offsets.upBOff
        self.downWOff = offsets.downWOff
        self.downSOff = offsets.downSOff
        self.downBOff = offsets.downBOff
    }

    /// Parameters for the GPU-routed verify kernels, which take their
    /// experts from a bound list instead of a tile binding.
    init(d: UInt32, routedIntermediate: UInt32, topK: UInt32, hiddenStrideElements: UInt32,
         offsets: MoEExpertOffsets) {
        self.init(pairStart: 0, pairCount: 0, d: d, routedIntermediate: routedIntermediate, topK: topK,
                  hiddenStrideElements: hiddenStrideElements,
                  binding: PrefillStreamedTileBinding.unbound, offsets: offsets)
    }
}

/// One entry of a GPU-routed verify layer's bound list: an expert's GPU
/// address and ID, as `VerifyBoundExpertMSL` reads it.
struct VerifyBoundExpert {
    var address: UInt64
    var expert: UInt32
    var pad: UInt32 = 0
}

public struct PrefillStreamedTileBinding {
    public let expertIDs: [Int]
    public let views: [TensorView]

    /// No experts: for parameters whose kernels take experts elsewhere.
    static var unbound: PrefillStreamedTileBinding { PrefillStreamedTileBinding(unchecked: [], views: []) }

    private init(unchecked expertIDs: [Int], views: [TensorView]) {
        self.expertIDs = expertIDs
        self.views = views
    }

    public init(expertIDs: [Int], views: [TensorView]) throws {
        guard !expertIDs.isEmpty else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding("tile binding must include at least one expert")
        }
        guard expertIDs.count <= 16 else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                "tile binding has \(expertIDs.count) experts; maximum is 16")
        }
        guard expertIDs.count == views.count else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                "expertIDs.count \(expertIDs.count) != views.count \(views.count)")
        }
        var seen = Set<Int>()
        for expert in expertIDs {
            guard expert >= 0 else {
                throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                    "expert id \(expert) must be non-negative")
            }
            guard seen.insert(expert).inserted else {
                throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                    "duplicate expert id \(expert) in tile binding")
            }
        }
        self.expertIDs = expertIDs
        self.views = views
    }

    public static func expertIDs(forTile tileIndex: Int,
                                 routes: PrefillMoEGroupedRoutes) throws -> [Int] {
        guard routes.tiles.indices.contains(tileIndex) else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                "tile index \(tileIndex) is out of range")
        }
        let tile = routes.tiles[tileIndex]
        let groupStart = Int(tile.groupStart)
        let groupCount = Int(tile.groupCount)
        guard groupCount > 0, groupCount <= 16 else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                "tile has \(groupCount) live experts; expected 1...16")
        }
        guard groupStart >= 0, groupStart + groupCount <= routes.groups.count else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                "tile group range \(groupStart)..<\(groupStart + groupCount) exceeds \(routes.groups.count)")
        }
        return routes.groups[groupStart..<(groupStart + groupCount)].map { Int($0.expert) }
    }

    public func validateCoversPairs(_ pairs: [PrefillTokenExpertPair],
                                    pairStart: Int,
                                    pairCount: Int) throws {
        guard pairStart >= 0, pairCount >= 0, pairStart + pairCount <= pairs.count else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                "pair range \(pairStart)..<\(pairStart + pairCount) exceeds \(pairs.count)")
        }
        for pair in pairs[pairStart..<(pairStart + pairCount)] {
            guard expertIDs.contains(Int(pair.expert)) else {
                throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                    "route expert \(pair.expert) is not bound in tile")
            }
        }
    }
}

enum PrefillGroupedRoutedMoEError: Error, Equatable, CustomStringConvertible {
    case invalidStreamedTileBinding(String)
    case allocationFailed(String)

    public var description: String {
        switch self {
        case .invalidStreamedTileBinding(let reason):
            return "invalid streamed tile binding: \(reason)"
        case .allocationFailed(let label):
            return "failed to allocate \(label)"
        }
    }
}

final class PrefillGroupedRoutedMoE {
    private let batchedPhase1PSO: any MTLComputePipelineState
    private let batchedDownPSO: any MTLComputePipelineState
    private let streamedArgEncoder: any MTLArgumentEncoder
    private let reusableStreamedArgumentBuffers: [any MTLBuffer]
    static let streamedArgumentBufferCount = 8
    private let reusableSortedPairsBuffer: any MTLBuffer
    private let context: MetalContext
    /// Verify activations, `[pair][F]` by the pair's index in the layer.
    private var verifyActs: any MTLBuffer?
    /// The GPU-routed verify kernels' activations, one row per route.
    private var boundActs: any MTLBuffer?
    /// `verify_bound_plan`'s output: a counter, the routed experts and the
    /// two indirect grids.
    private let boundPlan: any MTLBuffer
    private let boundPlanPSO: any MTLComputePipelineState
    private static let boundPlanRoutedEntryBytes = 80
    private static let boundPlanRowClasses = 4
    private static let boundPlanClassEntries = 128
    private static let boundPlanRoutedOffset = 256
    private static let boundPlanArgsOffset = boundPlanRoutedOffset
        + boundPlanRowClasses * boundPlanClassEntries * boundPlanRoutedEntryBytes
    /// Pairs per expert one verify dispatch takes (`FC_VERIFY_ROWS` <= 16).
    static let verifyMaxRows = 16

    func makeStreamedArgumentBuffer(device: any MTLDevice,
                                    binding: PrefillStreamedTileBinding,
                                    reuseSlot: Int = 0) throws -> PrefillStreamedTileArgumentBuffer {
        precondition(reusableStreamedArgumentBuffers.indices.contains(reuseSlot))
        let buffer = reusableStreamedArgumentBuffers[reuseSlot]

        streamedArgEncoder.setArgumentBuffer(buffer, offset: 0)
        for index in binding.views.indices {
            let view = binding.views[index]
            streamedArgEncoder.setBuffer(view.buffer, offset: Int(view.offset), index: index)
        }

        return PrefillStreamedTileArgumentBuffer(buffer: buffer)
    }

    init(context: MetalContext) throws {
        guard let plan = context.device.makeBuffer(length: Self.boundPlanArgsOffset + Self.boundPlanRowClasses * 24,
                                                   options: .storageModePrivate) else {
            throw MetalError.noDevice
        }
        plan.label = "prefill.groupedMoe.boundPlan"
        self.boundPlan = plan
        self.boundPlanPSO = try context.pipeline("verify_bound_plan")
        self.batchedPhase1PSO = try context.pipeline("prefill_grouped_routed_moe_batched_phase1")
        self.batchedDownPSO = try context.pipeline("prefill_grouped_routed_moe_batched_down")
        guard let streamedFn = context.library.makeFunction(name: "prefill_grouped_routed_moe_batched_phase1") else {
            throw MetalError.missingFunction("prefill_grouped_routed_moe_batched_phase1")
        }
        self.streamedArgEncoder = streamedFn.makeArgumentEncoder(
            bufferIndex: PrefillGroupedRoutedMoEBufferIndex.expertArgumentState)
        var argumentBuffers: [any MTLBuffer] = []
        // Two alternate for drained prefill tiles; verify commits up to eight
        // tiles of one layer before waiting, each with its own buffer.
        for index in 0..<Self.streamedArgumentBufferCount {
            guard let buffer = context.device.makeBuffer(
                length: streamedArgEncoder.encodedLength,
                options: .storageModeShared) else {
                throw MetalError.noDevice
            }
            buffer.label = "prefill.groupedMoe.streamedArgumentBuffer.\(index)"
            argumentBuffers.append(buffer)
        }
        self.reusableStreamedArgumentBuffers = argumentBuffers
        let maximumPairs = PrefillRuntimeConfig.maxChunkTokens * 8
        guard let pairBuffer = context.device.makeBuffer(
            length: maximumPairs * MemoryLayout<PrefillTokenExpertPair>.stride,
            options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        pairBuffer.label = "prefill.groupedMoe.sortedPairs"
        self.reusableSortedPairsBuffer = pairBuffer
        self.context = context
    }

    func makeStreamedMetadataBuffers(
        device: any MTLDevice,
        routes: PrefillMoEGroupedRoutes
    ) throws -> PrefillGroupedRoutedMoEStreamedMetadataBuffers {
        _ = device // Buffer ownership is fixed at runtime construction.
        let bytes = routes.sortedPairs.count * MemoryLayout<PrefillTokenExpertPair>.stride
        guard bytes <= reusableSortedPairsBuffer.length else {
            throw PrefillGroupedRoutedMoEError.allocationFailed(
                "prefill sorted route pairs exceed reusable capacity")
        }
        routes.sortedPairs.withUnsafeBytes { source in
            unsafe UnsafeMutableRawBufferPointer(start: reusableSortedPairsBuffer.contents(), count: bytes)
                .copyMemory(from: source)
        }
        return PrefillGroupedRoutedMoEStreamedMetadataBuffers(
            sortedPairs: reusableSortedPairsBuffer)
    }

    func encodeStreamedBatched(commandBuffer: any MTLCommandBuffer,
                               hidden: any MTLBuffer,
                               sortedPairs: any MTLBuffer,
                               routePartials: any MTLBuffer,
                               actScratch: any MTLBuffer,
                               argumentBuffer: PrefillStreamedTileArgumentBuffer,
                               binding: PrefillStreamedTileBinding,
                               params: PrefillGroupedRoutedMoEStreamedParams,
                               pairMicrobatchRows: Int) throws {
        precondition(params.liveExpertCount == UInt32(binding.views.count) && pairMicrobatchRows > 0)
        var consumed: UInt32 = 0
        while consumed < params.pairCount {
            var p = params
            p.pairStart = params.pairStart + consumed
            p.pairCount = min(UInt32(pairMicrobatchRows), params.pairCount - consumed)

            let gateUp = try commandBuffer.makeTracedComputeEncoder()
            gateUp.setComputePipelineState(batchedPhase1PSO)
            gateUp.setBuffer(hidden, offset: 0, index: PrefillGroupedRoutedMoEBufferIndex.hidden)
            gateUp.setBuffer(sortedPairs, offset: 0, index: PrefillGroupedRoutedMoEBufferIndex.sortedPairs)
            gateUp.setBuffer(actScratch, offset: 0, index: PrefillGroupedRoutedMoEBufferIndex.actScratch)
            gateUp.setBuffer(argumentBuffer.buffer, offset: 0,
                             index: PrefillGroupedRoutedMoEBufferIndex.expertArgumentState)
            gateUp.setValue(p, index: PrefillGroupedRoutedMoEBufferIndex.params)
            for view in binding.views {
                gateUp.useResource(view.buffer, usage: .read)
            }
            gateUp.dispatchThreads(MTLSize(width: Int(p.routedIntermediate),
                                           height: Int(p.pairCount),
                                           depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
            gateUp.endEncoding()

            let down = try commandBuffer.makeTracedComputeEncoder()
            down.setComputePipelineState(batchedDownPSO)
            down.setBuffer(sortedPairs, offset: 0, index: PrefillGroupedRoutedMoEBufferIndex.sortedPairs)
            down.setBuffer(routePartials, offset: 0,
                           index: PrefillGroupedRoutedMoEBufferIndex.routePartials)
            down.setBuffer(actScratch, offset: 0, index: PrefillGroupedRoutedMoEBufferIndex.actScratch)
            down.setBuffer(argumentBuffer.buffer, offset: 0,
                           index: PrefillGroupedRoutedMoEBufferIndex.expertArgumentState)
            down.setValue(p, index: PrefillGroupedRoutedMoEBufferIndex.params)
            for view in binding.views {
                down.useResource(view.buffer, usage: .read)
            }
            down.dispatchThreads(MTLSize(width: Int(p.d),
                                         height: Int(p.pairCount),
                                         depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
            down.endEncoding()

            consumed += p.pairCount
        }
    }

    /// A verify tile's routed experts: per live expert, gate + up + GELU then
    /// down, each weight row read once for all of the expert's tokens.
    /// Writes `routePartials` like `encodeStreamedBatched`.
    func encodeVerifyRows(commandBuffer: any MTLCommandBuffer,
                          hidden: any MTLBuffer,
                          sortedPairs: any MTLBuffer,
                          routePartials: any MTLBuffer,
                          argumentBuffer: PrefillStreamedTileArgumentBuffer,
                          binding: PrefillStreamedTileBinding,
                          params: PrefillGroupedRoutedMoEStreamedParams,
                          segments: [SIMD2<UInt32>],
                          totalPairs: Int) throws {
        let live = Int(params.liveExpertCount)
        precondition(segments.count == live && live == binding.views.count)
        let maxRows = Int(segments.map(\.y).max() ?? 0)
        guard maxRows > 0 else { return }
        precondition(maxRows <= Self.verifyMaxRows, "\(maxRows) pairs for one expert")
        let rows = maxRows <= 1 ? 1 : maxRows <= 2 ? 2 : maxRows <= 4 ? 4 : maxRows <= 8 ? 8 : 16
        let actBytes = totalPairs * Int(params.routedIntermediate) * MemoryLayout<Float16>.stride
        let acts: any MTLBuffer
        if let existing = verifyActs, existing.length >= actBytes {
            acts = existing
        } else {
            guard let buffer = context.device.makeBuffer(
                length: max(actBytes, PrefillRuntimeConfig.maxChunkTokens * 8 * Int(params.routedIntermediate) * 2),
                options: .storageModePrivate) else { throw MetalError.noDevice }
            buffer.label = "prefill.groupedMoe.verifyActs"
            verifyActs = buffer
            acts = buffer
        }
        let constants = [MetalFunctionConstant(index: 90, value: .uint32(UInt32(rows)))]
        for (name, width) in [("verify_routed_gate_up_act", Int(params.routedIntermediate)),
                              ("verify_routed_down", (Int(params.d) + 1) / 2)] {
            let enc = try commandBuffer.makeTracedComputeEncoder()
            enc.setComputePipelineState(try context.pipeline(name, constants: constants))
            enc.setBuffer(hidden, offset: 0, index: PrefillGroupedRoutedMoEBufferIndex.hidden)
            enc.setBuffer(sortedPairs, offset: 0, index: PrefillGroupedRoutedMoEBufferIndex.sortedPairs)
            enc.setBuffer(routePartials, offset: 0, index: PrefillGroupedRoutedMoEBufferIndex.routePartials)
            enc.setBuffer(acts, offset: 0, index: PrefillGroupedRoutedMoEBufferIndex.actScratch)
            enc.setBuffer(argumentBuffer.buffer, offset: 0,
                          index: PrefillGroupedRoutedMoEBufferIndex.expertArgumentState)
            enc.setValue(params, index: PrefillGroupedRoutedMoEBufferIndex.params)
            enc.setValues(segments, index: 11)
            for view in binding.views { enc.useResource(view.buffer, usage: .read) }
            enc.dispatchThreadgroups(MTLSize(width: (width + 7) / 8, height: live, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }
    }

    /// A verify block's routed experts on the GPU, before the host has read
    /// its routes: every route of `routeIDs` (`[rows][topK]`) to an expert in
    /// the `boundCount`-entry list at `bound`/`boundOffset` writes its
    /// partial; routes to any other expert are left for a host patch.
    /// `resident` holds every listed expert's buffer.
    func encodeVerifyBound(commandBuffer: any MTLCommandBuffer,
                           hidden: any MTLBuffer,
                           routeIDs: any MTLBuffer,
                           bound: any MTLBuffer, boundOffset: Int, boundCount: Int,
                           resident: [any MTLBuffer],
                           routePartials: any MTLBuffer,
                           rows: Int,
                           params: PrefillGroupedRoutedMoEStreamedParams) throws {
        guard boundCount > 0 else { return }
        precondition(rows > 0 && rows <= Self.verifyMaxRows)
        // `verify_bound_plan` lists at most this many experts per row class.
        guard boundCount <= Self.boundPlanClassEntries else {
            throw PrefillGroupedRoutedMoEError.allocationFailed(
                "\(boundCount) bound experts exceed the plan's \(Self.boundPlanClassEntries)")
        }
        let routeCount = rows * Int(params.topK)
        let actBytes = routeCount * Int(params.routedIntermediate) * MemoryLayout<Float16>.stride
        let acts: any MTLBuffer
        if let existing = boundActs, existing.length >= actBytes {
            acts = existing
        } else {
            guard let buffer = context.device.makeBuffer(
                length: Self.verifyMaxRows * Int(params.topK) * Int(params.routedIntermediate) * 2,
                options: .storageModePrivate) else { throw MetalError.noDevice }
            buffer.label = "prefill.groupedMoe.boundActs"
            boundActs = buffer
            acts = buffer
        }
        let count = UInt32(routeCount), bounds = UInt32(boundCount)
        // One serial encoder, so the experts are made resident once (each
        // costs ~2.7 µs of host time per encoder).
        let enc = try commandBuffer.makeTracedComputeEncoder()
        enc.setBuffer(hidden, offset: 0, index: 0)
        enc.setBuffer(routeIDs, offset: 0, index: 1)
        enc.setBuffer(bound, offset: boundOffset, index: 2)
        enc.setValue(count, index: 3)
        enc.setValue(bounds, index: 4)
        enc.setBuffer(routePartials, offset: 0, index: 5)
        enc.setBuffer(boundPlan, offset: 0, index: 6)
        enc.setBuffer(acts, offset: 0, index: 7)
        enc.setBuffer(boundPlan, offset: Self.boundPlanRoutedOffset, index: 8)
        enc.setValue(params, index: 10)
        enc.setBuffer(boundPlan, offset: Self.boundPlanArgsOffset, index: 12)
        enc.useResources(resident, usage: .read)
        enc.setComputePipelineState(boundPlanPSO)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        // Each row class (up to 2, 4, 8 or 16 rows) runs kernels
        // specialized to it: register arrays sized for the block's rows cost
        // occupancy (measured: an 8-row block's experts 8% slower).
        let threads = MTLSize(width: 256, height: 1, depth: 1)
        let entryBytes = Self.boundPlanClassEntries * Self.boundPlanRoutedEntryBytes
        for c in 0..<Self.boundPlanRowClasses where c == 0 || 1 << c < rows {
            let constants = [MetalFunctionConstant(index: 90, value: .uint32(UInt32(2 << c)))]
            let args = Self.boundPlanArgsOffset + c * 6 * MemoryLayout<UInt32>.stride
            enc.setBuffer(boundPlan, offset: Self.boundPlanRoutedOffset + c * entryBytes, index: 8)
            enc.setComputePipelineState(try context.pipeline("verify_bound_gate_up_act", constants: constants))
            enc.dispatchThreadgroups(indirectBuffer: boundPlan, indirectBufferOffset: args,
                                     threadsPerThreadgroup: threads)
            enc.setComputePipelineState(try context.pipeline("verify_bound_down", constants: constants))
            enc.dispatchThreadgroups(indirectBuffer: boundPlan,
                                     indirectBufferOffset: args + 3 * MemoryLayout<UInt32>.stride,
                                     threadsPerThreadgroup: threads)
        }
        enc.endEncoding()
    }
}
