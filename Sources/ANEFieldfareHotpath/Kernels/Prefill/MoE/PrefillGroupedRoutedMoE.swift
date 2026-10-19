import Foundation
import Metal
import Darwin

enum PrefillGroupedRoutedMoEBufferIndex {
    static let hidden = 0
    static let sortedPairs = 1
    static let routePartials = 5
    static let gateUpActScratch = 7
    static let downScratch = 8
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
}

public struct PrefillStreamedTileBinding: Equatable {
    public let expertIDs: [Int]
    public let views: [TensorView]

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

    public func localSlot(for expert: UInt32) -> Int? {
        expertIDs.firstIndex(of: Int(expert))
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
            guard localSlot(for: pair.expert) != nil else {
                throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                    "route expert \(pair.expert) is not bound in tile")
            }
        }
    }

    public static func == (lhs: PrefillStreamedTileBinding,
                           rhs: PrefillStreamedTileBinding) -> Bool {
        guard lhs.expertIDs == rhs.expertIDs, lhs.views.count == rhs.views.count else {
            return false
        }
        for index in lhs.views.indices {
            let l = lhs.views[index]
            let r = rhs.views[index]
            guard l.buffer === r.buffer,
                  l.offset == r.offset,
                  l.length == r.length,
                  l.scaleOffset == r.scaleOffset,
                  l.scaleLength == r.scaleLength,
                  l.biasOffset == r.biasOffset,
                  l.biasLength == r.biasLength,
                  l.shape == r.shape,
                  l.dtype == r.dtype else {
                return false
            }
        }
        return true
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

    @discardableResult
    func encodeStreamedBatched(commandBuffer: any MTLCommandBuffer,
                                      hidden: any MTLBuffer,
                                      hiddenOffset: Int = 0,
                                      sortedPairs: any MTLBuffer,
                                      sortedPairsOffset: Int = 0,
                                      routePartials: any MTLBuffer,
                                      routePartialsOffset: Int = 0,
                                      gateUpActScratch: any MTLBuffer,
                                      gateUpActScratchOffset: Int = 0,
                                      downScratch: any MTLBuffer,
                                      downScratchOffset: Int = 0,
                                      argumentBuffer: PrefillStreamedTileArgumentBuffer,
                                      binding: PrefillStreamedTileBinding,
                                      params: PrefillGroupedRoutedMoEStreamedParams,
                                      pairMicrobatchRows: Int = 32) -> Int {
        guard params.pairCount > 0,
              params.liveExpertCount == UInt32(binding.views.count),
              pairMicrobatchRows > 0 else { return 0 }
        var consumed: UInt32 = 0
        var microbatchCount = 0
        while consumed < params.pairCount {
            var p = params
            p.pairStart = params.pairStart + consumed
            p.pairCount = min(UInt32(pairMicrobatchRows), params.pairCount - consumed)

            if let enc = commandBuffer.makeComputeCommandEncoder() {
                enc.setComputePipelineState(batchedPhase1PSO)
                enc.setBuffer(hidden, offset: hiddenOffset, index: PrefillGroupedRoutedMoEBufferIndex.hidden)
                enc.setBuffer(sortedPairs, offset: sortedPairsOffset, index: PrefillGroupedRoutedMoEBufferIndex.sortedPairs)
                enc.setBuffer(gateUpActScratch, offset: gateUpActScratchOffset,
                              index: PrefillGroupedRoutedMoEBufferIndex.gateUpActScratch)
                enc.setBuffer(argumentBuffer.buffer, offset: 0,
                              index: PrefillGroupedRoutedMoEBufferIndex.expertArgumentState)
                enc.setValue(p, index: PrefillGroupedRoutedMoEBufferIndex.params)
                for view in binding.views {
                    enc.useResource(view.buffer, usage: .read)
                }
                enc.dispatchThreads(MTLSize(width: Int(p.routedIntermediate),
                                            height: Int(p.pairCount),
                                            depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
                enc.endEncoding()
            }

            if let enc = commandBuffer.makeComputeCommandEncoder() {
                enc.setComputePipelineState(batchedDownPSO)
                enc.setBuffer(sortedPairs, offset: sortedPairsOffset, index: PrefillGroupedRoutedMoEBufferIndex.sortedPairs)
                enc.setBuffer(routePartials, offset: routePartialsOffset,
                              index: PrefillGroupedRoutedMoEBufferIndex.routePartials)
                enc.setBuffer(gateUpActScratch, offset: gateUpActScratchOffset,
                              index: PrefillGroupedRoutedMoEBufferIndex.gateUpActScratch)
                enc.setBuffer(downScratch, offset: downScratchOffset,
                              index: PrefillGroupedRoutedMoEBufferIndex.downScratch)
                enc.setBuffer(argumentBuffer.buffer, offset: 0,
                              index: PrefillGroupedRoutedMoEBufferIndex.expertArgumentState)
                enc.setValue(p, index: PrefillGroupedRoutedMoEBufferIndex.params)
                for view in binding.views {
                    enc.useResource(view.buffer, usage: .read)
                }
                enc.dispatchThreads(MTLSize(width: Int(p.d),
                                            height: Int(p.pairCount),
                                            depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
                enc.endEncoding()
            }

            consumed += p.pairCount
            microbatchCount += 1
        }
        return microbatchCount
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
        let p = params
        let segs = segments
        for (name, width) in [("verify_routed_gate_up_act", Int(params.routedIntermediate)),
                              ("verify_routed_down", (Int(params.d) + 1) / 2)] {
            guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
            enc.setComputePipelineState(try context.pipeline(name, constants: constants))
            enc.setBuffer(hidden, offset: 0, index: PrefillGroupedRoutedMoEBufferIndex.hidden)
            enc.setBuffer(sortedPairs, offset: 0, index: PrefillGroupedRoutedMoEBufferIndex.sortedPairs)
            enc.setBuffer(routePartials, offset: 0, index: PrefillGroupedRoutedMoEBufferIndex.routePartials)
            enc.setBuffer(acts, offset: 0, index: PrefillGroupedRoutedMoEBufferIndex.gateUpActScratch)
            enc.setBuffer(argumentBuffer.buffer, offset: 0,
                          index: PrefillGroupedRoutedMoEBufferIndex.expertArgumentState)
            enc.setValue(p, index: PrefillGroupedRoutedMoEBufferIndex.params)
            enc.setValues(segs, index: 11)
            for view in binding.views { enc.useResource(view.buffer, usage: .read) }
            enc.dispatchThreadgroups(MTLSize(width: (width + 7) / 8, height: live, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }
    }

}
