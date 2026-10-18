import Foundation
import Metal
import ANEFieldfareHotpath

enum GateFailure: Error, CustomStringConvertible {
    case failed(String)
    var description: String {
        switch self { case .failed(let detail): return detail }
    }
}

@inline(__always)
func require(_ condition: @autoclosure () -> Bool, _ detail: String) throws {
    if !condition() { throw GateFailure.failed(detail) }
}

func requireThrows<T: Error & Equatable>(_ expected: T,
                                         _ detail: String,
                                         _ body: () throws -> Void) throws {
    do {
        try body()
        throw GateFailure.failed("\(detail): unexpectedly succeeded")
    } catch let error as T {
        try require(error == expected, "\(detail): got \(error), expected \(expected)")
    }
}

func runSchedulerAndLifetimeGate() throws {
    try require(RuntimeConfiguration.production.widePrefillTransport == .cachedPread,
                "bounded cached prefill is not the production default")
    try require(RuntimeConfiguration(widePrefillTransport: .preadDoubleBuffered)
        .widePrefillTransport == .preadDoubleBuffered,
                "pread candidate is not explicitly selectable")

    let config = PrefillRoutedTileSchedulerConfig(maxPendingDepth: 1, tileExperts: 8)
    try require(config.fitsSlotBudget(slotCount: 16), "two top-8 banks do not fit 16 slots")
    try require(!config.fitsSlotBudget(slotCount: 15), "15 slots incorrectly admitted")
    try require(!config.fitsSlotBudget(slotCount: 16, reservedHits: 1),
                "reserved seventeenth slot incorrectly admitted")

    let scheduler = PrefillRoutedTileScheduler(config: config)
    let bankA = Array(0..<8)
    let bankB = Array(8..<16)
    try require(scheduler.decide(PrefillRoutedTileSchedulerInput(
        hasPendingTile: true, pendingDepth: 1, pendingAssignedSlots: bankA,
        avoidingSlotPlanAvailable: true)) == .prefetchNext(avoidingSlots: bankA),
                "one pending tile did not select the disjoint plan")
    try require(scheduler.decide(PrefillRoutedTileSchedulerInput(
        hasPendingTile: true, pendingDepth: 2, pendingAssignedSlots: bankA,
        avoidingSlotPlanAvailable: true)) == .drainBeforeIssue(reason: .maxPendingDepthReached),
                "second pending tile did not force drain")
    try require(scheduler.decide(PrefillRoutedTileSchedulerInput(
        hasPendingTile: true, pendingDepth: 1, pendingAssignedSlots: [],
        avoidingSlotPlanAvailable: true)) ==
        .drainBeforeIssue(reason: .pendingTileHasNoAssignedSlots),
                "missing slot ownership did not fail closed")

    var lifetime = PrefillStreamedTileSlotLifetime()
    try lifetime.begin(tileIndex: 0, plannedSlots: bankA)
    try lifetime.begin(tileIndex: 1, plannedSlots: bankB)
    try requireThrows(PrefillStreamedTileLifetimeError.slotReuseBeforeCompletion(
        tileIndex: 2, conflictingTileIndex: 0, slots: bankA),
        "bank reuse before completion") {
        try lifetime.begin(tileIndex: 2, plannedSlots: bankA)
    }
    try lifetime.complete(tileIndex: 0)
    try lifetime.begin(tileIndex: 2, plannedSlots: bankA)
    try lifetime.complete(tileIndex: 1)
    try lifetime.complete(tileIndex: 2)
    try requireThrows(PrefillStreamedTileLifetimeError.duplicateSlots(
        tileIndex: 3, slots: [4, 4]), "duplicate ownership") {
        try lifetime.begin(tileIndex: 3, plannedSlots: [4, 4])
    }
}

func runStreamerGate() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw GateFailure.failed("Metal device unavailable")
    }
    let stride: UInt64 = 64
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("ane-fieldfare-workstream-p-\(UUID().uuidString).bin")
    defer { try? FileManager.default.removeItem(at: url) }
    try Data(repeating: 0xA5, count: Int(stride) * 16).write(to: url)
    let layout = StreamLayout(path: url.path, streamOffset: 0,
                              streamSize: stride * 16, expertsPerLayer: 16,
                              expertStride: stride)
    let streamer = try PreadExpertStreamer(layout: layout, device: device,
                                            slotCount: 16, slotCapacity: 16,
                                            cachePolicy: .lfu)
    let planA = streamer.planExpertsCached(experts: Array(0..<8))
    let planB = streamer.planExpertsCached(experts: Array(8..<16),
                                            avoidingSlots: Set(planA.assignedSlots))
    try require(Set(planA.assignedSlots).isDisjoint(with: planB.assignedSlots),
                "two live plans share slots")
    try require(Set(planA.assignedSlots + planB.assignedSlots) == Set(0..<16),
                "two plans do not cover exactly 16 slots")
    try require(streamer.planExpertsCachedIfPossible(
        experts: Array(0..<8), avoidingSlots: Set(0..<9)) == nil,
                "planner did not fail closed with nine live slots")

    let writer = try FileHandle(forWritingTo: url)
    try writer.truncate(atOffset: stride)
    try writer.close()
    do {
        _ = try streamer.loadExpert(layer: 0, expert: 1, slot: 0)
        throw GateFailure.failed("short read unexpectedly succeeded")
    } catch StreamerError.sizeMismatch(let expected, let actual) {
        try require(expected == stride && actual == 0,
                    "short read ledger mismatch: expected \(expected), actual \(actual)")
    }
    let retry = streamer.planExpertsCached(experts: [1])
    try require(retry.hits == 0 && retry.misses == [0],
                "failed read was published as a cache hit")
}

do {
    try runSchedulerAndLifetimeGate()
    try runStreamerGate()
    print("{\"schema\":\"ane-fieldfare.workstream-p-static.v1\",\"status\":\"validated\",\"transient_slots\":16,\"bank_size\":8,\"max_pending_depth\":1,\"short_read_fail_closed\":true}")
} catch {
    fputs("Workstream P gate rejected: \(error)\n", stderr)
    exit(1)
}
