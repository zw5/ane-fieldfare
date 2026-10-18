import Darwin
import Foundation
import Metal

public struct ExpertCachePlan: Sendable, Equatable {
    public let experts: [Int]
    public let assignedSlots: [Int]
    public let misses: [Int]
    public let hits: Int

    public init(experts: [Int], assignedSlots: [Int], misses: [Int], hits: Int) {
        self.experts = experts
        self.assignedSlots = assignedSlots
        self.misses = misses
        self.hits = hits
    }
}

public enum ExpertCachePolicy: String, Sendable {
    case lru
    case lfu
}

/// Counters for speculative expert prefetch on one streamer.
public struct ExpertPrefetchStats: Sendable, Equatable {
    /// Reads started for predicted experts.
    public var issued: UInt64 = 0
    /// Predicted experts already resident or already loading.
    public var alreadyPresent: UInt64 = 0
    /// Predictions dropped because no slot could be evicted for them.
    public var skipped: UInt64 = 0
    /// Prefetched slots a later plan hit before they were evicted.
    public var useful: UInt64 = 0
    /// Prefetched slots evicted, or overwritten, without ever being hit.
    public var wasted: UInt64 = 0
    /// Prefetch reads that failed; the slot is left empty.
    public var failed: UInt64 = 0
    /// Plans that waited for an in-flight prefetch of an expert they needed.
    public var planWaits: UInt64 = 0

    public init() {}

    public static func + (lhs: Self, rhs: Self) -> Self {
        var out = lhs
        out.issued &+= rhs.issued
        out.alreadyPresent &+= rhs.alreadyPresent
        out.skipped &+= rhs.skipped
        out.useful &+= rhs.useful
        out.wasted &+= rhs.wasted
        out.failed &+= rhs.failed
        out.planWaits &+= rhs.planWaits
        return out
    }
}

/// `pread`-based routed-expert streamer with a fixed per-layer slot cache.
///
/// Slots are empty, resident (holding one expert) or loading (a speculative
/// prefetch read is in flight). Plans never assign a loading slot; a plan that
/// needs an expert being prefetched waits for that read. Prefetch never evicts
/// a slot the most recent plan assigned, so it cannot race a plan's reads or
/// the GPU work that consumes them.
public final class PreadExpertStreamer: @unchecked Sendable {
    public static let scratchAlignment = 2 * 1024 * 1024
    /// Each expert read is split into this many concurrent `pread`s; one
    /// 3.4 MB read at queue depth 1 leaves most SSD bandwidth idle.
    public static let readSplit = 4
    public static var cachePolicyDefault: ExpertCachePolicy { .lfu }

    public let layout: StreamLayout
    public let slotCount: Int
    public let cachePolicy: ExpertCachePolicy

    private let fd: Int32
    private let device: MTLDevice
    private let slotPointers: [UnsafeMutableRawPointer]
    private let slotBuffers: [MTLBuffer]
    private let slotAllocationSize: Int

    private var nextSlot = 0
    private let cursorLock = NSLock()

    private var slotExpert: [Int]
    private var slotLastUse: [Int]
    private var expertUseCount: [Int]
    private var useClock = 0
    /// Expert a prefetch read is writing into the slot, or -1.
    private var slotLoadingExpert: [Int]
    /// The slot was filled by prefetch and no plan has hit it yet.
    private var slotPrefetchedUnused: [Bool]
    private var prefetchCounters = ExpertPrefetchStats()
    /// Guards all slot state; broadcast when a prefetch read settles.
    private let cacheLock = NSCondition()
    private static let prefetchQueue = DispatchQueue(
        label: "ane-fieldfare.expert-prefetch", qos: .userInitiated, attributes: .concurrent)

    public convenience init(layout: StreamLayout,
                            device: MTLDevice,
                            slotCount: Int,
                            cachePolicy: ExpertCachePolicy = .lfu) throws {
        try self.init(layout: layout, device: device, slotCount: slotCount,
                      cachePolicy: cachePolicy, fileDescriptor: nil)
    }

    package init(layout: StreamLayout,
                 device: MTLDevice,
                 slotCount: Int,
                 cachePolicy: ExpertCachePolicy = .lfu,
                 fileDescriptor: Int32?) throws {
        precondition(slotCount > 0, "expert cache needs at least one slot")
        self.layout = layout
        self.slotCount = slotCount
        self.cachePolicy = cachePolicy
        self.device = device
        let pageSize = Int(getpagesize())

        let openedFD = fileDescriptor.map { fcntl($0, F_DUPFD_CLOEXEC, 0) }
            ?? open(layout.path, O_RDONLY | O_NONBLOCK | O_CLOEXEC)
        guard openedFD >= 0 else {
            throw StreamerError.openFailed(path: layout.path, errno: errno)
        }
        self.fd = openedFD
        var closeFDOnFailure = true
        defer { if closeFDOnFailure { close(openedFD) } }

        var fileStats = stat()
        guard fstat(openedFD, &fileStats) == 0,
              (fileStats.st_mode & S_IFMT) == S_IFREG,
              fileStats.st_size >= 0 else {
            throw StreamerError.openFailed(
                path: layout.path, errno: errno == 0 ? EINVAL : errno)
        }
        let (required, requiredOverflow) = layout.streamOffset
            .addingReportingOverflow(layout.streamSize)
        guard !requiredOverflow, UInt64(fileStats.st_size) >= required else {
            throw StreamerError.sizeMismatch(
                expected: requiredOverflow ? UInt64.max : required,
                actual: UInt64(fileStats.st_size))
        }
        guard layout.expertStride > 0,
              layout.expertStride <= UInt64(Int.max - (pageSize - 1)) else {
            throw StreamerError.invalidIOSplitConfiguration(
                "expertStride \(layout.expertStride) is not addressable")
        }

        let allocationSize = ((Int(layout.expertStride) + pageSize - 1) / pageSize) * pageSize
        self.slotAllocationSize = allocationSize
        var pointers: [UnsafeMutableRawPointer] = []
        var buffers: [MTLBuffer] = []
        pointers.reserveCapacity(slotCount)
        buffers.reserveCapacity(slotCount)

        func unwind() {
            for index in buffers.count..<pointers.count {
                free(pointers[index])
            }
        }

        for _ in 0..<slotCount {
            var raw: UnsafeMutableRawPointer?
            let result = posix_memalign(&raw, Self.scratchAlignment, allocationSize)
            guard result == 0, let pointer = raw else {
                unwind()
                throw StreamerError.allocFailed(errno: result)
            }
            pointers.append(pointer)
            nonisolated(unsafe) let capturedPointer = pointer
            guard let buffer = device.makeBuffer(
                bytesNoCopy: pointer,
                length: allocationSize,
                options: .storageModeShared,
                deallocator: { _, _ in free(capturedPointer) })
            else {
                unwind()
                throw StreamerError.bufferWrapFailed
            }
            buffers.append(buffer)
        }

        self.slotPointers = pointers
        self.slotBuffers = buffers
        self.slotExpert = [Int](repeating: -1, count: slotCount)
        self.slotLastUse = [Int](repeating: 0, count: slotCount)
        self.slotLoadingExpert = [Int](repeating: -1, count: slotCount)
        self.slotPrefetchedUnused = [Bool](repeating: false, count: slotCount)
        self.expertUseCount = [Int](repeating: 0, count: max(1, layout.expertsPerLayer))
        closeFDOnFailure = false
    }

    deinit {
        // Prefetch reads hold `self`; none can be in flight here.
        close(fd)
    }

    public var prefetchStats: ExpertPrefetchStats {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        return prefetchCounters
    }

    /// Starts background reads for predicted experts that are neither resident
    /// nor loading, evicting the least valuable slots that are not loading,
    /// not holding a predicted expert and not part of the most recent plan.
    /// Returns immediately; correctness never depends on a prediction.
    public func prefetch(experts rawExperts: [Int]) {
        var seen = Set<Int>()
        let experts = rawExperts.filter { $0 >= 0 && $0 < layout.expertsPerLayer && seen.insert($0).inserted }
        guard !experts.isEmpty else { return }
        cacheLock.lock()
        let wanted = Set(experts)
        var protected = [Bool](repeating: false, count: slotCount)
        for slot in 0..<slotCount {
            protected[slot] = slotLoadingExpert[slot] >= 0
                || wanted.contains(slotExpert[slot])
                || (useClock > 0 && slotLastUse[slot] == useClock)
        }
        var victims = (0..<slotCount).filter { !protected[$0] }
            .sorted { shouldEvictSlot($0, before: $1) }[...]
        var issued: [(slot: Int, expert: Int)] = []
        for expert in experts {
            if slotExpert.contains(expert) || slotLoadingExpert.contains(expert) {
                prefetchCounters.alreadyPresent &+= 1
                continue
            }
            guard let slot = victims.popFirst() else {
                prefetchCounters.skipped &+= 1
                continue
            }
            if slotPrefetchedUnused[slot] { prefetchCounters.wasted &+= 1 }
            slotExpert[slot] = -1
            slotPrefetchedUnused[slot] = false
            slotLoadingExpert[slot] = expert
            prefetchCounters.issued &+= 1
            issued.append((slot, expert))
        }
        cacheLock.unlock()
        for (slot, expert) in issued {
            Self.prefetchQueue.async { [self] in
                let succeeded = (try? loadExpert(layer: 0, expert: expert, slot: slot)) != nil
                cacheLock.lock()
                slotLoadingExpert[slot] = -1
                if succeeded {
                    slotExpert[slot] = expert
                    slotPrefetchedUnused[slot] = true
                } else {
                    prefetchCounters.failed &+= 1
                }
                cacheLock.broadcast()
                cacheLock.unlock()
            }
        }
    }

    /// Blocks until no prefetch read is in flight.
    public func waitForPrefetches() {
        cacheLock.lock()
        while slotLoadingExpert.contains(where: { $0 >= 0 }) { cacheLock.wait() }
        cacheLock.unlock()
    }

    public func loadExpert(layer: Int, expert: Int) throws
        -> (buffer: MTLBuffer, offset: UInt64, size: UInt64) {
        cursorLock.lock()
        let slot = nextSlot
        nextSlot = (nextSlot + 1) % slotCount
        cursorLock.unlock()
        return try loadExpert(layer: layer, expert: expert, slot: slot)
    }

    public func loadExpert(layer: Int, expert: Int, slot: Int) throws
        -> (buffer: MTLBuffer, offset: UInt64, size: UInt64) {
        guard slot >= 0 && slot < slotCount else {
            throw StreamerError.slotOutOfRange(slot)
        }
        let regionOffset = layout.expertOffset(layer: layer, expert: expert)
        let (regionEnd, regionOverflow) = regionOffset.addingReportingOverflow(layout.expertStride)
        guard !regionOverflow, regionEnd <= layout.streamSize else {
            throw StreamerError.offsetOutOfRange(regionOffset)
        }
        let (absoluteOffset, absoluteOverflow) = layout.streamOffset
            .addingReportingOverflow(regionOffset)
        guard !absoluteOverflow, absoluteOffset <= UInt64(Int64.max) else {
            throw StreamerError.offsetOutOfRange(regionOffset)
        }
        try readSplitting(into: slotPointers[slot], fileOffset: absoluteOffset,
                          count: Int(layout.expertStride))
        return (slotBuffers[slot], 0, layout.expertStride)
    }

    public func loadExpertsCached(experts: [Int]) throws
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        try executeExpertCachePlan(planExpertsCached(experts: experts))
    }

    public func planExpertsCached(experts: [Int],
                                  avoidingSlots: Set<Int> = []) -> ExpertCachePlan {
        guard let plan = makeExpertCachePlan(experts: experts, avoidingSlots: avoidingSlots) else {
            preconditionFailure("expert cache cannot place requested misses")
        }
        return plan
    }

    public func planExpertsCachedIfPossible(experts: [Int],
                                            avoidingSlots: Set<Int> = []) -> ExpertCachePlan? {
        makeExpertCachePlan(experts: experts, avoidingSlots: avoidingSlots)
    }

    private func makeExpertCachePlan(experts: [Int],
                                     avoidingSlots rawAvoidingSlots: Set<Int>) -> ExpertCachePlan? {
        precondition(experts.count <= slotCount,
                     "expert cache needs at least \(experts.count) slots")
        let avoidingSlots = Set(rawAvoidingSlots.filter { $0 >= 0 && $0 < slotCount })

        cacheLock.lock()
        defer { cacheLock.unlock() }

        // Wait for in-flight prefetches of requested experts, and for enough
        // settled slots when loading slots would otherwise leave too few.
        let requested = Set(experts)
        var waited = false
        while true {
            let loading = slotLoadingExpert.filter { $0 >= 0 }
            if loading.isEmpty { break }
            let blocking = loading.contains(where: requested.contains)
            let settled = slotCount - loading.count - avoidingSlots.count
            if !blocking && settled >= experts.count { break }
            waited = true
            cacheLock.wait()
        }
        if waited { prefetchCounters.planWaits &+= 1 }

        let clock = useClock + 1
        var assignedSlots = [Int](repeating: -1, count: experts.count)
        var reserved = [Bool](repeating: false, count: slotCount)
        for slot in 0..<slotCount where slotLoadingExpert[slot] >= 0 {
            reserved[slot] = true
        }

        for index in experts.indices {
            for slot in 0..<slotCount
                where !avoidingSlots.contains(slot) &&
                    !reserved[slot] && slotExpert[slot] == experts[index] {
                assignedSlots[index] = slot
                reserved[slot] = true
                break
            }
        }
        for slot in avoidingSlots where !reserved[slot] {
            reserved[slot] = true
        }

        let misses = experts.indices.filter { assignedSlots[$0] == -1 }
        let evictable = (0..<slotCount)
            .filter { !reserved[$0] }
            .sorted { shouldEvictSlot($0, before: $1) }
        guard misses.count <= evictable.count else { return nil }

        useClock = clock
        for expert in experts where expert >= 0 && expert < expertUseCount.count {
            expertUseCount[expert] &+= 1
        }
        for slot in assignedSlots where slot >= 0 {
            slotLastUse[slot] = clock
            if slotPrefetchedUnused[slot] {
                slotPrefetchedUnused[slot] = false
                prefetchCounters.useful &+= 1
            }
        }
        for (offset, index) in misses.enumerated() {
            let slot = evictable[offset]
            assignedSlots[index] = slot
            reserved[slot] = true
            if slotPrefetchedUnused[slot] {
                slotPrefetchedUnused[slot] = false
                prefetchCounters.wasted &+= 1
            }
            slotExpert[slot] = -1
            slotLastUse[slot] = clock
        }

        return ExpertCachePlan(
            experts: experts,
            assignedSlots: assignedSlots,
            misses: misses,
            hits: experts.count - misses.count)
    }

    public func executeExpertCachePlan(_ plan: ExpertCachePlan) throws
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        precondition(plan.experts.count <= slotCount,
                     "expert cache plan exceeds slot count")
        precondition(plan.assignedSlots.count == plan.experts.count,
                     "expert cache plan slot count mismatch")

        let errorLock = NSLock()
        nonisolated(unsafe) var firstError: Error?
        DispatchQueue.concurrentPerform(iterations: plan.misses.count) { missOffset in
            let index = plan.misses[missOffset]
            do {
                _ = try self.loadExpert(
                    layer: 0,
                    expert: plan.experts[index],
                    slot: plan.assignedSlots[index])
            } catch {
                errorLock.lock()
                if firstError == nil { firstError = error }
                errorLock.unlock()
            }
        }
        if let firstError { throw firstError }

        cacheLock.lock()
        for index in plan.misses {
            slotExpert[plan.assignedSlots[index]] = plan.experts[index]
        }
        cacheLock.unlock()

        return expertCachePlanBuffers(plan)
    }

    public func expertCachePlanBuffers(_ plan: ExpertCachePlan)
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        precondition(plan.assignedSlots.count == plan.experts.count,
                     "expert cache plan slot count mismatch")
        return plan.assignedSlots.map { slot in
            (slotBuffers[slot], 0, layout.expertStride)
        }
    }

    private func shouldEvictSlot(_ lhs: Int, before rhs: Int) -> Bool {
        if cachePolicy == .lru {
            return slotLastUse[lhs] < slotLastUse[rhs]
        }
        let lhsExpert = slotExpert[lhs]
        let rhsExpert = slotExpert[rhs]
        if lhsExpert < 0 || rhsExpert < 0 {
            return lhsExpert < rhsExpert
        }
        let lhsCount = lhsExpert < expertUseCount.count ? expertUseCount[lhsExpert] : 0
        let rhsCount = rhsExpert < expertUseCount.count ? expertUseCount[rhsExpert] : 0
        if lhsCount != rhsCount { return lhsCount < rhsCount }
        return slotLastUse[lhs] < slotLastUse[rhs]
    }

    private func readSplitting(into destination: UnsafeMutableRawPointer,
                               fileOffset: UInt64,
                               count: Int) throws {
        let pageSize = Int(getpagesize())
        let parts = Self.readSplit
        let chunk = ((count + parts - 1) / parts + pageSize - 1) / pageSize * pageSize
        guard parts > 1, chunk < count else {
            return try readFull(into: destination, fileOffset: fileOffset, count: count)
        }
        let errorLock = NSLock()
        nonisolated(unsafe) var firstError: Error?
        nonisolated(unsafe) let base = destination
        DispatchQueue.concurrentPerform(iterations: (count + chunk - 1) / chunk) { part in
            let start = part * chunk
            do {
                try self.readFull(into: base.advanced(by: start), fileOffset: fileOffset + UInt64(start),
                                  count: min(chunk, count - start))
            } catch {
                errorLock.lock()
                if firstError == nil { firstError = error }
                errorLock.unlock()
            }
        }
        if let firstError { throw firstError }
    }

    private func readFull(into destination: UnsafeMutableRawPointer,
                          fileOffset: UInt64,
                          count: Int) throws {
        var filled = 0
        while filled < count {
            let (nextOffset, offsetOverflow) = fileOffset
                .addingReportingOverflow(UInt64(filled))
            guard !offsetOverflow, nextOffset <= UInt64(Int64.max) else {
                throw StreamerError.offsetOutOfRange(fileOffset)
            }
            let readCount = pread(
                fd,
                destination.advanced(by: filled),
                count - filled,
                off_t(nextOffset))
            if readCount < 0 {
                throw StreamerError.preadFailed(errno: errno)
            }
            if readCount == 0 {
                throw StreamerError.sizeMismatch(expected: UInt64(count), actual: UInt64(filled))
            }
            filled += readCount
        }
    }

    /// Model-derived CPU-side scratch owned by this streamer: its aligned
    /// slot allocation. Diagnostic metadata for residency reporting, not an
    /// ownership or lifetime API.
    public var diagnosticSlotScratchBytes: UInt64 {
        UInt64(slotCount) * UInt64(slotAllocationSize)
    }
}
