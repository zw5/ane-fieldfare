import Darwin
import Foundation
import Metal

public struct ExpertIOAdviceResult: Sendable, Equatable {
    public let requested: Int
    public let failed: Int
    public let calls: Int
    public let bytes: UInt64
    public let skipped: Int
    public let maxCallNanos: UInt64

    public init(requested: Int,
                failed: Int,
                calls: Int? = nil,
                bytes: UInt64 = 0,
                skipped: Int = 0,
                maxCallNanos: UInt64 = 0) {
        self.requested = requested
        self.failed = failed
        self.calls = calls ?? requested
        self.bytes = bytes
        self.skipped = skipped
        self.maxCallNanos = maxCallNanos
    }

    public static func skipped(requested: Int, bytes: UInt64 = 0) -> ExpertIOAdviceResult {
        ExpertIOAdviceResult(requested: requested,
                             failed: 0,
                             calls: 0,
                             bytes: bytes,
                             skipped: requested)
    }

}

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

public struct ExpertPreadStats: Sendable, Equatable {
    public let calls: Int
    public let bytes: UInt64
    public let nanos: UInt64

    public init(calls: Int = 0, bytes: UInt64 = 0, nanos: UInt64 = 0) {
        self.calls = calls
        self.bytes = bytes
        self.nanos = nanos
    }
}

public enum ExpertCachePolicy: String, Sendable {
    case lru
    case lfu
}

/// `pread`-based routed-expert streamer with a fixed per-layer slot cache.
public final class PreadExpertStreamer: @unchecked Sendable {
    public static let scratchAlignment = 2 * 1024 * 1024
    public static var cachePolicyDefault: ExpertCachePolicy { .lfu }

    public let layout: StreamLayout
    public private(set) var slotCount: Int
    public let slotCapacity: Int
    public let cachePolicy: ExpertCachePolicy
    public let isDirectMapped: Bool
    public let isWindowMapped: Bool

    private let fd: Int32
    private let device: MTLDevice
    private let slotPointers: [UnsafeMutableRawPointer]
    private var slotBuffers: [MTLBuffer]
    private var slotOffsets: [UInt64]
    private let windowLock = NSLock()
    private let emptyWindow: MTLBuffer?
    private var sourceMapping: UnsafeRawPointer?
    private var sourceMappingLength: Int = 0
    private let slotAllocationSize: Int

    private var nextSlot = 0
    private let cursorLock = NSLock()

    private var slotExpert: [Int]
    private var slotLastUse: [Int]
    private var expertUseCount: [Int]
    private var useClock = 0
    private let cacheLock = NSLock()

    public convenience init(layout: StreamLayout,
                            device: MTLDevice,
                            slotCount: Int,
                            cachePolicy: ExpertCachePolicy = .lfu) throws {
        try self.init(layout: layout, device: device, slotCount: slotCount,
                      slotCapacity: nil, cachePolicy: cachePolicy)
    }

    public convenience init(layout: StreamLayout,
                            device: MTLDevice,
                            slotCount: Int,
                            slotCapacity: Int?,
                            cachePolicy: ExpertCachePolicy = .lfu) throws {
        try self.init(layout: layout,
                      device: device,
                      slotCount: slotCount,
                      slotCapacity: slotCapacity,
                      cachePolicy: cachePolicy,
                      fileDescriptor: nil)
    }

    package init(layout: StreamLayout,
                 device: MTLDevice,
                 slotCount: Int,
                 slotCapacity: Int? = nil,
                 cachePolicy: ExpertCachePolicy = .lfu,
                 directMapped: Bool = false,
                 windowMapped: Bool = false,
                 allowMappedCopySource: Bool = true,
                 fileDescriptor: Int32?) throws {
        let resolvedCapacity = slotCapacity ?? slotCount
        precondition(slotCount > 0 && resolvedCapacity >= slotCount,
                     "slot capacity must cover the active slots")
        self.layout = layout
        self.slotCount = slotCount
        self.slotCapacity = resolvedCapacity
        self.cachePolicy = cachePolicy
        self.isDirectMapped = directMapped
        precondition(!directMapped || !windowMapped, "mapping modes are exclusive")
        self.isWindowMapped = windowMapped
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
        var offsets: [UInt64] = []
        pointers.reserveCapacity(slotCount)
        buffers.reserveCapacity(slotCount)

        func unwind() {
            for index in buffers.count..<pointers.count {
                free(pointers[index])
            }
        }

        if windowMapped {
            guard let empty = device.makeBuffer(length: pageSize, options: .storageModeShared) else {
                throw StreamerError.bufferWrapFailed
            }
            self.emptyWindow = empty
            buffers = Array(repeating: empty, count: resolvedCapacity)
            offsets = Array(repeating: 0, count: resolvedCapacity)
        } else if directMapped {
            self.emptyWindow = nil
            let mapLength = Int(layout.streamSize)
            let mapped = mmap(nil, mapLength, PROT_READ, MAP_PRIVATE, openedFD,
                              off_t(layout.streamOffset))
            guard mapped != MAP_FAILED, let mapped else {
                throw StreamerError.openFailed(path: layout.path, errno: errno)
            }
            nonisolated(unsafe) let captured = mapped
            guard let buffer = device.makeBuffer(
                bytesNoCopy: mapped, length: mapLength, options: .storageModeShared,
                deallocator: { _, _ in munmap(captured, mapLength) }) else {
                munmap(mapped, mapLength)
                throw StreamerError.bufferWrapFailed
            }
            buffers = Array(repeating: buffer, count: resolvedCapacity)
            offsets = (0..<resolvedCapacity).map { layout.expertOffset(layer: 0, expert: $0) }
        } else {
            self.emptyWindow = nil
            for _ in 0..<resolvedCapacity {
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
                offsets.append(0)
            }
        }

        self.slotPointers = pointers
        self.slotBuffers = buffers
        self.slotOffsets = offsets
        self.slotExpert = directMapped ? Array(0..<resolvedCapacity) :
            [Int](repeating: -1, count: resolvedCapacity)
        self.slotLastUse = [Int](repeating: 0, count: resolvedCapacity)
        self.expertUseCount = [Int](repeating: 0, count: max(1, layout.expertsPerLayer))
        if !directMapped && !windowMapped && allowMappedCopySource &&
            ProcessInfo.processInfo.environment["ANE_FIELDFARE_HOTPATH_MMAP_COPY"] == "1" {
            let mapLength = Int(layout.streamSize)
            let mapped = mmap(nil, mapLength, PROT_READ, MAP_PRIVATE, openedFD,
                              off_t(layout.streamOffset))
            if mapped != MAP_FAILED, let mapped {
                self.sourceMapping = UnsafeRawPointer(mapped)
                self.sourceMappingLength = mapLength
                _ = posix_madvise(mapped, mapLength, POSIX_MADV_RANDOM)
            }
        }
        closeFDOnFailure = false
    }

    deinit {
        if let sourceMapping, sourceMappingLength > 0 {
            munmap(UnsafeMutableRawPointer(mutating: sourceMapping), sourceMappingLength)
        }
        close(fd)
    }

    /// Reassign active cache capacity at a token boundary. Shrinking returns
    /// abandoned slot pages to the OS; growing exposes already allocated but
    /// untouched virtual carriers. Callers keep the model-wide slot sum fixed.
    public func setActiveSlotCount(_ count: Int) {
        precondition(count >= 8 && count <= slotCapacity)
        cacheLock.lock()
        defer { cacheLock.unlock() }
        if count < slotCount {
            for slot in count..<slotCount {
                slotExpert[slot] = -1
                slotLastUse[slot] = 0
                if slot < slotPointers.count {
                    _ = madvise(slotPointers[slot], slotAllocationSize, MADV_DONTNEED)
                }
                if let emptyWindow {
                    windowLock.lock()
                    slotBuffers[slot] = emptyWindow
                    slotOffsets[slot] = 0
                    windowLock.unlock()
                }
            }
        } else if count > slotCount {
            for slot in slotCount..<count {
                slotExpert[slot] = -1
                slotLastUse[slot] = 0
            }
        }
        slotCount = count
        if nextSlot >= count { nextSlot = 0 }
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
        try loadExpertMeasured(layer: layer, expert: expert, slot: slot).entry
    }

    private func loadExpertMeasured(layer: Int, expert: Int, slot: Int) throws
        -> (entry: (buffer: MTLBuffer, offset: UInt64, size: UInt64), stats: ExpertPreadStats) {
        guard slot >= 0 && slot < slotCount else {
            throw StreamerError.slotOutOfRange(slot)
        }
        let regionOffset = layout.expertOffset(layer: layer, expert: expert)
        let (regionEnd, regionOverflow) = regionOffset.addingReportingOverflow(layout.expertStride)
        guard !regionOverflow, regionEnd <= layout.streamSize else {
            throw StreamerError.offsetOutOfRange(regionOffset)
        }
        let stats: ExpertPreadStats
        let (absoluteOffset, absoluteOverflow) = layout.streamOffset
            .addingReportingOverflow(regionOffset)
        guard !absoluteOverflow, absoluteOffset <= UInt64(Int64.max) else {
            throw StreamerError.offsetOutOfRange(regionOffset)
        }
        if isWindowMapped {
            let entry = try mapExpertWindow(fileOffset: absoluteOffset, slot: slot)
            return (entry, ExpertPreadStats())
        }
        if let sourceMapping {
            memcpy(slotPointers[slot], sourceMapping.advanced(by: Int(regionOffset)),
                   Int(layout.expertStride))
            stats = ExpertPreadStats()
        } else {
            stats = try readFull(
                into: slotPointers[slot],
                fileOffset: absoluteOffset,
                count: Int(layout.expertStride))
        }
        return ((slotBuffers[slot], 0, layout.expertStride), stats)
    }

    /// Keep only cache-selected expert windows mapped. Metal retains each
    /// buffer through command completion; its deallocator owns the mapping,
    /// so eviction cannot unmap a buffer still used by an encoded command.
    private func mapExpertWindow(fileOffset: UInt64, slot: Int) throws
        -> (buffer: MTLBuffer, offset: UInt64, size: UInt64) {
        let page = UInt64(getpagesize())
        let base = fileOffset / page * page
        let delta = fileOffset - base
        guard delta + layout.expertStride <= UInt64(Int.max) - (page - 1) else {
            throw StreamerError.offsetOutOfRange(fileOffset)
        }
        let length = Int((delta + layout.expertStride + page - 1) / page * page)
        guard let mapped = mmap(nil, length, PROT_READ, MAP_PRIVATE, fd, off_t(base)),
              mapped != MAP_FAILED else {
            throw StreamerError.openFailed(path: layout.path, errno: errno)
        }
        nonisolated(unsafe) let captured = mapped
        guard let buffer = device.makeBuffer(
            bytesNoCopy: mapped, length: length, options: .storageModeShared,
            deallocator: { _, _ in munmap(captured, length) }) else {
            munmap(mapped, length)
            throw StreamerError.bufferWrapFailed
        }
        windowLock.lock()
        slotBuffers[slot] = buffer
        slotOffsets[slot] = delta
        windowLock.unlock()
        return (buffer, delta, layout.expertStride)
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

        let clock = useClock + 1
        var assignedSlots = [Int](repeating: -1, count: experts.count)
        var reserved = [Bool](repeating: false, count: slotCount)

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
        }
        for (offset, index) in misses.enumerated() {
            let slot = evictable[offset]
            assignedSlots[index] = slot
            reserved[slot] = true
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

    /// Execute a cache plan while accounting for the actual successful pread
    /// loop calls and bytes. A mapped-copy source reports zero and is rejected
    /// by the exact wide-prefill carrier before this method is reached.
    public func executeExpertCachePlanMeasured(_ plan: ExpertCachePlan) throws
        -> (buffers: [(buffer: MTLBuffer, offset: UInt64, size: UInt64)],
            stats: ExpertPreadStats) {
        precondition(plan.experts.count <= slotCount,
                     "expert cache plan exceeds slot count")
        precondition(plan.assignedSlots.count == plan.experts.count,
                     "expert cache plan slot count mismatch")

        let resultLock = NSLock()
        nonisolated(unsafe) var firstError: Error?
        nonisolated(unsafe) var calls = 0
        nonisolated(unsafe) var bytes: UInt64 = 0
        let wallStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        DispatchQueue.concurrentPerform(iterations: plan.misses.count) { missOffset in
            let index = plan.misses[missOffset]
            do {
                let result = try self.loadExpertMeasured(
                    layer: 0,
                    expert: plan.experts[index],
                    slot: plan.assignedSlots[index])
                resultLock.lock()
                calls += result.stats.calls
                bytes &+= result.stats.bytes
                resultLock.unlock()
            } catch {
                resultLock.lock()
                if firstError == nil { firstError = error }
                resultLock.unlock()
            }
        }
        if let firstError { throw firstError }

        cacheLock.lock()
        for index in plan.misses {
            slotExpert[plan.assignedSlots[index]] = plan.experts[index]
        }
        cacheLock.unlock()

        return (expertCachePlanBuffers(plan),
                ExpertPreadStats(
                    calls: calls,
                    bytes: bytes,
                    nanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - wallStart))
    }

    public func expertCachePlanBuffers(_ plan: ExpertCachePlan)
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        precondition(plan.assignedSlots.count == plan.experts.count,
                     "expert cache plan slot count mismatch")
        if isWindowMapped { windowLock.lock() }
        defer { if isWindowMapped { windowLock.unlock() } }
        return plan.assignedSlots.map { slot in
            (slotBuffers[slot], slotOffsets[slot], layout.expertStride)
        }
    }

    /// Direct mappings have the invariant `slot == expert`.  Bypass the
    /// generic cache planner's lock, hit scan, usage accounting, and eviction
    /// sort on the wide-prefill rail.
    public func directMappedExpertBuffers(experts: [Int])
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        precondition(isDirectMapped, "direct expert lookup requires a mapped streamer")
        return experts.map { expert in
            precondition(expert >= 0 && expert < slotCount,
                         "mapped expert \(expert) is out of range")
            return (slotBuffers[expert], slotOffsets[expert], layout.expertStride)
        }
    }

    public func adviseExpertCachePlanMisses(_ plan: ExpertCachePlan) -> ExpertIOAdviceResult {
        let experts = plan.misses.map { plan.experts[$0] }
        return adviseRanges(expertAdviceRanges(experts: experts), requested: experts.count)
    }

    public func adviseExperts(experts: [Int]) -> ExpertIOAdviceResult {
        adviseRanges(expertAdviceRanges(experts: experts), requested: experts.count)
    }

    public func adviseExpertMisses(experts: [Int]) -> ExpertIOAdviceResult {
        cacheLock.lock()
        let misses = experts.filter { !slotExpert.contains($0) }
        cacheLock.unlock()
        return adviseRanges(expertAdviceRanges(experts: misses), requested: misses.count)
    }

    static func coalescedAdjacentAdviceRanges(_ ranges: [(offset: UInt64, count: UInt64)])
        -> [(offset: UInt64, count: UInt64)] {
        let sorted = ranges.filter { $0.count > 0 }.sorted {
            $0.offset == $1.offset ? $0.count < $1.count : $0.offset < $1.offset
        }
        var result: [(offset: UInt64, count: UInt64)] = []
        for range in sorted {
            guard var last = result.popLast() else {
                result.append(range)
                continue
            }
            let lastEnd = last.offset &+ last.count
            let rangeEnd = range.offset &+ range.count
            if range.offset <= lastEnd {
                last.count = max(lastEnd, rangeEnd) - last.offset
                result.append(last)
            } else {
                result.append(last)
                result.append(range)
            }
        }
        return result
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

    private func expertAdviceRanges(experts: [Int]) -> [(offset: UInt64, count: UInt64)] {
        experts.compactMap { expert in
            let regionOffset = layout.expertOffset(layer: 0, expert: expert)
            guard regionOffset + layout.expertStride <= layout.streamSize else { return nil }
            return (layout.streamOffset + regionOffset, layout.expertStride)
        }
    }

    private func adviseRanges(_ ranges: [(offset: UInt64, count: UInt64)],
                              requested: Int) -> ExpertIOAdviceResult {
        let coalesced = Self.coalescedAdjacentAdviceRanges(ranges)
        var failed = 0
        var bytes: UInt64 = 0
        var maxCallNanos: UInt64 = 0
        for range in coalesced {
            let result = RDAdvice.call(fd: fd, offset: range.offset, byteCount: range.count)
            if !result.succeeded { failed += 1 }
            bytes &+= result.requestedBytes
            maxCallNanos = max(maxCallNanos, result.elapsedNanos)
        }
        return ExpertIOAdviceResult(
            requested: requested,
            failed: failed,
            calls: coalesced.count,
            bytes: bytes,
            maxCallNanos: maxCallNanos)
    }

    private func readFull(into destination: UnsafeMutableRawPointer,
                          fileOffset: UInt64,
                          count: Int) throws -> ExpertPreadStats {
        var filled = 0
        var calls = 0
        let start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
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
            calls += 1
            if readCount < 0 {
                throw StreamerError.preadFailed(errno: errno)
            }
            if readCount == 0 {
                throw StreamerError.sizeMismatch(expected: UInt64(count), actual: UInt64(filled))
            }
            filled += readCount
        }
        return ExpertPreadStats(
            calls: calls,
            bytes: UInt64(filled),
            nanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start)
    }

    /// Model-derived CPU-side scratch owned by this streamer: its aligned
    /// slot allocation. Diagnostic metadata for residency reporting, not an
    /// ownership or lifetime API.
    public var diagnosticSlotScratchBytes: UInt64 {
        UInt64(slotCount) * UInt64(slotAllocationSize)
    }
}
