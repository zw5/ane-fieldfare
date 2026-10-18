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

/// A slot-sized buffer outside every slot pool, which a routed expert is read
/// into before its layer is planned. Adopting it into a plan swaps it with the
/// slot the plan assigned to that expert, so afterwards this object holds the
/// buffer that left the slot. Every slot buffer frees its own memory, so a
/// buffer may move between layers' streamers.
public final class ExpertStagingBuffer: @unchecked Sendable {
    public fileprivate(set) var pointer: UnsafeMutableRawPointer
    public fileprivate(set) var buffer: MTLBuffer

    fileprivate init(pointer: UnsafeMutableRawPointer, buffer: MTLBuffer) {
        self.pointer = pointer
        self.buffer = buffer
    }
}

public enum ExpertCachePolicy: String, Sendable {
    case lru
    case lfu
}

/// `pread`-based routed-expert streamer with a fixed per-layer slot cache.
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
    // Mutated only by `adoptStagedExpert`, under `cacheLock`.
    private var slotPointers: [UnsafeMutableRawPointer]
    private var slotBuffers: [MTLBuffer]
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
            guard let buffer = Self.wrapSlot(pointer, length: allocationSize, device: device) else {
                unwind()
                throw StreamerError.bufferWrapFailed
            }
            buffers.append(buffer)
        }

        self.slotPointers = pointers
        self.slotBuffers = buffers
        self.slotExpert = [Int](repeating: -1, count: slotCount)
        self.slotLastUse = [Int](repeating: 0, count: slotCount)
        self.expertUseCount = [Int](repeating: 0, count: max(1, layout.expertsPerLayer))
        closeFDOnFailure = false
    }

    deinit {
        close(fd)
    }

    /// The buffer owns `pointer` from here on and frees it when released.
    private static func wrapSlot(_ pointer: UnsafeMutableRawPointer,
                                 length: Int,
                                 device: MTLDevice) -> MTLBuffer? {
        nonisolated(unsafe) let capturedPointer = pointer
        return device.makeBuffer(
            bytesNoCopy: pointer,
            length: length,
            options: .storageModeShared,
            deallocator: { _, _ in free(capturedPointer) })
    }

    /// A buffer allocated exactly like one of this streamer's slots.
    public func makeStagingBuffer() throws -> ExpertStagingBuffer {
        var raw: UnsafeMutableRawPointer?
        let result = posix_memalign(&raw, Self.scratchAlignment, slotAllocationSize)
        guard result == 0, let pointer = raw else {
            throw StreamerError.allocFailed(errno: result)
        }
        guard let buffer = Self.wrapSlot(pointer, length: slotAllocationSize, device: device) else {
            free(pointer)
            throw StreamerError.bufferWrapFailed
        }
        return ExpertStagingBuffer(pointer: pointer, buffer: buffer)
    }

    /// Experts currently held in a slot.
    public func residentExperts() -> Set<Int> {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        return Set(slotExpert.filter { $0 >= 0 })
    }

    /// Reads one expert into memory outside the slot pool, normally a staging
    /// buffer's `pointer` captured before any adoption could swap it.
    public func readEarly(expert: Int, into destination: UnsafeMutableRawPointer) throws {
        let regionOffset = layout.expertOffset(layer: 0, expert: expert)
        guard regionOffset + layout.expertStride <= layout.streamSize else {
            throw StreamerError.offsetOutOfRange(regionOffset)
        }
        try readSplitting(into: destination, fileOffset: layout.streamOffset + regionOffset,
                          count: Int(layout.expertStride))
    }

    /// Puts `staging`'s buffer in the slot `plan` assigned to its miss at
    /// `index`, and leaves the slot's previous buffer in `staging`. Call it
    /// before the plan's buffers are bound, and execute the plan skipping
    /// `index`. The previous buffer was last used for an expert the plan
    /// evicted, the same condition under which a miss is read into it.
    public func adoptStagedExpert(_ staging: ExpertStagingBuffer, plan: ExpertCachePlan, index: Int) {
        precondition(plan.misses.contains(index), "only a planned miss can adopt a staged read")
        precondition(staging.buffer.length == slotAllocationSize,
                     "staging buffer size differs from this streamer's slots")
        let slot = plan.assignedSlots[index]
        cacheLock.lock()
        swap(&slotPointers[slot], &staging.pointer)
        swap(&slotBuffers[slot], &staging.buffer)
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
        try executeExpertCachePlan(plan, skippingMisses: [], beforeAdmitting: {})
    }

    /// Reads the plan's misses except `skippingMisses`, whose bytes arrive by
    /// another route (an adopted staged read). `beforeAdmitting` runs after
    /// this plan's reads and before any miss is marked resident; it must not
    /// return until the skipped misses' bytes are in place.
    public func executeExpertCachePlan(_ plan: ExpertCachePlan,
                                       skippingMisses: Set<Int>,
                                       beforeAdmitting: () throws -> Void) throws
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        precondition(plan.experts.count <= slotCount,
                     "expert cache plan exceeds slot count")
        precondition(plan.assignedSlots.count == plan.experts.count,
                     "expert cache plan slot count mismatch")

        let toRead = plan.misses.filter { !skippingMisses.contains($0) }
        let errorLock = NSLock()
        nonisolated(unsafe) var firstError: Error?
        DispatchQueue.concurrentPerform(iterations: toRead.count) { missOffset in
            let index = toRead[missOffset]
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
        try beforeAdmitting()

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
        cacheLock.lock()
        let buffers = slotBuffers
        cacheLock.unlock()
        return plan.assignedSlots.map { slot in
            (buffers[slot], 0, layout.expertStride)
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
