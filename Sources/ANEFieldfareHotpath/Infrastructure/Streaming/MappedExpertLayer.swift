import Darwin
import Foundation
public import Metal
import Synchronization

package enum MappedExpertLayerError: Error, CustomStringConvertible {
    case mapFailed(path: String, errno: Int32)
    case sizeMismatch(expected: UInt64, actual: UInt64)
    case misaligned(String)
    case bufferWrapFailed(expert: Int)
    case expertOutOfRange(Int)
    case preadFailed(errno: Int32)
    case shortRead(expected: Int, actual: Int)

    public var description: String {
        switch self {
        case .mapFailed(let path, let error):
            return "mmap(\(path)) failed: errno \(error)"
        case .sizeMismatch(let expected, let actual):
            return "expert file size mismatch: expected \(expected), got \(actual)"
        case .misaligned(let detail):
            return "expert layout is not page aligned: \(detail)"
        case .bufferWrapFailed(let expert):
            return "failed to wrap mapped expert \(expert) in an MTLBuffer"
        case .expertOutOfRange(let expert):
            return "expert \(expert) is out of range"
        case .preadFailed(let error):
            return "pread failed: errno \(error)"
        case .shortRead(let expected, let actual):
            return "short expert read: expected \(expected) bytes, got \(actual)"
        }
    }
}

/// One routed-expert layer file, mapped read-only and served to Metal from the
/// OS page cache.
///
/// Every expert is a no-copy `MTLBuffer` over its own page-aligned slice of
/// the mapping, so a resident expert costs the GPU nothing to bind and the
/// page cache holds as many experts as free memory allows. Correctness never
/// depends on residency: the GPU faults in any page that is absent when a
/// command buffer runs, and `read` only makes that happen earlier and in
/// parallel.
public final class MappedExpertLayer: @unchecked Sendable {
    // SAFETY: every stored property is a `let`. The buffers alias a
    // `PROT_READ` mapping, so neither the CPU nor the GPU can write through
    // them, and `mapping` keeps that memory alive for as long as any buffer
    // does. Concurrent readers only bind buffers and issue reads.

    /// The file mapping and descriptor. Every expert buffer's deallocator
    /// holds it, so a buffer that outlives its layer (a caller's view, a
    /// command buffer in flight) never reads unmapped memory.
    @safe private final class Mapping: @unchecked Sendable {
        // SAFETY: immutable after init. `base` is unmapped only in `deinit`,
        // which runs after the last expert buffer's deallocator releases it.
        let fd: Int32
        let base: UnsafeMutableRawPointer
        let bytes: Int
        init(fd: Int32, base: UnsafeMutableRawPointer, bytes: Int) {
            self.fd = fd
            unsafe self.base = unsafe base
            self.bytes = bytes
        }
        deinit {
            unsafe munmap(base, bytes)
            close(fd)
        }
    }

    /// Each expert read is split into this many concurrent `pread`s; one
    /// read at queue depth 1 leaves most SSD bandwidth idle.
    public static let readSplit = 4

    public let expertCount: Int
    public let expertBytes: Int
    public let buffers: [any MTLBuffer]

    private let mapping: Mapping
    private var fd: Int32 { mapping.fd }
    private var base: UnsafeMutableRawPointer { unsafe mapping.base }
    private let expertOffsets: [Int]
    private let pageSize = Int(getpagesize())

    /// `fileDescriptor` is duplicated; the caller keeps ownership of its copy.
    public init(fileDescriptor: Int32,
                path: String,
                device: any MTLDevice,
                expertCount: Int,
                expertBytes: Int,
                expertOffsets: [UInt64]) throws {
        precondition(expertOffsets.count == expertCount, "one offset per expert")
        let page = Int(getpagesize())
        guard expertBytes > 0, expertBytes % page == 0 else {
            throw MappedExpertLayerError.misaligned("expert stride \(expertBytes), page \(page)")
        }
        let fd = fcntl(fileDescriptor, F_DUPFD_CLOEXEC, 0)
        guard fd >= 0 else { throw MappedExpertLayerError.mapFailed(path: path, errno: errno) }
        var closeOnFailure = true
        defer { if closeOnFailure { close(fd) } }

        var info = stat()
        guard unsafe fstat(fd, &info) == 0 else { throw MappedExpertLayerError.mapFailed(path: path, errno: errno) }
        let offsets = expertOffsets.map { Int($0) }
        let required = (offsets.max() ?? 0) + expertBytes
        guard Int(info.st_size) >= required else {
            throw MappedExpertLayerError.sizeMismatch(expected: UInt64(required), actual: UInt64(info.st_size))
        }
        if let bad = offsets.first(where: { $0 % page != 0 }) {
            throw MappedExpertLayerError.misaligned("expert offset \(bad), page \(page)")
        }
        let mapped = (required + page - 1) / page * page
        guard let raw = unsafe mmap(nil, mapped, PROT_READ, MAP_SHARED, fd, 0), unsafe raw != MAP_FAILED else {
            throw MappedExpertLayerError.mapFailed(path: path, errno: errno)
        }
        // From here the mapping owns the descriptor and unmaps on release.
        closeOnFailure = false
        let mapping = unsafe Mapping(fd: fd, base: raw, bytes: mapped)
        var buffers: [any MTLBuffer] = []
        buffers.reserveCapacity(expertCount)
        for (expert, offset) in offsets.enumerated() {
            guard let buffer = unsafe device.makeBuffer(bytesNoCopy: raw + offset, length: expertBytes,
                                                 options: .storageModeShared,
                                                 deallocator: { _, _ in withExtendedLifetime(mapping) {} }) else {
                throw MappedExpertLayerError.bufferWrapFailed(expert: expert)
            }
            buffer.label = "\((path as NSString).lastPathComponent).e\(expert)"
            buffers.append(buffer)
        }
        self.mapping = mapping
        self.expertCount = expertCount
        self.expertBytes = expertBytes
        self.expertOffsets = offsets
        self.buffers = buffers
    }

    private func checked(_ expert: Int) throws -> Int {
        guard expert >= 0, expert < expertCount else { throw MappedExpertLayerError.expertOutOfRange(expert) }
        return expertOffsets[expert]
    }

    /// Pages probed per residency check. `mincore` is paid per page and decode
    /// checks every routed expert of every layer. An expert's pages are read
    /// together and used together, so they age out together; a spread sample
    /// of single pages decides as well as all of them. A wrong "resident"
    /// only costs a GPU page fault.
    public static let residencyProbes = 8

    /// Whether `expert` is in memory or already being read in.
    public func isResident(_ expert: Int, probes: Int = residencyProbes) -> Bool {
        guard expert >= 0, expert < expertCount else { return false }
        let pages = expertBytes / pageSize
        let probes = min(probes, pages)
        var state: CChar = 0
        for probe in 0..<probes {
            let page = probe * (pages - 1) / max(1, probes - 1)
            guard unsafe mincore(base + expertOffsets[expert] + page * pageSize, pageSize, &state) == 0,
                  state & CChar(MINCORE_INCORE) != 0 else { return false }
        }
        return true
    }

    /// Starts reading `experts` into the page cache and returns at once.
    /// Storage serves them in the background; nothing waits for them.
    public func prefetch(_ experts: [Int]) {
        for expert in experts {
            guard let offset = try? checked(expert) else { continue }
            var advice = radvisory(ra_offset: off_t(offset), ra_count: Int32(expertBytes))
            _ = unsafe fcntl(fd, F_RDADVISE, &advice)
        }
    }

    /// Reads `experts` into the page cache with `readSplit` concurrent
    /// `pread`s each and returns when every byte has arrived.
    public func read(_ experts: [Int]) throws {
        guard !experts.isEmpty else { return }
        let offsets = try experts.map(checked)
        let part = (expertBytes / Self.readSplit + pageSize - 1) / pageSize * pageSize
        let parts = (expertBytes + part - 1) / part
        let firstError = Mutex<(any Error)?>(nil)
        // SAFETY: `concurrentPerform` fans blocking `pread`s out over GCD
        // workers and returns when all finish; it never runs on the Swift
        // concurrency pool.
        DispatchQueue.concurrentPerform(iterations: offsets.count * parts) { index in
            let start = (index % parts) * part
            let count = min(part, expertBytes - start)
            do {
                unsafe try ReadSink.with(bytes: part) { sink in
                    unsafe try Self.readFull(fd: fd, into: sink, offset: offsets[index / parts] + start, count: count)
                }
            } catch {
                firstError.withLock { if $0 == nil { $0 = error } }
            }
        }
        if let error = firstError.withLock({ $0 }) { throw error }
    }

    private static func readFull(fd: Int32, into sink: UnsafeMutableRawPointer, offset: Int, count: Int) throws {
        var filled = 0
        while filled < count {
            let n = unsafe pread(fd, sink, count - filled, off_t(offset + filled))
            if n < 0 {
                if errno == EINTR { continue }
                throw MappedExpertLayerError.preadFailed(errno: errno)
            }
            if n == 0 { throw MappedExpertLayerError.shortRead(expected: count, actual: filled) }
            filled += n
        }
    }
}

/// Reusable destinations for page-cache fills. The copy out of the page cache
/// is discarded; only the read's side effect, a resident page, matters.
private enum ReadSink {
    /// Each sink is held by one read at a time: it leaves the pool for the
    /// read and returns when the read is done.
    private static let free = Mutex<[[UInt8]]>([])

    static func with<T>(bytes: Int, _ body: (UnsafeMutableRawPointer) throws -> T) throws -> T {
        var sink = free.withLock { free in
            free.firstIndex { $0.count >= bytes }.map { free.remove(at: $0) }
        } ?? [UInt8](repeating: 0, count: bytes)
        defer { free.withLock { [sink] in $0.append(sink) } }
        return try sink.withUnsafeMutableBytes { buffer in
            guard let base = buffer.baseAddress else { throw MappedExpertLayerError.shortRead(expected: bytes, actual: 0) }
            return unsafe try body(base)
        }
    }
}
