import Foundation

/// Per-worker scratch buffer with a hard upper bound. Calls beyond the bound
/// throw rather than silently grow — the M1a copy budget caps per-worker
/// staging well under 1 MB.
///
/// `@safe`: the scratch owns its allocation and frees it exactly once.
/// `ensure` and `zeroBuffer` are `@unsafe`: they return the raw buffer
/// itself, valid until the next `ensure` grows it or the scratch is released.
@safe
public final class BoundedScratch {
    public static let defaultLimitBytes: Int = 1_048_576 - 1  // strictly under 1 MB

    public let limitBytes: Int
    public private(set) var largestEverBytes: Int = 0

    private var buffer: UnsafeMutableRawBufferPointer

    public init(limitBytes: Int = BoundedScratch.defaultLimitBytes) {
        self.limitBytes = limitBytes
        // Page-aligned starting size for the all-zero buffer we keep around for
        // hashing gap regions. 16 KB is arm64 page size; reading zero bytes
        // straight from this static avoids touching the bigger working buffer.
        let initial = 16_384
        unsafe self.buffer = UnsafeMutableRawBufferPointer.allocate(byteCount: initial,
                                                             alignment: 16_384)
        unsafe self.buffer.initializeMemory(as: UInt8.self, repeating: 0)
        self.largestEverBytes = initial
    }

    deinit { unsafe buffer.deallocate() }

    /// Grow the scratch to at least `byteCount` if needed. Returns the
    /// guaranteed-zeroed buffer.
    @discardableResult
    @unsafe public func ensure(_ byteCount: Int) throws -> UnsafeMutableRawBufferPointer {
        if byteCount > limitBytes {
            throw unsafe RepackError.scratchExceeded(requested: byteCount, limit: limitBytes)
        }
        if unsafe byteCount <= buffer.count { return unsafe buffer }
        let newCount = unsafe max(byteCount, buffer.count * 2)
        let bounded = min(newCount, limitBytes)
        if bounded < byteCount {
            throw unsafe RepackError.scratchExceeded(requested: byteCount, limit: limitBytes)
        }
        let newBuf = UnsafeMutableRawBufferPointer.allocate(byteCount: bounded,
                                                            alignment: 16_384)
        unsafe newBuf.initializeMemory(as: UInt8.self, repeating: 0)
        unsafe buffer.deallocate()
        unsafe buffer = unsafe newBuf
        if bounded > largestEverBytes { largestEverBytes = bounded }
        return unsafe buffer
    }

    @unsafe public var zeroBuffer: UnsafeRawBufferPointer {
        unsafe UnsafeRawBufferPointer(start: buffer.baseAddress, count: buffer.count)
    }
}
