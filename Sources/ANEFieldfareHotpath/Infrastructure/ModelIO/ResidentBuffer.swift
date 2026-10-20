import Foundation
import Darwin
import Metal

/// `mmap`'d view of `model_weights.bin`'s tensor data region, wrapped in
/// one shared `MTLBuffer`. All resident `TensorView`s alias byte offsets
/// inside `buffer`.
final class ResidentBuffer {
    let buffer: any MTLBuffer

    /// `mmap` the page-aligned window covering `[fileOffset, fileOffset + residentSize)`
    /// inside the file at `fileURL`. The wrapped `MTLBuffer` starts at the
    /// (sub-page) offset within the mapping so the resident bytes start at
    /// byte 0 of the buffer.
    init(fileURL: URL,
                fileOffset: UInt64,
                residentSize: UInt64,
                device: any MTLDevice,
                fileDescriptor: Int32? = nil) throws {
        let pageSize = Int(getpagesize())

        let fd = unsafe fileDescriptor.map { fcntl($0, F_DUPFD_CLOEXEC, 0) }
            ?? open(fileURL.path, O_RDONLY | O_NONBLOCK | O_CLOEXEC)
        guard fd >= 0 else {
            throw ModelError.posixFailed(call: "open(\(fileURL.path))", errno: errno)
        }
        defer { close(fd) }

        var info = stat()
        guard unsafe fstat(fd, &info) == 0 else {
            throw ModelError.posixFailed(call: "fstat(\(fileURL.path))", errno: errno)
        }
        guard (info.st_mode & S_IFMT) == S_IFREG, info.st_size >= 0 else {
            throw ModelError.indexCorrupt(detail: "resident weights are not a regular file")
        }
        let (fileEnd, endOverflow) = fileOffset.addingReportingOverflow(residentSize)
        // fileEnd <= st_size bounds both the offset and the size by Int64.max.
        guard residentSize > 0, !endOverflow, fileEnd <= UInt64(info.st_size) else {
            throw ModelError.indexCorrupt(detail: "resident mapping exceeds the weights file")
        }

        let alignedOffset = (fileOffset / UInt64(pageSize)) * UInt64(pageSize)
        let sliceShift = Int(fileOffset - alignedOffset)
        let mappedLen = sliceShift + Int(residentSize)
        let mapped = unsafe mmap(nil, mappedLen, PROT_READ, MAP_PRIVATE,
                          fd, off_t(alignedOffset))
        guard let base = unsafe mapped, unsafe base != MAP_FAILED else {
            throw ModelError.posixFailed(call: "mmap", errno: errno)
        }

        _ = unsafe posix_madvise(base, mappedLen, POSIX_MADV_RANDOM)

        let sliceStart = unsafe base.advanced(by: sliceShift)

        // The deallocator gets the slice start back and recovers the mapping
        // base from it. Do NOT capture self here — that would create a retain
        // cycle through the MTLBuffer.
        guard let buf = unsafe device.makeBuffer(
            bytesNoCopy: sliceStart,
            length: Int(residentSize),
            options: .storageModeShared,
            deallocator: { sliceStart, _ in
                unsafe munmap(sliceStart.advanced(by: -sliceShift), mappedLen)
            }
        ) else {
            unsafe munmap(base, mappedLen)
            throw ModelError.residentBufferWrapFailed
        }

        self.buffer = buf
    }
}
