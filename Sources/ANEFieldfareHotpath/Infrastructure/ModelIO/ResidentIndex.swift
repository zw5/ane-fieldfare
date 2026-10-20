import Foundation
import Darwin
import ANEFieldfareImportedFormat

/// On-disk header. `indexSize` is the full byte size of the leading index
/// region — it INCLUDES the 24-byte header itself, the entry table, the
/// string table, and the writer's 16 KB page padding. The resident tensor
/// region starts at file byte `indexSize`.
struct ResidentIndexHeader: Sendable, Equatable {
    let indexSize: UInt64
    let residentSize: UInt64
    let entryCount: UInt64
}

public struct ResidentIndexEntry: Sendable {
    public let name: String
    public let dtype: UInt8
    /// Absolute file offset of the packed weight bytes (≥ `indexSize`).
    public let fileOffset: UInt64
    public let sizeBytes: UInt64
    public let shape: (UInt32, UInt32, UInt32, UInt32)
    public let scaleOffset: UInt64
    public let scaleSize: UInt64
    public let biasOffset: UInt64
    public let biasSize: UInt64
}

/// Parsed resident index: header + name-keyed entries.
struct ResidentIndex: Sendable {
    let header: ResidentIndexHeader
    let entries: [String: ResidentIndexEntry]
}

enum ResidentIndexReader {
    static let defaultMaxBytes = GTurboFormatV1.residentIndexMaxBytes

    /// `pread` the header + index region out of `model_weights.bin`. The
    /// tensor data region (starting at byte `header.indexSize`) is **not**
    /// read here — that's the resident-buffer materialization job.
    package static func load(fileDescriptor fd: Int32,
                             displayPath: String,
                             maxBytes: UInt64 = defaultMaxBytes) throws -> ResidentIndex {
        let headerBytes = GTurboFormatV1.residentHeaderBytes
        var headerBuf = [UInt8](repeating: 0, count: headerBytes)
        try headerBuf.withUnsafeMutableBytes {
            try unsafe preadExactly(fd: fd, into: $0, offset: 0,
                             field: "IndexHeader")
        }
        let wireHeader: GTurboResidentIndexHeaderV1
        do {
            wireHeader = try headerBuf.withUnsafeBytes {
                try unsafe GTurboResidentIndexCodec.decodeHeader($0)
            }
        } catch {
            throw ModelError.indexCorrupt(detail: "\(error)")
        }
        guard wireHeader.indexSize <= maxBytes else {
            throw ModelError.indexCorrupt(
                detail: "indexSize \(wireHeader.indexSize) exceeds metadata cap \(maxBytes)")
        }
        guard wireHeader.indexSize >= UInt64(headerBytes) else {
            throw ModelError.indexCorrupt(detail: "indexSize is smaller than the header")
        }
        var st = stat()
        guard unsafe fstat(fd, &st) == 0, st.st_size >= 0 else {
            throw ModelError.posixFailed(call: "fstat(\(displayPath))", errno: errno)
        }
        guard wireHeader.indexSize <= UInt64(st.st_size),
              wireHeader.indexSize <= UInt64(Int.max) else {
            throw ModelError.indexCorrupt(detail: "index region exceeds file size")
        }

        // -- entire index region (header + entries + string table + padding)
        let regionLen = Int(wireHeader.indexSize)
        var indexBuf = [UInt8](repeating: 0, count: regionLen)
        indexBuf.replaceSubrange(0..<headerBytes, with: headerBuf)
        try indexBuf.withUnsafeMutableBytes { raw in
            try unsafe preadExactly(
                fd: fd,
                into: UnsafeMutableRawBufferPointer(
                    rebasing: raw[headerBytes..<regionLen]),
                offset: off_t(headerBytes),
                field: "index region")
        }

        let wireEntries: [GTurboResidentIndexEntryV1]
        do {
            wireEntries = try indexBuf.withUnsafeBytes {
                try unsafe GTurboResidentIndexCodec.decodeRegion($0, header: wireHeader)
            }
        } catch {
            throw ModelError.indexCorrupt(detail: "\(error)")
        }
        let header = ResidentIndexHeader(indexSize: wireHeader.indexSize,
                                         residentSize: wireHeader.residentSize,
                                         entryCount: wireHeader.entryCount)
        let entries = Dictionary(uniqueKeysWithValues: wireEntries.map { wire in
            let shape = wire.shape
            return (wire.name, ResidentIndexEntry(
                name: wire.name, dtype: wire.dtype,
                fileOffset: wire.fileOffset, sizeBytes: wire.sizeBytes,
                shape: (shape[0], shape[1], shape[2], shape[3]),
                scaleOffset: wire.scaleOffset, scaleSize: wire.scaleSize,
                biasOffset: wire.biasOffset, biasSize: wire.biasSize))
        })
        return ResidentIndex(header: header, entries: entries)
    }

    private static func preadExactly(fd: Int32,
                                     into buffer: UnsafeMutableRawBufferPointer,
                                     offset: off_t,
                                     field: String) throws {
        var total = 0
        while total < buffer.count {
            let count = unsafe pread(fd, buffer.baseAddress?.advanced(by: total),
                              buffer.count - total, offset + off_t(total))
            if count < 0, errno == EINTR { continue }
            guard count > 0 else {
                throw ModelError.indexCorrupt(
                    detail: "short read for \(field) (\(total)/\(buffer.count))")
            }
            total += count
        }
    }
}
