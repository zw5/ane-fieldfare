import Foundation
import ANEFieldfareImportedFormat

/// A well-formed SHA-256 for fixture manifest file entries.
package let fixtureSHA = String(repeating: "a", count: 64)

/// One resident-index entry for synthetic `model_weights.bin` fixtures.
struct ResidentEntry: Sendable {
    let name: String
    /// dtype byte for IndexEntry: 0 = U32, 1 = BF16, 2 = FP16, 3 = FP32.
    let dtype: UInt8
    /// Logical shape after dequant (max rank 4; trailing zeros).
    let logicalShape4: [UInt32]
    /// File offset where the (packed) weight bytes start.
    let fileOffset: UInt64
    /// Size in bytes of the weight bytes.
    let sizeBytes: UInt64
    /// Offset where BF16 scales start (0 if none).
    let scaleOffset: UInt64
    let scaleSize: UInt64
    /// Offset where BF16 biases start (0 if none).
    let biasOffset: UInt64
    let biasSize: UInt64
}

/// Little-endian binary encoders for the resident `IndexHeader` and 72-byte
/// `IndexEntry` records.
enum GTurboBinary {
    static let indexHeaderBytes = GTurboFormatV1.residentHeaderBytes
    static let indexEntryBytes = GTurboFormatV1.residentEntryBytes

    /// Write `IndexHeader { indexSize, residentSize, entryCount }` (24 bytes, LE).
    static func writeIndexHeader(into buf: UnsafeMutableRawPointer,
                                        indexSize: UInt64,
                                        residentSize: UInt64,
                                        entryCount: UInt64) {
        unsafe GTurboResidentIndexCodec.writeHeader(
            into: buf,
            header: GTurboResidentIndexHeaderV1(
                indexSize: indexSize,
                residentSize: residentSize,
                entryCount: entryCount))
    }

    /// Write one `IndexEntry` (72 bytes, LE) at `dst`. See gturbo-format.md.
    static func writeIndexEntry(into dst: UnsafeMutableRawPointer,
                                       entry: ResidentEntry,
                                       nameOffset: UInt32) {
        unsafe GTurboResidentIndexCodec.writeEntry(
            into: dst,
            entry: GTurboResidentIndexEntryV1(
                name: entry.name,
                dtype: entry.dtype,
                fileOffset: entry.fileOffset,
                sizeBytes: entry.sizeBytes,
                shape: entry.logicalShape4,
                scaleOffset: entry.scaleOffset,
                scaleSize: entry.scaleSize,
                biasOffset: entry.biasOffset,
                biasSize: entry.biasSize),
            nameOffset: nameOffset)
    }
}
