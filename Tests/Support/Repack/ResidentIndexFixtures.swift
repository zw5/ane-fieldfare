import Foundation
import ANEFieldfareImportedFormat

// Resident-index fixture types, kept from the upstream repack writer for the
// loader tests that build synthetic model_weights.bin files.

/// One physical tensor in a safetensors shard. Coordinates are absolute file
/// offsets so the writer can map and copy a single tensor without re-parsing
/// the shard header.
struct SourceTensor: Sendable, Hashable {
    enum Dtype: UInt8, Sendable, Hashable {
        case u32  = 0
        case bf16 = 1
        case fp16 = 2
        case fp32 = 3

        var elementBytes: Int {
            switch self { case .u32: 4; case .bf16: 2; case .fp16: 2; case .fp32: 4 }
        }
    }

    let name: String
    let shardPath: String
    let dtype: Dtype
    let shape: [UInt64]
    let absoluteOffset: UInt64
    let sizeBytes: UInt64
}

/// Bit-width override resolved from `config.json -> quantization`.
struct QuantSpec: Sendable, Hashable {
    let bits: Int
}

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
    /// Quantization spec (nil for unquantized scalars/norms).
    let quantSpec: QuantSpec?

    /// Source tensors that supply this entry's bytes.
    let sourceWeight: SourceTensor
    let sourceScales: SourceTensor?
    let sourceBiases: SourceTensor?
}

/// Little-endian binary encoders for the resident `IndexHeader` and 72-byte
/// `IndexEntry` records. Used by the resident writer and the
/// matching loader-side parsers; kept in one place so the on-disk layout
/// changes only here.
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
