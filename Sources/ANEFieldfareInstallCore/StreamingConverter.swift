import CommonCrypto
package import Foundation

/// SHA-256 whose running state can be saved and restored, so a hash over a
/// 14 GB download survives an interrupted install.
package struct ResumableSHA256: Sendable {
    private var context = CC_SHA256_CTX()

    package init() { unsafe _ = CC_SHA256_Init(&context) }

    package mutating func update(_ bytes: ArraySlice<UInt8>) {
        bytes.withUnsafeBufferPointer { buffer in
            unsafe _ = CC_SHA256_Update(&context, buffer.baseAddress, CC_LONG(buffer.count))
        }
    }

    package func finalize() -> String {
        var copy = context
        var digest = [UInt8](repeating: 0, count: Int(CC_SHA256_DIGEST_LENGTH))
        unsafe _ = CC_SHA256_Final(&digest, &copy)
        return digest.map { ($0 < 16 ? "0" : "") + String($0, radix: 16) }.joined()
    }

    /// The raw context. Only restored on the machine that saved it.
    package var state: Data {
        var copy = context
        return unsafe Data(bytes: &copy, count: MemoryLayout<CC_SHA256_CTX>.size)
    }

    package init?(state: Data) {
        guard state.count == MemoryLayout<CC_SHA256_CTX>.size else { return nil }
        unsafe state.withUnsafeBytes { raw in
            withUnsafeMutableBytes(of: &context) { unsafe $0.copyMemory(from: raw) }
        }
    }
}

/// A point where every source byte before `offset` is hashed into `hash` and
/// every segment ending at or before it is written.
package struct ConversionCheckpoint: Sendable {
    package let offset: UInt64
    package let hash: ResumableSHA256
}

package enum ConversionError: Error, Equatable, CustomStringConvertible, Sendable {
    case outOfOrder(expected: UInt64, got: UInt64)
    case unexpectedContent(at: UInt64)
    case truncated(at: UInt64)

    package var description: String {
        switch self {
        case let .outOfOrder(expected, got): "source bytes out of order: expected \(expected), got \(got)"
        case let .unexpectedContent(at): "source bytes at \(at) differ from the expected table"
        case let .truncated(at): "source ended at \(at), inside a tensor"
        }
    }
}

/// Converts source bytes, fed strictly in file order, through a plan's
/// segments. Bytes outside every segment (header, padding) are hashed and
/// dropped. Output goes to `write(file, offset, bytes)`.
package struct StreamingConverter: Sendable {
    package typealias Write = (_ file: Int, _ offset: UInt64, _ bytes: [UInt8]) throws -> Void

    private let segments: [Segment]
    private var next: Int
    private var position: UInt64
    private var hash: ResumableSHA256
    private var pending: [UInt8] = []
    package private(set) var checkpoint: ConversionCheckpoint

    package init(segments: [Segment], resumingAt start: ConversionCheckpoint? = nil) {
        self.segments = segments
        let start = start ?? ConversionCheckpoint(offset: 0, hash: ResumableSHA256())
        self.position = start.offset
        self.hash = start.hash
        self.checkpoint = start
        // A checkpoint never falls inside a segment.
        self.next = segments.firstIndex { $0.sourceEnd > start.offset } ?? segments.count
    }

    package var offset: UInt64 { position }
    package var finished: Bool { next == segments.count }

    package mutating func consume(_ chunk: ArraySlice<UInt8>, at offset: UInt64, write: Write) throws {
        guard offset == position else { throw ConversionError.outOfOrder(expected: position, got: offset) }
        var cursor = chunk.startIndex
        while cursor < chunk.endIndex {
            let available = UInt64(chunk.endIndex - cursor)
            guard next < segments.count else {
                hash.update(chunk[cursor...])
                position += available
                return
            }
            let segment = segments[next]
            let take: UInt64
            if position < segment.source {
                take = min(available, segment.source - position)
            } else {
                take = min(available, segment.sourceEnd - position)
                pending.append(contentsOf: chunk[cursor..<(cursor + Int(take))])
            }
            hash.update(chunk[cursor..<(cursor + Int(take))])
            position += take
            cursor += Int(take)
            if position == segment.sourceEnd {
                try Self.convert(segment, pending[...], write: write)
                pending.removeAll(keepingCapacity: true)
                next += 1
                checkpoint = ConversionCheckpoint(offset: position, hash: hash)
            }
        }
    }

    /// The source's SHA-256 after its last byte; throws if a segment is unfinished.
    package func finish(sourceSize: UInt64) throws -> String {
        guard position == sourceSize, finished else { throw ConversionError.truncated(at: position) }
        return hash.finalize()
    }

    static func convert(_ segment: Segment, _ rows: ArraySlice<UInt8>, write: Write) throws {
        var weights: [UInt8] = [], scales: [UInt8] = [], biases: [UInt8] = []
        switch segment.operation {
        case let .repackQ4_0(cols):
            RowCodec.repackQ4_0(rows, cols: cols, weights: &weights, scales: &scales)
        case let .q6KToQ4_0(cols):
            RowCodec.q6KToQ4_0(rows, cols: cols, weights: &weights, scales: &scales)
        case .f32ToBF16:
            RowCodec.f32ToBF16(rows, out: &weights)
        case let .f32ToAffine8(cols):
            RowCodec.f32ToAffine8(rows, cols: cols, weights: &weights, scales: &scales, biases: &biases)
        case .copy:
            weights.append(contentsOf: rows)
        case let .bf16ToQ4_0(cols):
            RowCodec.bf16ToQ4_0(rows, cols: cols, weights: &weights, scales: &scales)
        case let .f32PatchToBF16(channels, size):
            RowCodec.f32PatchToBF16(rows, channels: channels, size: size, out: &weights)
        case let .expect(bytes):
            guard rows.elementsEqual(bytes) else { throw ConversionError.unexpectedContent(at: segment.source) }
        }
        for (output, bytes) in zip(segment.outputs, [weights, scales, biases]) {
            try write(output.file, output.offset, bytes)
        }
    }
}
