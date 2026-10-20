import Foundation
import Metal

/// The tokens a decode most likely emits, as an ascending ID list the greedy
/// head scores in place of the full 262k-row vocabulary.
///
/// A guess from this list is only a guess: `HotHeadDecoder` checks every
/// guessed token against the full head before handing it out, so the list's
/// contents change speed, never output. It starts from a ranked list shipped
/// in `HotVocabulary/vocab-<vocab>.bin` (special and byte tokens, then tokens
/// by frequency in English chat and Python source) and grows at run time with
/// prompt tokens and every token the check finds outside it. The head rows are
/// read where they are, so the list costs 4 bytes per token.
///
/// File layout, little-endian: 8-byte magic `GFHOTV1\0`, UInt32 vocab size,
/// UInt32 count, then `count` UInt32 token IDs, most frequent first.
final class HotVocabulary {
    static let magic: [UInt8] = Array("GFHOTV1".utf8) + [0]
    /// IDs taken from the shipped list; the rest of `capacity` is for
    /// tokens added at run time.
    static let shippedCount = 56 * 1024
    static let capacity = 64 * 1024

    /// `capacity` UInt32 slots, the first `count` holding IDs in ascending
    /// order.
    let ids: any MTLBuffer
    private(set) var count = 0
    private var member: [Bool]

    /// nil when the runtime ships no list for a `vocab`-token vocabulary.
    static func shipped(vocab: Int) -> URL? {
        Bundle.module.url(forResource: "vocab-\(vocab)", withExtension: "bin",
                          subdirectory: "HotVocabulary")
    }

    /// Loads the ranked list at `url` and keeps its first `shippedCount` IDs.
    init(url: URL, vocab: Int, device: any MTLDevice, shipped: Int = HotVocabulary.shippedCount) throws {
        let data = try Data(contentsOf: url)
        guard data.count >= 16, Array(data.prefix(8)) == Self.magic else {
            throw ModelError.indexCorrupt(detail: "hot vocabulary \(url.lastPathComponent): bad magic")
        }
        let header = unsafe data.withUnsafeBytes { raw in
            unsafe (UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: 8, as: UInt32.self)),
                    UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: 12, as: UInt32.self)))
        }
        guard header.0 == UInt32(vocab), data.count == 16 + 4 * Int(header.1) else {
            throw ModelError.indexCorrupt(detail: "hot vocabulary \(url.lastPathComponent): header "
                + "\(header) does not match a \(vocab)-token head")
        }
        let ranked = unsafe data.withUnsafeBytes { raw in
            (0..<Int(header.1)).map {
                unsafe UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: 16 + 4 * $0, as: UInt32.self))
            }
        }
        self.ids = try device.newBuffer(length: Self.capacity * MemoryLayout<UInt32>.size,
                                        options: .storageModeShared)
        member = Array(repeating: false, count: vocab)
        insert(ranked.prefix(shipped))
    }

    func contains(_ token: UInt32) -> Bool { Int(token) < member.count && member[Int(token)] }

    /// Adds `tokens` not yet listed, up to `capacity`. Call only while no
    /// encoded guess is in flight: the list is re-sorted in place.
    func insert<S: Sequence<UInt32>>(_ tokens: S) {
        let list = unsafe ids.contents().bindMemory(to: UInt32.self, capacity: Self.capacity)
        let before = count
        for token in tokens where Int(token) < member.count && !member[Int(token)] {
            guard count < Self.capacity else { break }
            member[Int(token)] = true
            unsafe list[count] = token
            count += 1
        }
        // Ascending IDs keep the guess's reads in head order.
        if count > before {
            var sorted = unsafe UnsafeMutableBufferPointer(start: list, count: count)
            unsafe sorted.sort()
        }
    }
}
