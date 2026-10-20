import Testing
import Foundation
import Metal
@testable import ANEFieldfareHotpath

@Suite struct ResidentBufferTests {
    @Test func unalignedFileOffsetStillExposesPayloadAtOffsetZero() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("fieldfare-resident-unaligned-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }

        // Offset that is not a multiple of getpagesize() — say 137.
        let preLen = 137
        let preamble = Data(repeating: 0x55, count: preLen)
        var payload = Data(count: 64)
        for i in 0..<payload.count { payload[i] = UInt8(0x80 | (i & 0x7F)) }
        try (preamble + payload).write(to: url)

        let resident = try ResidentBuffer(
            fileURL: url, fileOffset: UInt64(preLen),
            residentSize: UInt64(payload.count), device: device)
        for i in 0..<payload.count {
            let got = unsafe resident.buffer.contents().load(
                fromByteOffset: i, as: UInt8.self)
            #expect(got == UInt8(0x80 | (i & 0x7F)), "byte \(i)")
        }
    }
}
