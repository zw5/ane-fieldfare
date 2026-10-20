import Testing
import Foundation
@testable import ANEFieldfareHotpath
@testable import ANEFieldfareRepackTestSupport

/// Writer-encoded round trips are covered end to end by the
/// `ModelLoaderTests.writeToySynthetic` load tests.
@Suite struct ResidentIndexTests {
    @Test func rejectsIndexRegionAboveMetadataCapBeforeAllocation() throws {
        let claimed = ResidentIndexReader.defaultMaxBytes + 1
        var data = Data(repeating: 0, count: GTurboBinary.indexHeaderBytes)
        unsafe data.withUnsafeMutableBytes {
            guard let base = $0.baseAddress else { return }
            unsafe GTurboBinary.writeIndexHeader(
                into: base, indexSize: claimed,
                residentSize: 0, entryCount: 0)
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-index-over-cap-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }
        try data.write(to: url)
        let fd = unsafe open(url.path, O_RDONLY | O_CLOEXEC)
        try #require(fd >= 0)
        defer { close(fd) }
        #expect {
            try ResidentIndexReader.load(fileDescriptor: fd, displayPath: url.path)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("metadata cap")
        }
    }
}
