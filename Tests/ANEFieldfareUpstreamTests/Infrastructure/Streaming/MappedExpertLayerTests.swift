import Testing
import Foundation
import Darwin
import Metal
@testable import ANEFieldfareHotpath

/// Bounds checks of the page-cache expert store over a synthetic layer file.
/// Byte binding is covered by StreamingKernelIntegrationTests.
@Suite struct MappedExpertLayerTests {
    static let pageSize = Int(getpagesize())
    /// One expert = 2 pages.
    static let expertStride = 2 * pageSize
    static let numExperts = 4
    /// Experts start one page in, to exercise non-zero offsets.
    static let headerBytes = pageSize

    static var offsets: [UInt64] { (0..<numExperts).map { UInt64(headerBytes + $0 * expertStride) } }

    static func writeSyntheticLayer() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mapped-expert-layer-\(UUID().uuidString).bin")
        try Data(count: headerBytes + numExperts * expertStride).write(to: url)
        return url
    }

    static func open(_ url: URL, device: any MTLDevice, stride: Int = expertStride,
                     offsets: [UInt64] = offsets) throws -> MappedExpertLayer {
        let fd = unsafe Darwin.open(url.path, O_RDONLY)
        #expect(fd >= 0)
        defer { close(fd) }
        return try MappedExpertLayer(fileDescriptor: fd, path: url.path, device: device,
                                     expertCount: offsets.count, expertBytes: stride,
                                     expertOffsets: offsets)
    }

    @Test func prefetchAcceptsAnyExpert() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let layer = try Self.open(url, device: device)
        layer.prefetch([1, 3, 99, -1])
        #expect(!layer.isResident(99))
        #expect(!layer.isResident(-1))
    }

    @Test func outOfRangeExpertReadThrows() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let layer = try Self.open(url, device: device)
        #expect(throws: ExpertStoreError.self) { try layer.read([Self.numExperts]) }
    }

    @Test func misalignedStrideOrOffsetThrows() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: ExpertStoreError.self) {
            _ = try Self.open(url, device: device, stride: Self.expertStride - 1)
        }
        #expect(throws: ExpertStoreError.self) {
            _ = try Self.open(url, device: device, offsets: Self.offsets.map { $0 + 1 })
        }
    }

    @Test func fileShorterThanLayoutThrows() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let offsets = Self.offsets + [UInt64(Self.headerBytes + Self.numExperts * Self.expertStride)]
        #expect(throws: ExpertStoreError.self) {
            _ = try Self.open(url, device: device, offsets: offsets)
        }
    }
}
