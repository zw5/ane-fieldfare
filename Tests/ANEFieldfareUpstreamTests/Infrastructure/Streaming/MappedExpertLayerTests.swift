import Testing
import Foundation
import Darwin
import Metal
@testable import ANEFieldfareHotpath

/// The page-cache expert store over a synthetic layer file of tagged expert
/// blobs. No real Gemma weights.
@Suite struct MappedExpertLayerTests {
    static let pageSize = Int(getpagesize())
    /// One expert = 2 pages, every byte tagged so a view identifies its expert.
    static let expertStride = 2 * pageSize
    static let numExperts = 4
    /// Experts start one page in, to exercise non-zero offsets.
    static let headerBytes = pageSize

    static func tagByte(_ expert: Int) -> UInt8 { UInt8(0xA0 + expert) }
    static var offsets: [UInt64] { (0..<numExperts).map { UInt64(headerBytes + $0 * expertStride) } }

    static func writeSyntheticLayer() throws -> URL {
        var bytes = [UInt8](repeating: 0, count: headerBytes + numExperts * expertStride)
        for e in 0..<numExperts {
            let start = headerBytes + e * expertStride
            for i in 0..<expertStride { bytes[start + i] = tagByte(e) }
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mapped-expert-layer-\(UUID().uuidString).bin")
        try Data(bytes).write(to: url)
        return url
    }

    static func open(_ url: URL, device: MTLDevice, stride: Int = expertStride,
                     offsets: [UInt64] = offsets) throws -> MappedExpertLayer {
        let fd = Darwin.open(url.path, O_RDONLY)
        #expect(fd >= 0)
        defer { close(fd) }
        return try MappedExpertLayer(fileDescriptor: fd, path: url.path, device: device,
                                     expertCount: offsets.count, expertBytes: stride,
                                     expertOffsets: offsets)
    }

    @Test func eachBufferBindsItsExpertsBytes() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let layer = try Self.open(url, device: device)
        #expect(layer.buffers.count == Self.numExperts)
        for e in 0..<Self.numExperts {
            let buffer = layer.buffers[e]
            #expect(buffer.length == Self.expertStride)
            let bytes = UnsafeRawBufferPointer(start: buffer.contents(), count: buffer.length)
            #expect(bytes.allSatisfy { $0 == Self.tagByte(e) })
        }
    }

    @Test func readMakesExpertsResident() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let layer = try Self.open(url, device: device)
        try layer.read([0, 2])
        #expect(layer.isResident(0))
        #expect(layer.isResident(2))
        try layer.read([])
    }

    @Test func prefetchAcceptsAnyExpert() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let layer = try Self.open(url, device: device)
        layer.prefetch([1, 3, 99, -1])
        #expect(!layer.isResident(99))
        #expect(!layer.isResident(-1))
    }

    @Test func outOfRangeExpertReadThrows() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let layer = try Self.open(url, device: device)
        #expect(throws: ExpertStoreError.self) { try layer.read([Self.numExperts]) }
    }

    @Test func misalignedStrideOrOffsetThrows() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
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
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let offsets = Self.offsets + [UInt64(Self.headerBytes + Self.numExperts * Self.expertStride)]
        #expect(throws: ExpertStoreError.self) {
            _ = try Self.open(url, device: device, offsets: offsets)
        }
    }
}
