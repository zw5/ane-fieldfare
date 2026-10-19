import Foundation
import Metal
import Testing

@testable import ANEFieldfareHotpath

extension ModelLoaderTests {
    @Test func acceptsRootSymlinkAndRetainsOpenedDirectory() throws {
        let target = try Self.writeToySynthetic()
        let alias = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-root-\(UUID().uuidString)")
        let replacement = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-replacement-\(UUID().uuidString)")
        defer {
            try? FileManager.default.removeItem(at: alias)
            try? FileManager.default.removeItem(at: target)
            try? FileManager.default.removeItem(at: replacement)
        }
        try FileManager.default.createDirectory(at: replacement,
                                                withIntermediateDirectories: false)
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: target)
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(
            directoryURL: alias, device: device, expecting: .gemma4Toy())

        try FileManager.default.removeItem(at: alias)
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: replacement)
        _ = try model.routedExpert(layer: 0, expert: 0)
    }

    @Test func rejectsManifestLeafSymlink() throws {
        let dir = try Self.writeToySynthetic()
        let outside = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-manifest-\(UUID().uuidString).json")
        defer {
            try? FileManager.default.removeItem(at: dir)
            try? FileManager.default.removeItem(at: outside)
        }
        let manifest = dir.appendingPathComponent("manifest.json")
        try FileManager.default.moveItem(at: manifest, to: outside)
        try FileManager.default.createSymbolicLink(at: manifest, withDestinationURL: outside)
        let device = try #require(MTLCreateSystemDefaultDevice())

        #expect(throws: ModelError.self) {
            try Model.load(directoryURL: dir, device: device, expecting: .gemma4Toy())
        }
    }

    @Test func rejectsPackedExpertsParentSymlink() throws {
        let dir = try Self.writeToySynthetic()
        let outside = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-packed-\(UUID().uuidString)")
        defer {
            try? FileManager.default.removeItem(at: dir)
            try? FileManager.default.removeItem(at: outside)
        }
        let packed = dir.appendingPathComponent("packed_experts")
        try FileManager.default.moveItem(at: packed, to: outside)
        try FileManager.default.createSymbolicLink(at: packed, withDestinationURL: outside)
        let device = try #require(MTLCreateSystemDefaultDevice())

        #expect(throws: ModelError.self) {
            try Model.load(directoryURL: dir, device: device, expecting: .gemma4Toy())
        }
    }

    @Test func rejectsRoutedLayerLeafSymlinkOnLazyOpen() throws {
        let dir = try Self.writeToySynthetic()
        let outside = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-layer-\(UUID().uuidString).bin")
        defer {
            try? FileManager.default.removeItem(at: dir)
            try? FileManager.default.removeItem(at: outside)
        }
        let layer = dir.appendingPathComponent("packed_experts/layer_00.bin")
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(directoryURL: dir, device: device, expecting: .gemma4Toy())
        try FileManager.default.moveItem(at: layer, to: outside)
        try FileManager.default.createSymbolicLink(at: layer, withDestinationURL: outside)

        #expect(throws: ModelError.self) {
            try model.routedExpert(layer: 0, expert: 0)
        }
    }
}
