import Foundation
import Metal
import Testing

@testable import ANEFieldfareHotpath

extension ModelLoaderTests {
    @Test func acceptsRootSymlinkAndRetainsOpenedDirectory() throws {
        let target = try Self.writeToySynthetic()
        let alias = FileManager.default.temporaryDirectory
            .appendingPathComponent("fieldfare-root-\(UUID().uuidString)")
        let replacement = FileManager.default.temporaryDirectory
            .appendingPathComponent("fieldfare-replacement-\(UUID().uuidString)")
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
        _ = try model.routedExpertViews(layer: 0, experts: [0])[0]
    }

    /// A symlinked file or directory inside the model is refused when it is
    /// opened: the manifest and resident file at load, an expert layer on
    /// its first use.
    @Test(arguments: ["manifest.json", "resident.bin", "experts"])
    func rejectsSymlinkedInstallEntry(_ entry: String) throws {
        let dir = try Self.writeToySynthetic()
        let outside = FileManager.default.temporaryDirectory
            .appendingPathComponent("fieldfare-outside-\(UUID().uuidString)")
        defer {
            try? FileManager.default.removeItem(at: dir)
            try? FileManager.default.removeItem(at: outside)
        }
        let link = dir.appendingPathComponent(entry)
        try FileManager.default.moveItem(at: link, to: outside)
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: outside)
        let device = try #require(MTLCreateSystemDefaultDevice())

        #expect(throws: ModelError.self) {
            let model = try Model.load(directoryURL: dir, device: device, expecting: .gemma4Toy())
            _ = try model.routedExpertViews(layer: 0, experts: [0])
        }
    }

    @Test func rejectsRoutedLayerLeafSymlinkOnLazyOpen() throws {
        let dir = try Self.writeToySynthetic()
        let outside = FileManager.default.temporaryDirectory
            .appendingPathComponent("fieldfare-layer-\(UUID().uuidString).bin")
        defer {
            try? FileManager.default.removeItem(at: dir)
            try? FileManager.default.removeItem(at: outside)
        }
        let layer = dir.appendingPathComponent("experts/layer_00.bin")
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(directoryURL: dir, device: device, expecting: .gemma4Toy())
        try FileManager.default.moveItem(at: layer, to: outside)
        try FileManager.default.createSymbolicLink(at: layer, withDestinationURL: outside)

        #expect(throws: ModelError.self) {
            try model.routedExpertViews(layer: 0, experts: [0])[0]
        }
    }
}
