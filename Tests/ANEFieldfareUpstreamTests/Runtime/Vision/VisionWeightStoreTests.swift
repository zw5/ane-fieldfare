import Foundation
import Metal
import Testing
@testable import ANEFieldfareHotpath
import ANEFieldfareModelFormat

@Suite struct VisionWeightStoreTests {
    private func open(_ dir: URL) throws -> VisionWeightStore {
        let directory = try ModelDirectory(rootURL: dir)
        let manifest = try ModelManifest.load(directory, path: dir.path, expecting: .gemma4Toy())
        return try VisionWeightStore.open(directory: directory, directoryURL: dir, manifest: manifest)
    }

    @Test func mapsTheVisionSectionOfTheModel() throws {
        let dir = try ModelLoaderTests.writeToySynthetic { manifest in
            manifest = Self.withVision(manifest, fileSize: 256)
        }
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data((0..<256).map { UInt8($0) }).write(to: dir.appendingPathComponent("vision.bin"))

        let store = try open(dir)
        #expect(store.tensorNames(withPrefix: "vision_tower.") == ["vision_tower.a", "vision_tower.b"])
        let mapped = try store.mapRegion(tensorNames: ["vision_tower.b"],
                                         device: try #require(MTLCreateSystemDefaultDevice()))
        // From b's weights at 64 through the end of its scales at 132.
        #expect(mapped.buffer.length == 68)
        #expect(try mapped.offset(of: "vision_tower.b") == 0)
        #expect(try mapped.scalesOffset(of: "vision_tower.b") == 64)
        #expect(unsafe mapped.buffer.contents().load(fromByteOffset: 0, as: UInt8.self) == 64)
    }

    @Test func reportsAModelInstalledWithoutVision() throws {
        let dir = try ModelLoaderTests.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try open(dir)
        } throws: { error in
            if case VisionPackError.notInstalled = error { return true }
            return false
        }
    }

    @Test func rejectsAVisionFileOfTheWrongSize() throws {
        let dir = try ModelLoaderTests.writeToySynthetic { manifest in
            manifest = Self.withVision(manifest, fileSize: 256)
        }
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data(count: 128).write(to: dir.appendingPathComponent("vision.bin"))
        #expect(throws: VisionPackError.self) { _ = try open(dir) }
    }

    /// `vision.bin`: a 32-value BF16 tensor at 0 and a Q4_0 `[2, 32]` tensor
    /// at 64 with its scales at 128.
    private static func withVision(_ manifest: FieldfareManifest, fileSize: UInt64) -> FieldfareManifest {
        var files = manifest.files
        files["vision.bin"] = FieldfareFile(size: fileSize, sha256: "")
        let vision = FieldfareTensorFile(file: "vision.bin", tensors: [
            FieldfareTensor(name: "vision_tower.a", encoding: .bf16, shape: [32],
                            weights: FieldfareRegion(offset: 0, size: 64)),
            FieldfareTensor(name: "vision_tower.b", encoding: .q4_0, shape: [2, 32],
                            weights: FieldfareRegion(offset: 64, size: 32),
                            scales: FieldfareRegion(offset: 128, size: 4)),
        ])
        return FieldfareManifest(modelID: manifest.modelID, source: manifest.source, arch: manifest.arch,
                                 files: files, resident: manifest.resident, experts: manifest.experts,
                                 vision: vision)
    }
}
