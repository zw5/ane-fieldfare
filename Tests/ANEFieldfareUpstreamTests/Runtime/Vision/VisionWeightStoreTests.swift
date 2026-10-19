import Foundation
import Metal
import Testing
@testable import ANEFieldfareHotpath
@testable import ANEFieldfareImportedFormat

@Suite struct VisionWeightStoreTests {
    @Test func companionLocationPreservesTextModelStem() throws {
        let text = URL(fileURLWithPath: "/models/custom-name.gturbo")
        #expect(try VisionPackLocation.companionURL(forTextModel: text).path
                == "/models/custom-name.vision.gturbo")
        #expect(throws: VisionPackError.self) {
            try VisionPackLocation.companionURL(
                forTextModel: URL(fileURLWithPath: "/models/custom-name"))
        }
    }

    @Test func opensBoundedCompanion() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }

        let store = try VisionWeightStore.open(
            directoryURL: fixture.root,
            compatibleTextSourceSnapshotHash: "text-snapshot")
        #expect(store.manifest.modelID == "fixture/model")
        #expect(store.manifest.tensors.map(\.name) == ["vision.weight"])
        let mapped = try store.mapRegion(
            executionPosition: 0,
            device: try #require(MTLCreateSystemDefaultDevice()))
        #expect(mapped.residentSize == 2)
        #expect(mapped.mappedLength == 2)
    }

    @Test func rejectsWrongTextModel() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        #expect(throws: VisionPackError.self) {
            try VisionWeightStore.open(
                directoryURL: fixture.root,
                compatibleTextSourceSnapshotHash: "other")
        }
    }

    private func makeFixture() throws -> (root: URL, manifest: GTurboVisionManifestV1) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("vision-weight-store-\(UUID().uuidString).vision.gturbo")
        try FileManager.default.createDirectory(at: root,
                                                withIntermediateDirectories: false)

        let weights = Data(repeating: 0, count: Int(GTurboVisionFormatV1.alignmentBytes))
        let processor = Data("{}".utf8)
        let manifest = GTurboVisionManifestV1(
            modelID: "fixture/model", sourceRevision: "fixture-revision",
            compatibleTextSourceSnapshotHash: "text-snapshot",
            files: [
                GTurboVisionFormatV1.weightsFile: .init(size: UInt64(weights.count)),
                GTurboVisionFormatV1.processorFile: .init(size: UInt64(processor.count)),
            ],
            tensors: [.init(
                name: "vision.weight", executionPosition: 0,
                offset: 0, size: 2, shape: [1], dtype: .bf16)])

        try weights.write(to: root.appendingPathComponent(GTurboVisionFormatV1.weightsFile))
        try processor.write(to: root.appendingPathComponent(GTurboVisionFormatV1.processorFile))
        try GTurboVisionManifestCodec.encode(manifest).write(
            to: root.appendingPathComponent(GTurboVisionFormatV1.manifestFile))
        return (root, manifest)
    }
}
