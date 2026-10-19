import Foundation
import Testing
@testable import ANEFieldfareImportedFormat

private enum VisionFormatFixture {
    static func manifest(tensors: [GTurboVisionTensorRegionV1]? = nil)
        -> GTurboVisionManifestV1 {
        let defaultTensors = [
            GTurboVisionTensorRegionV1(
                name: "vision_tower.patch_embedder.input_proj.weight",
                executionPosition: 0, offset: 0, size: 512,
                shape: [16, 16], dtype: .bf16),
            GTurboVisionTensorRegionV1(
                name: "embed_vision.embedding_projection.weight",
                executionPosition: 1, offset: GTurboVisionFormatV1.alignmentBytes,
                size: 256, shape: [8, 8], dtype: .u32,
                quantization: GTurboVisionQuantizationV1(
                    weightBits: 4, scheme: "affine", scaleType: "bf16",
                    biasType: "bf16", groupSize: 64)),
        ]
        return GTurboVisionManifestV1(
            modelID: "fixture/model", sourceRevision: "fixture-revision",
            compatibleTextSourceSnapshotHash: "fixture-text-snapshot",
            files: [
                GTurboVisionFormatV1.weightsFile: GTurboManifestFileV1(
                    size: 2 * GTurboVisionFormatV1.alignmentBytes),
                GTurboVisionFormatV1.processorFile: GTurboManifestFileV1(size: 2),
            ],
            tensors: tensors ?? defaultTensors)
    }
}

@Suite struct GTurboVisionFormatCompatibilityTests {
    @Test func manifestRoundTrips() throws {
        let manifest = VisionFormatFixture.manifest()
        let manifestData = try GTurboVisionManifestCodec.encode(manifest)
        #expect(try GTurboVisionManifestCodec.decode(manifestData) == manifest)
    }

    @Test func rejectsUnknownVersion() {
        let original = VisionFormatFixture.manifest()
        let changed = GTurboVisionManifestV1(
            versionMajor: 2, modelID: original.modelID,
            sourceRevision: original.sourceRevision,
            compatibleTextSourceSnapshotHash: original.compatibleTextSourceSnapshotHash,
            files: original.files, tensors: original.tensors)
        #expect(throws: GTurboFormatError.self) {
            try GTurboVisionManifestCodec.encode(changed)
        }
    }

    @Test func rejectsReorderedRegions() {
        let original = VisionFormatFixture.manifest().tensors
        let changed = [
            GTurboVisionTensorRegionV1(
                name: original[0].name, executionPosition: 1,
                offset: original[0].offset, size: original[0].size,
                shape: original[0].shape, dtype: original[0].dtype),
            original[1],
        ]
        #expect(throws: GTurboFormatError.self) {
            try GTurboVisionManifestCodec.encode(
                VisionFormatFixture.manifest(tensors: changed))
        }
    }

    @Test func rejectsMisalignedOverlappingAndOutOfFileRegions() {
        let invalid: [GTurboVisionTensorRegionV1] = [
            .init(name: "a", executionPosition: 0, offset: 1,
                  size: 2, shape: [1], dtype: .bf16),
            .init(name: "b", executionPosition: 1,
                  offset: 2 * GTurboVisionFormatV1.alignmentBytes,
                  size: 2, shape: [1], dtype: .bf16),
        ]
        #expect(throws: GTurboFormatError.self) {
            try GTurboVisionManifestCodec.encode(
                VisionFormatFixture.manifest(tensors: invalid))
        }
    }

    @Test func rejectsInvalidStorageSizeAndQuantization() {
        let badSize = [GTurboVisionTensorRegionV1(
            name: "a", executionPosition: 0, offset: 0,
            size: 3, shape: [1], dtype: .bf16)]
        #expect(throws: GTurboFormatError.self) {
            try GTurboVisionManifestCodec.encode(
                VisionFormatFixture.manifest(tensors: badSize))
        }

        let badQuant = [GTurboVisionTensorRegionV1(
            name: "a", executionPosition: 0, offset: 0,
            size: 4, shape: [1], dtype: .u32)]
        #expect(throws: GTurboFormatError.self) {
            try GTurboVisionManifestCodec.encode(
                VisionFormatFixture.manifest(tensors: badQuant))
        }
    }

    @Test func rejectsUnexpectedPayloadFile() {
        let original = VisionFormatFixture.manifest()
        var files = original.files
        files["extra.bin"] = .init(size: 1)
        let changed = GTurboVisionManifestV1(
            modelID: original.modelID, sourceRevision: original.sourceRevision,
            compatibleTextSourceSnapshotHash: original.compatibleTextSourceSnapshotHash,
            files: files, tensors: original.tensors)
        #expect(throws: GTurboFormatError.self) {
            try GTurboVisionManifestCodec.encode(changed)
        }
    }
}
