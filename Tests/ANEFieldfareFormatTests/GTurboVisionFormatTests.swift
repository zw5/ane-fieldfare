import Foundation
import Testing
@testable import ANEFieldfareImportedFormat
import ANEFieldfareFormatTestSupport

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
            modelID: "fixture/model",
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
            compatibleTextSourceSnapshotHash: original.compatibleTextSourceSnapshotHash,
            files: original.files, tensors: original.tensors)
        #expect(throws: GTurboFormatError.self) {
            try GTurboVisionStructuralValidator.validate(changed)
        }
    }

    /// One manifest per rule, so each fails on its own check.
    @Test func rejectsMisalignedOverlappingAndOutOfFileRegions() {
        let align = GTurboVisionFormatV1.alignmentBytes
        func region(_ name: String, _ position: Int, offset: UInt64, elements: UInt64) -> GTurboVisionTensorRegionV1 {
            .init(name: name, executionPosition: position, offset: offset,
                  size: 2 * elements, shape: [elements], dtype: .bf16)
        }
        let placement = GTurboFormatError.invalid(field: "vision.tensor.offset",
                                                  reason: "misaligned or overlapping region")
        let cases: [([GTurboVisionTensorRegionV1], GTurboFormatError)] = [
            ([region("a", 0, offset: 1, elements: 1)], placement),
            ([region("a", 0, offset: 0, elements: align / 2 + 1), region("b", 1, offset: align, elements: 1)],
             placement),
            ([region("a", 0, offset: 0, elements: 1), region("b", 1, offset: 2 * align, elements: 1)],
             .invalid(field: "vision.tensor.range", reason: "region exceeds vision_weights.bin")),
        ]
        for (tensors, error) in cases {
            #expect(throws: error) {
                try GTurboVisionStructuralValidator.validate(VisionFormatFixture.manifest(tensors: tensors))
            }
        }
    }

    @Test func rejectsUnexpectedPayloadFile() {
        let original = VisionFormatFixture.manifest()
        var files = original.files
        files["extra.bin"] = .init(size: 1)
        let changed = GTurboVisionManifestV1(
            modelID: original.modelID,
            compatibleTextSourceSnapshotHash: original.compatibleTextSourceSnapshotHash,
            files: files, tensors: original.tensors)
        #expect(throws: GTurboFormatError.self) {
            try GTurboVisionStructuralValidator.validate(changed)
        }
    }
}
