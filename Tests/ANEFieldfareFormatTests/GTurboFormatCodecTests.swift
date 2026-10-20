import Foundation
import Testing
@testable import ANEFieldfareImportedFormat
import ANEFieldfareFormatTestSupport

private enum FormatFixture {
    static func manifest() -> GTurboManifestV1 {
        GTurboManifestV1(
            versionMinor: 0,
            flags: [
                "streamingPresent": true,
                "turboQuantKV": false,
                "aneSharedExpert": false,
            ],
            modelID: "fixture/model",
            sourceSnapshotHash: "snapshot",
            arch: GTurboManifestArchV1(
                hiddenSize: 64, ffnIntermediate: 128, moeIntermediateSize: 32,
                numHeads: 4, numKVHeads: 2, numFullKVHeads: 1,
                headDim: 16, fullHeadDim: 32, vocabSize: 1024,
                slidingWindow: 128, finalLogitSoftcap: 30,
                ropeTheta: 10_000, fullRopeTheta: 1_000_000,
                partialRotaryFactor: 0.25, numLayers: 1, numExperts: 2,
                topKExperts: 1, tieWordEmbeddings: true, attentionKEqV: true,
                hiddenActivation: "gelu_pytorch_tanh", fullAttentionLayerMask: [0]),
            quant: quant,
            files: [
                "model_weights.bin": GTurboManifestFileV1(size: 16_448),
                "packed_experts/layout.json": GTurboManifestFileV1(size: 1),
                "packed_experts/layer_00.bin": GTurboManifestFileV1(
                    size: 2 * GTurboFormatV1.alignmentBytes),
            ],
            expertsPerLayer: 2,
            numLayers: 1,
            expertStride: GTurboFormatV1.alignmentBytes)
    }

    static let quantSlot = GTurboManifestQuantSlotV1(
        weightBits: 4, scheme: "affine", scaleType: "BF16",
        biasType: "BF16", groupSize: 64)

    static let quant = GTurboManifestQuantV1(
        embedding: quantSlot, attention: quantSlot, router: quantSlot,
        sharedExpert: quantSlot, routedExpert: quantSlot)

    static func layout(explicitIDs: Bool = true,
                       explicitRanks: Bool = true) -> GTurboPackedExpertsLayoutV1 {
        let tensor = GTurboSubTensorV1(
            offset: 0, size: 32, dtype: "U32", shape: [8, 8], bits: 4)
        return GTurboPackedExpertsLayoutV1(
            expertStride: GTurboFormatV1.alignmentBytes,
            numLayers: 1,
            expertsPerLayer: 2,
            layers: [GTurboLayerV1(
                layer: 0,
                file: "layer_00.bin",
                experts: (0..<2).map { expert in
                    GTurboExpertV1(
                        expert: explicitIDs ? expert : nil,
                        physicalRank: explicitRanks ? expert : nil,
                        offset: UInt64(expert) * GTurboFormatV1.alignmentBytes,
                        size: GTurboFormatV1.alignmentBytes,
                        tensors: ["gate": tensor])
                })])
    }

    static let residentHeader = GTurboResidentIndexHeaderV1(
        indexSize: GTurboFormatV1.alignmentBytes, residentSize: 64, entryCount: 1)

    /// One-entry resident index named "weight" with scale and bias companions.
    static func residentBytes(fileOffset: UInt64 = GTurboFormatV1.alignmentBytes,
                              sizeBytes: UInt64 = 16,
                              scaleOffset: UInt64 = GTurboFormatV1.alignmentBytes + 16,
                              header: GTurboResidentIndexHeaderV1 = residentHeader) -> Data {
        let name = Array("weight".utf8)
        let nameOffset = 96
        var bytes = Data(repeating: 0, count: Int(header.indexSize))
        unsafe bytes.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress else { return }
            unsafe GTurboResidentIndexCodec.writeHeader(into: base, header: header)
            let entry = GTurboResidentIndexEntryV1(
                name: "weight", dtype: GTurboFormatV1.DType.u32.rawValue,
                fileOffset: fileOffset, sizeBytes: sizeBytes, shape: [1, 1, 0, 0],
                scaleOffset: scaleOffset, scaleSize: 8,
                biasOffset: GTurboFormatV1.alignmentBytes + 24, biasSize: 8)
            unsafe GTurboResidentIndexCodec.writeEntry(
                into: base.advanced(by: GTurboFormatV1.residentHeaderBytes),
                entry: entry, nameOffset: UInt32(nameOffset))
            for (index, byte) in name.enumerated() {
                unsafe raw[nameOffset + index] = byte
            }
        }
        return bytes
    }

    static func decodeResident(_ bytes: Data,
                               header: GTurboResidentIndexHeaderV1 = residentHeader) throws
        -> [GTurboResidentIndexEntryV1] {
        try unsafe bytes.withUnsafeBytes {
            try unsafe GTurboResidentIndexCodec.decodeRegion($0, header: header)
        }
    }
}

@Suite struct GTurboManifestCodecTests {
    @Test func roundTrips() throws {
        let manifest = FormatFixture.manifest()
        #expect(try GTurboManifestCodec.decode(GTurboManifestCodec.encode(manifest)) == manifest)
    }

    @Test(arguments: ["sourceSnapshotHash", "quant"])
    func acceptsAbsentLegacyOptionalField(_ key: String) throws {
        let data = try GTurboManifestCodec.encode(FormatFixture.manifest())
        var root = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        root.removeValue(forKey: key)
        let changed = try JSONSerialization.data(withJSONObject: root)
        let decoded = try GTurboManifestCodec.decode(changed)
        if key == "sourceSnapshotHash" { #expect(decoded.sourceSnapshotHash == nil) }
        if key == "quant" { #expect(decoded.quant == nil) }
    }

    @Test func rejectsUnsafeManifestPaths() throws {
        var root = try #require(JSONSerialization.jsonObject(
            with: GTurboManifestCodec.encode(FormatFixture.manifest())) as? [String: Any])
        var files = try #require(root["files"] as? [String: Any])
        files["../escape.bin"] = files.removeValue(forKey: "model_weights.bin")
        root["files"] = files
        let data = try JSONSerialization.data(withJSONObject: root)
        #expect(throws: GTurboFormatError.self) {
            try GTurboManifestCodec.validate(GTurboManifestCodec.decode(data))
        }
    }
}

@Suite struct GTurboPackedExpertsLayoutCodecTests {
    @Test func roundTripPreservesIdentityFallback() throws {
        let layout = FormatFixture.layout(explicitIDs: false, explicitRanks: false)
        let encoded = try GTurboPackedExpertsLayoutCodec.encode(layout)
        #expect(try GTurboPackedExpertsLayoutCodec.decode(encoded) == layout)
    }

    @Test func rejectsSlashBearingLayerBasename() throws {
        let base = FormatFixture.layout()
        let layout = GTurboPackedExpertsLayoutV1(
            expertStride: base.expertStride, numLayers: 1, expertsPerLayer: 2,
            layers: [GTurboLayerV1(
                layer: 0, file: "nested/layer.bin", experts: base.layers[0].experts)])
        #expect(throws: GTurboFormatError.self) {
            try GTurboV1StructuralValidator.validate(layout)
        }
    }

    @Test func crossValidationRequiresExactLayerManifestEntry() throws {
        var sizes = FormatFixture.manifest().files.mapValues(\.size)
        sizes["packed_experts/layer_00.bin"] = nil
        sizes["packed_experts/./layer_00.bin"] = 2 * GTurboFormatV1.alignmentBytes
        #expect(throws: GTurboFormatError.self) {
            try GTurboV1StructuralValidator.crossValidate(
                manifestNumLayers: 1, manifestExpertsPerLayer: 2,
                manifestExpertStride: GTurboFormatV1.alignmentBytes,
                manifestFileSizes: sizes, layout: FormatFixture.layout())
        }
    }

    @Test func rejectsTensorOutsideExpertBlob() throws {
        let stride = GTurboFormatV1.alignmentBytes
        let invalid = GTurboSubTensorV1(
            offset: stride - 8, size: 16, dtype: "U32", shape: [1], bits: 4)
        let expert = GTurboExpertV1(
            expert: 0, physicalRank: 0, offset: 0, size: stride,
            tensors: ["gate": invalid])
        let layout = GTurboPackedExpertsLayoutV1(
            expertStride: stride, numLayers: 1, expertsPerLayer: 1,
            layers: [GTurboLayerV1(layer: 0, file: "layer_00.bin", experts: [expert])])
        #expect(throws: GTurboFormatError.self) {
            try GTurboV1StructuralValidator.validate(layout)
        }
    }
}

@Suite struct GTurboResidentIndexCodecTests {
    @Test func decodesValidRegion() throws {
        let entries = try FormatFixture.decodeResident(FormatFixture.residentBytes())
        #expect(entries.map(\.name) == ["weight"])
    }

    @Test func rejectsOverlappingResidentPayloadRanges() {
        let bytes = FormatFixture.residentBytes(
            scaleOffset: GTurboFormatV1.alignmentBytes + 8)
        #expect(throws: GTurboFormatError.self) {
            try FormatFixture.decodeResident(bytes)
        }
    }

    @Test func rejectsEntryTableOverflow() {
        let header = GTurboResidentIndexHeaderV1(
            indexSize: GTurboFormatV1.alignmentBytes,
            residentSize: 64, entryCount: UInt64.max)
        let bytes = FormatFixture.residentBytes(header: header)
        #expect(throws: GTurboFormatError.self) {
            try FormatFixture.decodeResident(bytes, header: header)
        }
    }

    @Test func rejectsPayloadRangeOutsideResidentRegion() {
        let bytes = FormatFixture.residentBytes(
            fileOffset: GTurboFormatV1.alignmentBytes + 56, sizeBytes: 16)
        #expect(throws: GTurboFormatError.self) {
            try FormatFixture.decodeResident(bytes)
        }
    }
}
