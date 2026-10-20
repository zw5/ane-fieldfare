import Testing
import Foundation
@testable import ANEFieldfareHotpath
@testable import ANEFieldfareImportedFormat
@testable import ANEFieldfareFormatTestSupport
import ANEFieldfareValidationSupport

@Suite struct ModelLoaderTests {
    /// Build a minimal valid `model.gturbo/` directory in a temp dir and
    /// return the URL. Uses the toy ArchConfig `gemma4Toy()`: 2 layers,
    /// 8 experts, hidden 64, vocab 1024. Resident contains the embedding
    /// (alias: lmHead), final norm, and the tiny layer-resident tensors needed
    /// to construct `RealForwardRunner` in unit tests.
    static func writeToySynthetic() throws -> URL {
        let toy = ArchConfig.gemma4Toy()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-toy-\(UUID().uuidString)")
        let exp = dir.appendingPathComponent("packed_experts")
        try FileManager.default.createDirectory(at: exp, withIntermediateDirectories: true)

        // 1. Resident region: embedding + final norm + runner-init tensors.
        struct ResidentSpec {
            let name: String
            let dtype: UInt8
            let shape: [UInt32]
            let weightBytes: UInt64
            let scaleBytes: UInt64
            let biasBytes: UInt64
        }

        let d = toy.hiddenSize
        let f = toy.intermediateSize

        func affine(_ name: String, rows: Int, cols: Int, bits: Int = 4) -> ResidentSpec {
            let groups = (cols + Quantization.groupSize - 1) / Quantization.groupSize
            let auxBytes = UInt64(rows * groups * MemoryLayout<UInt16>.stride)
            return ResidentSpec(name: name, dtype: 0,
                                shape: [UInt32(rows), UInt32(cols), 0, 0],
                                weightBytes: UInt64(rows * cols * bits / 8),
                                scaleBytes: auxBytes, biasBytes: auxBytes)
        }

        func bf16(_ name: String, count: Int) -> ResidentSpec {
            ResidentSpec(name: name, dtype: 1, shape: [UInt32(count), 0, 0, 0],
                         weightBytes: UInt64(count * MemoryLayout<UInt16>.stride),
                         scaleBytes: 0, biasBytes: 0)
        }

        func appendU16(_ values: [UInt16], to bytes: inout [UInt8]) {
            for value in values {
                bytes.append(UInt8(truncatingIfNeeded: value))
                bytes.append(UInt8(truncatingIfNeeded: value >> 8))
            }
        }

        func toyExpertBlob(expert: Int) -> (bytes: [UInt8], tensors: [String: [String: Any]]) {
            var bytes: [UInt8] = []
            var tensors: [String: [String: Any]] = [:]

            func addProjection(prefix: String, rows: Int, cols: Int, role: Int) {
                let quantized = (0..<rows).map { row in
                    Quantization.quantizeInt4Affine((0..<cols).map { col -> Float in
                        Float(expert + 1) * 0.001
                            + Float(role + 1) * 0.003
                            + Float((row % 7) - 3) * 0.0004
                            + Float((col % 11) - 5) * 0.0002
                    })
                }
                var offset = bytes.count
                for row in quantized { bytes.append(contentsOf: row.packed) }
                tensors[prefix] = [
                    "offset": offset, "size": bytes.count - offset,
                    "dtype": "U32", "shape": [rows, cols], "bits": 4,
                ]
                for (suffix, values) in [("scales", quantized.map(\.scales)),
                                         ("biases", quantized.map(\.biases))] {
                    offset = bytes.count
                    for row in values { appendU16(row, to: &bytes) }
                    tensors["\(prefix)_\(suffix)"] = [
                        "offset": offset, "size": bytes.count - offset,
                        "dtype": "BF16", "shape": [rows, cols / Quantization.groupSize],
                    ]
                }
            }

            addProjection(prefix: "gate", rows: toy.moeIntermediateSize, cols: d, role: 0)
            addProjection(prefix: "up", rows: toy.moeIntermediateSize, cols: d, role: 1)
            addProjection(prefix: "down", rows: d, cols: toy.moeIntermediateSize, role: 2)
            return (bytes, tensors)
        }

        var specs: [ResidentSpec] = [
            affine("language_model.model.embed_tokens.weight", rows: toy.vocabSize, cols: d),
            bf16("language_model.model.norm.weight", count: d),
        ]
        for L in 0..<toy.numLayers {
            let isFull = toy.fullAttentionLayerMask[L] != 0
            let headDim = isFull ? toy.fullHeadDim : toy.headDim
            let numKVHeads = isFull ? toy.numFullKVHeads : toy.numKVHeads
            let qDim = toy.numHeads * headDim
            let kvDim = numKVHeads * headDim
            let p = "language_model.model.layers.\(L)."
            specs += [
                bf16("\(p)input_layernorm.weight", count: d),
                affine("\(p)self_attn.q_proj.weight", rows: qDim, cols: d),
                affine("\(p)self_attn.k_proj.weight", rows: kvDim, cols: d),
                affine("\(p)self_attn.v_proj.weight", rows: kvDim, cols: d),
                affine("\(p)self_attn.o_proj.weight", rows: d, cols: qDim),
                bf16("\(p)post_attention_layernorm.weight", count: d),
                bf16("\(p)pre_feedforward_layernorm.weight", count: d),
                bf16("\(p)pre_feedforward_layernorm_2.weight", count: d),
                bf16("\(p)self_attn.q_norm.weight", count: headDim),
                bf16("\(p)self_attn.k_norm.weight", count: headDim),
                affine("\(p)mlp.gate_proj.weight", rows: f, cols: d),
                affine("\(p)mlp.up_proj.weight", rows: f, cols: d),
                affine("\(p)mlp.down_proj.weight", rows: d, cols: f),
                bf16("\(p)post_feedforward_layernorm_1.weight", count: d),
                bf16("\(p)post_feedforward_layernorm_2.weight", count: d),
                bf16("\(p)post_feedforward_layernorm.weight", count: d),
                bf16("\(p)layer_scalar", count: 1),
                bf16("\(p)router.scale", count: d),
                affine("\(p)router.proj.weight", rows: toy.numExperts, cols: d, bits: 8),
                bf16("\(p)router.per_expert_scale", count: toy.numExperts),
            ]
        }

        let names = specs.map(\.name)
        let stringTable = Array(names.joined().utf8)
        let entriesBase = GTurboBinary.indexHeaderBytes
        let entryBytes = GTurboBinary.indexEntryBytes
        let stringTableBase = entriesBase + names.count * entryBytes
        var nameAbsOffsets: [UInt32] = []
        var cursor = 0
        for n in names {
            nameAbsOffsets.append(UInt32(stringTableBase + cursor))
            cursor += n.utf8.count
        }
        let rawIndexBytes = UInt64(stringTableBase + stringTable.count)
        let alignment = GTurboFormatV1.alignmentBytes
        let indexBytes = ((rawIndexBytes + alignment - 1) / alignment) * alignment

        var entries: [ResidentEntry] = []
        var payloadCursor = indexBytes
        for spec in specs {
            let payloadAlignment: UInt64 = spec.dtype == 0
                ? UInt64(MemoryLayout<UInt32>.alignment)
                : UInt64(MemoryLayout<UInt16>.alignment)
            payloadCursor = ((payloadCursor + payloadAlignment - 1) / payloadAlignment)
                * payloadAlignment
            let weightOffset = payloadCursor
            let scaleOffset = spec.scaleBytes > 0 ? weightOffset + spec.weightBytes : 0
            let biasOffset = spec.biasBytes > 0 ? scaleOffset + spec.scaleBytes : 0
            entries.append(ResidentEntry(
                name: spec.name, dtype: spec.dtype, logicalShape4: spec.shape,
                fileOffset: weightOffset, sizeBytes: spec.weightBytes,
                scaleOffset: scaleOffset, scaleSize: spec.scaleBytes,
                biasOffset: biasOffset, biasSize: spec.biasBytes))
            payloadCursor += spec.weightBytes + spec.scaleBytes + spec.biasBytes
        }
        let residentSize = payloadCursor - indexBytes

        let totalBytes = Int(indexBytes + residentSize)
        var fileBuf = [UInt8](repeating: 0, count: totalBytes)
        fileBuf.replaceSubrange(stringTableBase..<(stringTableBase + stringTable.count),
                                with: stringTable)
        fileBuf.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress else { return }
            unsafe GTurboBinary.writeIndexHeader(into: base,
                                          indexSize: indexBytes,
                                          residentSize: residentSize,
                                          entryCount: UInt64(entries.count))
            for (i, e) in entries.enumerated() {
                let dst = unsafe base.advanced(by: entriesBase + i * entryBytes)
                unsafe GTurboBinary.writeIndexEntry(into: dst, entry: e,
                                             nameOffset: nameAbsOffsets[i])
            }
            // Recognizable resident payload pattern in the norm region only;
            // other payload bytes stay zero except router.scale unit BF16s.
            let normStart = Int(entries[1].fileOffset)
            for i in 0..<Int(entries[1].sizeBytes) {
                unsafe base.advanced(by: normStart + i)
                    .assumingMemoryBound(to: UInt8.self)[0] = UInt8(0xC0 | (i & 0x3F))
            }
            for entry in entries where entry.dtype == 0 {
                unsafe memset(base.advanced(by: Int(entry.fileOffset)), 0x11, Int(entry.sizeBytes))
                if entry.scaleSize > 0 {
                    let scales = unsafe base.advanced(by: Int(entry.scaleOffset))
                        .assumingMemoryBound(to: UInt16.self)
                    for i in 0..<(Int(entry.scaleSize) / MemoryLayout<UInt16>.stride) {
                        unsafe scales[i] = Quantization.bf16Bits(0.01)
                    }
                }
            }
            for entry in entries where entry.dtype == 1 && entry.name != "language_model.model.norm.weight" {
                let dst = unsafe base.advanced(by: Int(entry.fileOffset))
                    .assumingMemoryBound(to: UInt16.self)
                for i in 0..<(Int(entry.sizeBytes) / MemoryLayout<UInt16>.stride) {
                    unsafe dst[i] = Quantization.bf16Bits(1.0)
                }
            }
        }
        try Data(fileBuf).write(to: dir.appendingPathComponent("model_weights.bin"))

        // 2. packed_experts: 2 layer files, each `expertsPerLayer * expertStride`
        // bytes. expertStride must be a multiple of getpagesize() (16 KB).
        let expertStride: UInt64 = 16384
        let layerBytes = Int(expertStride) * toy.numExperts
        let blobs = (0..<toy.numExperts).map { toyExpertBlob(expert: $0) }
        for L in 0..<toy.numLayers {
            var payload = Data(count: layerBytes)
            for (E, blob) in blobs.enumerated() {
                let baseB = E * Int(expertStride)
                precondition(blob.bytes.count <= Int(expertStride),
                             "toy expert blob exceeds stride")
                payload.replaceSubrange(baseB..<(baseB + blob.bytes.count), with: blob.bytes)
                // Tag bytes so tests can identify which blob is being read.
                payload[baseB + 0] = UInt8(L)
                payload[baseB + 1] = UInt8(E)
                payload[baseB + 2] = 0xC1
                payload[baseB + 3] = 0xC2
            }
            try payload.write(to: exp.appendingPathComponent(unsafe String(format: "layer_%02d.bin", L)))
        }

        // 3. layout.json
        let layersArr: [[String: Any]] = (0..<toy.numLayers).map { L in
            [
                "layer": L,
                "file": unsafe String(format: "layer_%02d.bin", L),
                "experts": blobs.enumerated().map { E, blob -> [String: Any] in
                    ["expert": E, "offset": UInt64(E) * expertStride,
                     "size": expertStride, "tensors": blob.tensors]
                },
            ]
        }
        let layoutRoot: [String: Any] = [
            "expertStride": expertStride,
            "numLayers": toy.numLayers,
            "expertsPerLayer": toy.numExperts,
            "layers": layersArr,
        ]
        let layoutData = try JSONSerialization.data(
            withJSONObject: layoutRoot, options: [.sortedKeys])
        try layoutData.write(to: exp.appendingPathComponent("layout.json"))

        // 4. manifest.json
        var files: [String: [String: Any]] = [
            "model_weights.bin": ["size": totalBytes],
            "packed_experts/layout.json": ["size": layoutData.count],
        ]
        for L in 0..<toy.numLayers {
            files[unsafe String(format: "packed_experts/layer_%02d.bin", L)] = ["size": layerBytes]
        }

        let archDict: [String: Any] = [
            "hiddenSize": toy.hiddenSize, "ffnIntermediate": toy.intermediateSize,
            "moeIntermediateSize": toy.moeIntermediateSize,
            "numHeads": toy.numHeads, "numKVHeads": toy.numKVHeads,
            "numFullKVHeads": toy.numFullKVHeads,
            "headDim": toy.headDim, "fullHeadDim": toy.fullHeadDim,
            "vocabSize": toy.vocabSize, "slidingWindow": toy.slidingWindow,
            "finalLogitSoftcap": toy.finalLogitSoftcap,
            "ropeTheta": toy.ropeTheta, "fullRopeTheta": toy.fullRopeTheta,
            "partialRotaryFactor": toy.partialRotaryFactor,
            "numLayers": toy.numLayers, "numExperts": toy.numExperts,
            "topKExperts": toy.topKExperts,
            "tieWordEmbeddings": toy.tieWordEmbeddings,
            "attentionKEqV": toy.attentionKEqV,
            "hiddenActivation": toy.hiddenActivation,
            "fullAttentionLayerMask": toy.fullAttentionLayerMask.map { Int($0) },
        ]
        func quantSlot(_ weightBits: Int) -> [String: Any] {
            ["weightBits": weightBits, "scheme": "affine", "scaleType": "BF16",
             "biasType": "BF16", "groupSize": Quantization.groupSize]
        }
        let manifestRoot: [String: Any] = [
            "magic": "GTURBO",
            "versionMajor": 1,
            "versionMinor": 0,
            "flags": ["streamingPresent": true, "turboQuantKV": false, "aneSharedExpert": false],
            "modelID": "toy",
            "arch": archDict,
            "files": files,
            "expertsPerLayer": toy.numExperts,
            "numLayers": toy.numLayers,
            "expertStride": expertStride,
            "quant": [
                "embedding": quantSlot(4),
                "attention": quantSlot(4),
                "router": quantSlot(8),
                "sharedExpert": quantSlot(4),
                "routedExpert": quantSlot(4),
            ],
        ]
        let manifestData = try JSONSerialization.data(withJSONObject: manifestRoot,
            options: [.sortedKeys, .withoutEscapingSlashes])
        try manifestData.write(to: dir.appendingPathComponent("manifest.json"))
        return dir
    }
}
