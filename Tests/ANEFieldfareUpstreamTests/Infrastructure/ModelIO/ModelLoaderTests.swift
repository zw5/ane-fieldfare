import Testing
import Foundation
import CryptoKit
@testable import ANEFieldfareHotpath
import ANEFieldfareModelFormat

@Suite struct ModelLoaderTests {
    /// Build a minimal valid `.fieldfare/` directory in a temp dir and return
    /// its URL. Uses the toy ArchConfig `gemma4Toy()`: 2 layers, 8 experts,
    /// hidden 64, vocab 1024. The resident file holds every tensor the
    /// runtime schema requires; Q4_0 weights are 0x11 with scale 0.01, the
    /// final norm is patterned `0xC0 | (i & 0x3F)`, other BF16 tensors are 1.
    /// Each expert blob starts with the tag bytes (layer, expert, 0xC1, 0xC2).
    static func writeToySynthetic(mutate: (inout FieldfareManifest) -> Void = { _ in }) throws -> URL {
        let toy = ArchConfig.gemma4Toy()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("fieldfare-toy-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir.appendingPathComponent("experts"),
                                                withIntermediateDirectories: true)

        var resident = Data()
        var tensors: [FieldfareTensor] = []
        func region(_ bytes: Data) -> FieldfareRegion {
            while resident.count % Int(FieldfareFormat.regionAlignment) != 0 { resident.append(0) }
            let r = FieldfareRegion(offset: UInt64(resident.count), size: UInt64(bytes.count))
            resident.append(bytes)
            return r
        }
        func half(_ value: Float, count: Int) -> Data {
            var data = Data()
            for _ in 0..<count { withUnsafeBytes(of: Float16(value).bitPattern.littleEndian) { unsafe data.append(contentsOf: $0) } }
            return data
        }
        func q4(_ name: String, _ rows: Int, _ cols: Int) {
            tensors.append(FieldfareTensor(
                name: name, encoding: .q4_0, shape: [UInt64(rows), UInt64(cols)],
                weights: region(Data(repeating: 0x11, count: rows * cols / 2)),
                scales: region(half(0.01, count: rows * cols / 32))))
        }
        func bf16(_ name: String, _ count: Int, pattern: ((Int) -> UInt8)? = nil) {
            var data = Data()
            if let pattern {
                data = Data((0..<(2 * count)).map(pattern))
            } else {
                for _ in 0..<count { withUnsafeBytes(of: Quantization.bf16Bits(1).littleEndian) { unsafe data.append(contentsOf: $0) } }
            }
            tensors.append(FieldfareTensor(name: name, encoding: .bf16, shape: [UInt64(count)], weights: region(data)))
        }
        let d = toy.hiddenSize
        q4("embed_tokens.weight", toy.vocabSize, d)
        bf16("norm.weight", d) { UInt8(0xC0 | ($0 & 0x3F)) }
        for layer in 0..<toy.numLayers {
            let p = "layers.\(layer)."
            let full = toy.fullAttentionLayerMask[layer] != 0
            let hd = full ? toy.fullHeadDim : toy.headDim
            let kv = (full ? toy.numFullKVHeads : toy.numKVHeads) * hd, q = toy.numHeads * hd
            for name in ["input_layernorm.weight", "post_attention_layernorm.weight",
                         "pre_feedforward_layernorm.weight", "pre_feedforward_layernorm_2.weight",
                         "post_feedforward_layernorm_1.weight", "post_feedforward_layernorm_2.weight",
                         "post_feedforward_layernorm.weight", "router.scale"] { bf16(p + name, d) }
            bf16(p + "self_attn.q_norm.weight", hd)
            bf16(p + "self_attn.k_norm.weight", hd)
            bf16(p + "router.per_expert_scale", toy.numExperts)
            bf16(p + "layer_scalar", 1)
            q4(p + "self_attn.q_proj.weight", q, d)
            q4(p + "self_attn.k_proj.weight", kv, d)
            if !full { q4(p + "self_attn.v_proj.weight", kv, d) }
            q4(p + "self_attn.o_proj.weight", d, q)
            q4(p + "mlp.gate_proj.weight", toy.intermediateSize, d)
            q4(p + "mlp.up_proj.weight", toy.intermediateSize, d)
            q4(p + "mlp.down_proj.weight", d, toy.intermediateSize)
            tensors.append(FieldfareTensor(
                name: p + "router.proj.weight", encoding: .affine8, shape: [UInt64(toy.numExperts), UInt64(d)],
                weights: region(Data(repeating: 0x80, count: toy.numExperts * d)),
                scales: region(Data(count: toy.numExperts * d / 64 * 2)),
                biases: region(Data(count: toy.numExperts * d / 64 * 2))))
        }
        try resident.write(to: dir.appendingPathComponent("resident.bin"))

        // Expert blob: gate, up, down, each Q4_0 weights then scales.
        let stride: UInt64 = 16_384
        var blob: UInt64 = 0
        func expertTensor(_ name: String, _ rows: Int, _ cols: Int) -> FieldfareTensor {
            let w = FieldfareRegion(offset: blob, size: UInt64(rows * cols / 2))
            blob += (w.size + 63) / 64 * 64
            let s = FieldfareRegion(offset: blob, size: UInt64(rows * cols / 32 * 2))
            blob += (s.size + 63) / 64 * 64
            return FieldfareTensor(name: name, encoding: .q4_0, shape: [UInt64(rows), UInt64(cols)],
                                   weights: w, scales: s)
        }
        let f = toy.moeIntermediateSize
        let expertTensors = [expertTensor("gate", f, d), expertTensor("up", f, d), expertTensor("down", d, f)]
        precondition(blob <= stride, "toy expert blob exceeds stride")
        var expertFiles: [String] = []
        for layer in 0..<toy.numLayers {
            var payload = Data(count: Int(stride) * toy.numExperts)
            for e in 0..<toy.numExperts {
                let base = e * Int(stride)
                payload[base] = UInt8(layer)
                payload[base + 1] = UInt8(e)
                payload[base + 2] = 0xC1
                payload[base + 3] = 0xC2
            }
            let path = "experts/layer_0\(layer).bin"
            try payload.write(to: dir.appendingPathComponent(path))
            expertFiles.append(path)
        }

        var files: [String: FieldfareFile] = [:]
        for path in ["resident.bin"] + expertFiles {
            let data = try Data(contentsOf: dir.appendingPathComponent(path))
            files[path] = FieldfareFile(size: UInt64(data.count),
                                        sha256: SHA256.hash(data: data).map { unsafe String(format: "%02x", $0) }.joined())
        }
        var manifest = FieldfareManifest(
            modelID: "toy",
            source: FieldfareSource(files: []),
            arch: FieldfareArch(
                hiddenSize: toy.hiddenSize, intermediateSize: toy.intermediateSize,
                moeIntermediateSize: toy.moeIntermediateSize, numHeads: toy.numHeads,
                numKVHeads: toy.numKVHeads, numFullKVHeads: toy.numFullKVHeads,
                headDim: toy.headDim, fullHeadDim: toy.fullHeadDim, vocabSize: toy.vocabSize,
                slidingWindow: toy.slidingWindow, finalLogitSoftcap: toy.finalLogitSoftcap,
                ropeTheta: toy.ropeTheta, fullRopeTheta: toy.fullRopeTheta,
                partialRotaryFactor: toy.partialRotaryFactor, numLayers: toy.numLayers,
                numExperts: toy.numExperts, topKExperts: toy.topKExperts,
                tieWordEmbeddings: toy.tieWordEmbeddings, attentionKEqV: toy.attentionKEqV,
                hiddenActivation: toy.hiddenActivation,
                fullAttentionLayers: toy.fullAttentionLayerMask.indices.filter { toy.fullAttentionLayerMask[$0] != 0 }),
            files: files,
            resident: FieldfareTensorFile(file: "resident.bin", tensors: tensors),
            experts: FieldfareExperts(stride: stride, perLayer: toy.numExperts, files: expertFiles,
                                      tensors: expertTensors),
            vision: nil)
        mutate(&manifest)
        // Written without the codec's validation, so tests can build invalid manifests.
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(manifest).write(to: dir.appendingPathComponent("manifest.json"))
        return dir
    }
}

extension ArchConfig {
    /// Tiny baseline used across the loader tests: 2 layers (the second full
    /// attention), hidden 64, vocab 1024, 8 experts. Numbers are intentionally toy.
    static func gemma4Toy() -> ArchConfig {
        ArchConfig(
            hiddenSize: 64,
            intermediateSize: 256,
            moeIntermediateSize: 128,
            numHeads: 4,
            numKVHeads: 2,
            numFullKVHeads: 1,
            headDim: 16,
            fullHeadDim: 32,
            vocabSize: 1024,
            slidingWindow: 256,
            finalLogitSoftcap: 30.0,
            ropeTheta: 10_000.0,
            fullRopeTheta: 1_000_000.0,
            partialRotaryFactor: 0.25,
            numLayers: 2,
            numExperts: 8,
            topKExperts: 2,
            tieWordEmbeddings: true,
            attentionKEqV: true,
            fullAttentionLayerMask: [0, 1],
            hiddenActivation: "gelu_pytorch_tanh"
        )
    }
}
