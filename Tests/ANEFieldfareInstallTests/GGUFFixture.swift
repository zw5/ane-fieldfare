import Foundation
@testable import ANEFieldfareInstallCore

/// Writes a small GGUF with the tensor names, types and metadata of Gemma 4
/// 26B-A4B's QAT release: two layers (the second full attention, without
/// v_proj), four experts, hidden size 256.
enum GGUFFixture {
    static let hidden = 256, shared = 128, moe = 64, heads = 2, headDim = 64, fullHeadDim = 128
    static let layers = 2, experts = 4, vocab = 64
    static let fullLayers: Set<Int> = [1]

    struct Tensor {
        let name: String
        let dims: [UInt64]
        let type: GGUFHeader.TensorType
        let data: [UInt8]
    }

    /// Deterministic bytes from a 64-bit LCG.
    struct Bytes {
        var state: UInt64
        mutating func next() -> UInt8 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return UInt8(truncatingIfNeeded: state >> 33)
        }
        mutating func half(_ magnitude: Float) -> [UInt8] {
            let value = Float16(magnitude * (Float(next()) / 255 + 0.5))
            return [UInt8(truncatingIfNeeded: value.bitPattern), UInt8(truncatingIfNeeded: value.bitPattern >> 8)]
        }
        mutating func f32(_ value: Float) -> [UInt8] {
            (0..<4).map { UInt8(truncatingIfNeeded: value.bitPattern >> (8 * $0)) }
        }
    }

    static func q4_0(_ name: String, cols: Int, rows: Int, extra: [UInt64] = [], bytes: inout Bytes) -> Tensor {
        var data: [UInt8] = []
        let blocks = cols / 32 * rows * Int(extra.reduce(1, *))
        for _ in 0..<blocks {
            data += bytes.half(0.01)
            for _ in 0..<16 { data.append(bytes.next()) }
        }
        return Tensor(name: name, dims: [UInt64(cols), UInt64(rows)] + extra, type: .q4_0, data: data)
    }

    static func q6_K(_ name: String, cols: Int, rows: Int, bytes: inout Bytes) -> Tensor {
        var data: [UInt8] = []
        for _ in 0..<(cols / 256 * rows) {
            for _ in 0..<208 { data.append(bytes.next()) }
            data += bytes.half(0.001)
        }
        return Tensor(name: name, dims: [UInt64(cols), UInt64(rows)], type: .q6_K, data: data)
    }

    /// BF16-exact F32 values, as Google's norms are.
    static func f32(_ name: String, dims: [Int], bytes: inout Bytes, exact: Bool = true) -> Tensor {
        var data: [UInt8] = []
        for _ in 0..<dims.reduce(1, *) {
            var bits = (Float(bytes.next()) / 64 - 2).bitPattern
            if exact { bits &= 0xFFFF_0000 }
            data += bytes.f32(Float(bitPattern: bits))
        }
        return Tensor(name: name, dims: dims.map(UInt64.init), type: .f32, data: data)
    }

    static func tensors(seed: UInt64 = 1) -> [Tensor] {
        var b = Bytes(state: seed)
        var t: [Tensor] = [q6_K("token_embd.weight", cols: hidden, rows: vocab, bytes: &b),
                           f32("output_norm.weight", dims: [hidden], bytes: &b)]
        var rope: [UInt8] = []
        for i in 0..<(fullHeadDim / 2) { rope += b.f32(i < fullHeadDim / 8 ? 1 : 1e30) }
        t.append(Tensor(name: "rope_freqs.weight", dims: [UInt64(fullHeadDim / 2)], type: .f32, data: rope))
        for layer in 0..<layers {
            let p = "blk.\(layer)."
            let full = fullLayers.contains(layer)
            let hd = full ? fullHeadDim : headDim
            let kv = 1
            t.append(q4_0(p + "attn_q.weight", cols: hidden, rows: heads * hd, bytes: &b))
            t.append(q4_0(p + "attn_k.weight", cols: hidden, rows: kv * hd, bytes: &b))
            if !full { t.append(q4_0(p + "attn_v.weight", cols: hidden, rows: kv * hd, bytes: &b)) }
            t.append(q4_0(p + "attn_output.weight", cols: heads * hd, rows: hidden, bytes: &b))
            t.append(f32(p + "attn_q_norm.weight", dims: [hd], bytes: &b))
            t.append(f32(p + "attn_k_norm.weight", dims: [hd], bytes: &b))
            for norm in ["attn_norm", "post_attention_norm", "ffn_norm", "pre_ffw_norm_2",
                         "post_ffw_norm", "post_ffw_norm_1", "post_ffw_norm_2"] {
                t.append(f32(p + norm + ".weight", dims: [hidden], bytes: &b))
            }
            t.append(q4_0(p + "ffn_gate.weight", cols: hidden, rows: shared, bytes: &b))
            t.append(q4_0(p + "ffn_up.weight", cols: hidden, rows: shared, bytes: &b))
            t.append(q4_0(p + "ffn_down.weight", cols: shared, rows: hidden, bytes: &b))
            t.append(f32(p + "ffn_gate_inp.weight", dims: [hidden, experts], bytes: &b, exact: false))
            t.append(f32(p + "ffn_gate_inp.scale", dims: [hidden], bytes: &b))
            t.append(f32(p + "ffn_down_exps.scale", dims: [experts], bytes: &b))
            t.append(f32(p + "layer_output_scale.weight", dims: [1], bytes: &b))
            t.append(q4_0(p + "ffn_gate_up_exps.weight", cols: hidden, rows: 2 * moe,
                          extra: [UInt64(experts)], bytes: &b))
            t.append(q4_0(p + "ffn_down_exps.weight", cols: moe, rows: hidden,
                          extra: [UInt64(experts)], bytes: &b))
        }
        return t
    }

    /// The vision `mmproj`: one layer, hidden 64, patch 4, projecting to
    /// the text fixture's hidden size.
    static let visionHidden = 64, visionFF = 96, patch = 4

    static func visionTensors(seed: UInt64 = 7) -> [Tensor] {
        var b = Bytes(state: seed)
        func bf16(_ name: String, dims: [Int]) -> Tensor {
            var data: [UInt8] = []
            for _ in 0..<dims.reduce(1, *) { data.append(b.next()); data.append(b.next() & 0x3F) }
            return Tensor(name: name, dims: dims.map(UInt64.init), type: .bf16, data: data)
        }
        let h = visionHidden
        var t = [bf16("mm.input_projection.weight", dims: [h, hidden]),
                 f32("v.patch_embd.weight", dims: [patch, patch, 3, h], bytes: &b),
                 f32("v.position_embd.weight", dims: [h, 16, 2], bytes: &b),
                 f32("v.std_bias", dims: [h], bytes: &b),
                 f32("v.std_scale", dims: [h], bytes: &b)]
        let p = "v.blk.0."
        for name in ["ln1", "ln2", "attn_post_norm", "ffn_post_norm"] { t.append(f32(p + name + ".weight", dims: [h], bytes: &b)) }
        for name in ["attn_q_norm", "attn_k_norm"] { t.append(f32(p + name + ".weight", dims: [16], bytes: &b)) }
        for name in ["attn_q", "attn_k", "attn_v", "attn_out"] { t.append(bf16(p + name + ".weight", dims: [h, h])) }
        t.append(bf16(p + "ffn_gate.weight", dims: [h, visionFF]))
        t.append(bf16(p + "ffn_up.weight", dims: [h, visionFF]))
        t.append(bf16(p + "ffn_down.weight", dims: [visionFF, h]))
        return t
    }

    static func mmproj(_ tensors: [Tensor]) -> [UInt8] {
        gguf(tensors, metadata: [
            ("clip.vision.projector_type", { w in w.u32(8); w.string("gemma4v") }),
            ("clip.vision.embedding_length", { w in w.u32(4); w.u32(UInt32(visionHidden)) }),
            ("clip.vision.block_count", { w in w.u32(4); w.u32(1) }),
            ("clip.vision.patch_size", { w in w.u32(4); w.u32(UInt32(patch)) }),
            ("clip.vision.projection_dim", { w in w.u32(4); w.u32(UInt32(hidden)) }),
        ])
    }

    /// Little-endian GGUF writer.
    struct Writer {
        var out: [UInt8] = []
        mutating func u32(_ v: UInt32) { for i in 0..<4 { out.append(UInt8(truncatingIfNeeded: v >> (8 * UInt32(i)))) } }
        mutating func u64(_ v: UInt64) { for i in 0..<8 { out.append(UInt8(truncatingIfNeeded: v >> (8 * UInt64(i)))) } }
        mutating func string(_ s: String) { u64(UInt64(s.utf8.count)); out += Array(s.utf8) }
    }

    static func gguf(_ tensors: [Tensor], metadata: [(String, (inout Writer) -> Void)]) -> [UInt8] {
        var w = Writer()
        w.out = Array("GGUF".utf8)
        w.u32(3)
        w.u64(UInt64(tensors.count))
        w.u64(UInt64(metadata.count))
        for (key, write) in metadata { w.string(key); write(&w) }
        var offset: UInt64 = 0
        var offsets: [UInt64] = []
        for t in tensors {
            offsets.append(offset)
            offset = (offset + UInt64(t.data.count) + 31) / 32 * 32
        }
        for (t, o) in zip(tensors, offsets) {
            w.string(t.name)
            w.u32(UInt32(t.dims.count))
            for d in t.dims { w.u64(d) }
            w.u32(t.type.rawValue)
            w.u64(o)
        }
        while w.out.count % 32 != 0 { w.out.append(0) }
        let dataStart = w.out.count
        for (t, o) in zip(tensors, offsets) {
            while w.out.count < dataStart + Int(o) { w.out.append(0) }
            w.out += t.data
        }
        return w.out
    }

    static func gguf(_ tensors: [Tensor]) -> [UInt8] {
        var out: [UInt8] = Array("GGUF".utf8)
        func u32(_ v: UInt32) { for i in 0..<4 { out.append(UInt8(truncatingIfNeeded: v >> (8 * UInt32(i)))) } }
        func u64(_ v: UInt64) { for i in 0..<8 { out.append(UInt8(truncatingIfNeeded: v >> (8 * UInt64(i)))) } }
        func string(_ s: String) { u64(UInt64(s.utf8.count)); out += Array(s.utf8) }
        func f32(_ v: Float) { u32(v.bitPattern) }

        let full = (0..<layers).map { fullLayers.contains($0) }
        var kvs: [(String, () -> Void)] = [
            ("general.architecture", { u32(8); string("gemma4") }),
            ("general.alignment", { u32(4); u32(32) }),
            ("gemma4.block_count", { u32(4); u32(UInt32(layers)) }),
            ("gemma4.embedding_length", { u32(4); u32(UInt32(hidden)) }),
            ("gemma4.feed_forward_length", { u32(4); u32(UInt32(shared)) }),
            ("gemma4.expert_feed_forward_length", { u32(4); u32(UInt32(moe)) }),
            ("gemma4.attention.head_count", { u32(4); u32(UInt32(heads)) }),
            ("gemma4.attention.head_count_kv", {
                u32(9); u32(4); u64(UInt64(layers)); for _ in 0..<layers { u32(1) }
            }),
            ("gemma4.attention.sliding_window_pattern", {
                u32(9); u32(7); u64(UInt64(layers)); for f in full { out.append(f ? 0 : 1) }
            }),
            ("gemma4.attention.key_length", { u32(4); u32(UInt32(fullHeadDim)) }),
            ("gemma4.attention.value_length", { u32(4); u32(UInt32(fullHeadDim)) }),
            ("gemma4.attention.key_length_swa", { u32(4); u32(UInt32(headDim)) }),
            ("gemma4.attention.value_length_swa", { u32(4); u32(UInt32(headDim)) }),
            ("gemma4.rope.dimension_count", { u32(4); u32(UInt32(fullHeadDim)) }),
            ("gemma4.rope.dimension_count_swa", { u32(4); u32(UInt32(headDim)) }),
            ("gemma4.attention.sliding_window", { u32(4); u32(1024) }),
            ("gemma4.final_logit_softcapping", { u32(6); f32(30) }),
            ("gemma4.rope.freq_base", { u32(6); f32(1_000_000) }),
            ("gemma4.rope.freq_base_swa", { u32(6); f32(10_000) }),
            ("gemma4.expert_count", { u32(4); u32(UInt32(experts)) }),
            ("gemma4.expert_used_count", { u32(4); u32(2) }),
            ("tokenizer.ggml.tokens", {
                u32(9); u32(8); u64(UInt64(vocab)); for i in 0..<vocab { string("t\(i)") }
            }),
        ]
        // A long array is skipped, not kept.
        kvs.append(("tokenizer.ggml.merges", {
            u32(9); u32(8); u64(5000); for _ in 0..<5000 { string("a b") }
        }))

        u32(3)
        u64(UInt64(tensors.count))
        u64(UInt64(kvs.count))
        for (key, write) in kvs { string(key); write() }
        var offset: UInt64 = 0
        var offsets: [UInt64] = []
        for t in tensors {
            offsets.append(offset)
            offset = (offset + UInt64(t.data.count) + 31) / 32 * 32
        }
        for (t, o) in zip(tensors, offsets) {
            string(t.name)
            u32(UInt32(t.dims.count))
            for d in t.dims { u64(d) }
            u32(t.type.rawValue)
            u64(o)
        }
        while out.count % 32 != 0 { out.append(0) }
        let dataStart = out.count
        for (t, o) in zip(tensors, offsets) {
            while out.count < dataStart + Int(o) { out.append(0) }
            out += t.data
        }
        return out
    }
}
