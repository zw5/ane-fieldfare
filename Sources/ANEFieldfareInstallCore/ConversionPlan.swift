package import ANEFieldfareModelFormat

/// Output files of a conversion, by index into `ConversionPlan.files`.
package struct PlannedFile: Sendable, Equatable {
    package let path: String
    package let size: UInt64
}

/// One run of whole source rows that converts on its own: the bytes at
/// `[source, source + rows * rowBytes)` become one contiguous run in each of
/// `outputs` (weights, then scales, then biases, as the operation produces).
package struct Segment: Sendable, Equatable {
    package enum Operation: Sendable, Equatable {
        case repackQ4_0(cols: Int)
        case q6KToQ4_0(cols: Int)
        case f32ToBF16
        case f32ToAffine8(cols: Int)
        /// BF16 copied unchanged.
        case copy
        case bf16ToQ4_0(cols: Int)
        /// A conv kernel row `[channels][size][size]` (F32) as the linear
        /// patch projection's row `[size][size][channels]` (BF16).
        case f32PatchToBF16(channels: Int, size: Int)
        /// Writes nothing: the rows must equal these bytes exactly.
        case expect([UInt8])
    }

    package struct Output: Sendable, Equatable {
        package let file: Int
        package let offset: UInt64
    }

    package let source: UInt64
    package let rows: Int
    package let rowBytes: Int
    package let operation: Operation
    package let outputs: [Output]

    package var sourceEnd: UInt64 { source + UInt64(rows * rowBytes) }
}

/// Everything decided from the GGUF header before any tensor data is read.
package struct ConversionPlan: Sendable {
    package let files: [PlannedFile]
    package let resident: FieldfareTensorFile
    package let experts: FieldfareExperts
    package let arch: FieldfareArch
    /// In source order, non-overlapping.
    package let segments: [Segment]
}

package enum ConversionPlanError: Error, Equatable, CustomStringConvertible, Sendable {
    case unsupported(String)

    package var description: String {
        switch self { case let .unsupported(reason): "unsupported GGUF: \(reason)" }
    }
}

/// Gemma 4 26B-A4B from Google's QAT Q4_0 GGUF.
package enum Gemma4Plan {
    package static let residentFile = "resident.bin"
    static func expertFile(_ layer: Int) -> String {
        "experts/layer_\(layer < 10 ? "0" : "")\(layer).bin"
    }
    /// Source segments stay below this many bytes so conversion memory is bounded.
    static let segmentBytes = 4 * 1024 * 1024

    /// GGUF per-layer name → `.fieldfare` name, for every non-expert tensor.
    static let layerTensors: [(gguf: String, name: String)] = [
        ("attn_q.weight", "self_attn.q_proj.weight"),
        ("attn_k.weight", "self_attn.k_proj.weight"),
        ("attn_v.weight", "self_attn.v_proj.weight"),
        ("attn_output.weight", "self_attn.o_proj.weight"),
        ("attn_q_norm.weight", "self_attn.q_norm.weight"),
        ("attn_k_norm.weight", "self_attn.k_norm.weight"),
        ("attn_norm.weight", "input_layernorm.weight"),
        ("post_attention_norm.weight", "post_attention_layernorm.weight"),
        ("ffn_norm.weight", "pre_feedforward_layernorm.weight"),
        ("pre_ffw_norm_2.weight", "pre_feedforward_layernorm_2.weight"),
        ("post_ffw_norm.weight", "post_feedforward_layernorm.weight"),
        ("post_ffw_norm_1.weight", "post_feedforward_layernorm_1.weight"),
        ("post_ffw_norm_2.weight", "post_feedforward_layernorm_2.weight"),
        ("ffn_gate.weight", "mlp.gate_proj.weight"),
        ("ffn_up.weight", "mlp.up_proj.weight"),
        ("ffn_down.weight", "mlp.down_proj.weight"),
        ("ffn_gate_inp.weight", "router.proj.weight"),
        ("ffn_gate_inp.scale", "router.scale"),
        ("ffn_down_exps.scale", "router.per_expert_scale"),
        ("layer_output_scale.weight", "layer_scalar"),
    ]

    package static func plan(_ header: GGUFHeader) throws -> ConversionPlan {
        let arch = try architecture(header)
        let layers = arch.numLayers
        var builder = Builder()
        let resident = builder.addFile(residentFile)
        var residentTensors: [FieldfareTensor] = []
        var consumed: Set<String> = []

        func source(_ name: String) throws -> GGUFHeader.Tensor {
            guard let tensor = header.tensor(name) else { throw ConversionPlanError.unsupported("missing \(name)") }
            consumed.insert(name)
            return tensor
        }
        func add(_ gguf: GGUFHeader.Tensor, as name: String) throws {
            residentTensors.append(try builder.addResident(gguf, name: name, file: resident))
        }

        try add(try source("token_embd.weight"), as: "embed_tokens.weight")
        for layer in 0..<layers {
            for (ggufName, name) in layerTensors {
                let full = "blk.\(layer).\(ggufName)"
                // Full-attention layers have no v_proj: K doubles as V.
                if ggufName == "attn_v.weight", header.tensor(full) == nil,
                   arch.fullAttentionLayers.contains(layer) { continue }
                try add(try source(full), as: "layers.\(layer).\(name)")
            }
        }
        try add(try source("output_norm.weight"), as: "norm.weight")
        builder.files[resident] = PlannedFile(path: residentFile, size: builder.cursor)

        builder.segments.append(try ropeFrequencies(try source("rope_freqs.weight"), arch: arch))

        // Expert blob: gate, up, down, each weights then scales.
        let d = UInt64(arch.hiddenSize), f = UInt64(arch.moeIntermediateSize)
        var blob: UInt64 = 0
        func region(_ size: UInt64) -> FieldfareRegion {
            let r = FieldfareRegion(offset: blob, size: size)
            blob = Builder.aligned(blob + size)
            return r
        }
        func q4Tensor(_ name: String, rows: UInt64, cols: UInt64) throws -> FieldfareTensor {
            let sizes = try FieldfareTensor.regionSizes(encoding: .q4_0, shape: [rows, cols], field: name)
            return FieldfareTensor(name: name, encoding: .q4_0, shape: [rows, cols],
                                   weights: region(sizes.weights), scales: region(sizes.scales))
        }
        let gate = try q4Tensor("gate", rows: f, cols: d)
        let up = try q4Tensor("up", rows: f, cols: d)
        let down = try q4Tensor("down", rows: d, cols: f)
        let stride = (blob + FieldfareFormat.expertAlignment - 1) / FieldfareFormat.expertAlignment
            * FieldfareFormat.expertAlignment
        let numExperts = arch.numExperts

        var expertFiles: [String] = []
        for layer in 0..<layers {
            let path = expertFile(layer)
            expertFiles.append(path)
            let file = builder.addFile(path, size: UInt64(numExperts) * stride)
            let gateUp = try source("blk.\(layer).ffn_gate_up_exps.weight")
            let downs = try source("blk.\(layer).ffn_down_exps.weight")
            guard gateUp.type == .q4_0, gateUp.dims == [d, 2 * f, UInt64(numExperts)],
                  downs.type == .q4_0, downs.dims == [f, d, UInt64(numExperts)] else {
                throw ConversionPlanError.unsupported("layer \(layer) expert tensors")
            }
            // One segment per expert half: the first f rows of gate_up are the
            // gate, the next f the up projection.
            let halfBytes = UInt64(f) * gateUp.rowBytes
            let downBytes = UInt64(d) * downs.rowBytes
            for e in 0..<UInt64(numExperts) {
                let base = e * stride
                for (half, tensor) in [gate, up].enumerated() {
                    builder.segments.append(Segment(
                        source: gateUp.offset + (2 * e + UInt64(half)) * halfBytes,
                        rows: Int(f), rowBytes: Int(gateUp.rowBytes), operation: .repackQ4_0(cols: Int(d)),
                        outputs: tensor.outputs(file: file, base: base)))
                }
                builder.segments.append(Segment(
                    source: downs.offset + e * downBytes,
                    rows: Int(d), rowBytes: Int(downs.rowBytes), operation: .repackQ4_0(cols: Int(f)),
                    outputs: down.outputs(file: file, base: base)))
            }
        }

        let unknown = header.tensors.map(\.name).filter { !consumed.contains($0) }
        guard unknown.isEmpty else {
            throw ConversionPlanError.unsupported("unexpected tensors \(unknown.prefix(5))")
        }
        builder.segments.sort { $0.source < $1.source }
        for (a, b) in zip(builder.segments, builder.segments.dropFirst()) where a.sourceEnd > b.source {
            throw ConversionPlanError.unsupported("overlapping source rows at \(b.source)")
        }
        return ConversionPlan(
            files: builder.files,
            resident: FieldfareTensorFile(file: residentFile, tensors: residentTensors),
            experts: FieldfareExperts(stride: stride, perLayer: numExperts, files: expertFiles,
                                      tensors: [gate, up, down]),
            arch: arch,
            segments: builder.segments)
    }

    /// Layout state: files, the resident write cursor and the segments.
    struct Builder {
        var files: [PlannedFile] = []
        var cursor: UInt64 = 0
        var segments: [Segment] = []

        static func aligned(_ x: UInt64) -> UInt64 {
            (x + FieldfareFormat.regionAlignment - 1) / FieldfareFormat.regionAlignment
                * FieldfareFormat.regionAlignment
        }

        mutating func addFile(_ path: String, size: UInt64 = 0) -> Int {
            files.append(PlannedFile(path: path, size: size))
            return files.count - 1
        }

        mutating func reserve(_ size: UInt64) -> FieldfareRegion {
            let region = FieldfareRegion(offset: cursor, size: size)
            cursor = Self.aligned(cursor + size)
            return region
        }

        /// Lays out one resident tensor and adds the segments that fill it.
        mutating func addResident(_ gguf: GGUFHeader.Tensor, name: String, file: Int) throws -> FieldfareTensor {
            let encoding: FieldfareEncoding
            let operation: Segment.Operation
            let shape: [UInt64] = gguf.dims.count == 1 ? [gguf.cols] : [gguf.rows, gguf.cols]
            switch (gguf.type, gguf.dims.count) {
            case (.q4_0, 2):
                encoding = .q4_0
                operation = .repackQ4_0(cols: Int(gguf.cols))
            case (.q6_K, 2):
                encoding = .q4_0
                operation = .q6KToQ4_0(cols: Int(gguf.cols))
            case (.f32, 1):
                encoding = .bf16
                operation = .f32ToBF16
            case (.f32, 2) where name.hasSuffix("router.proj.weight"):
                encoding = .affine8
                operation = .f32ToAffine8(cols: Int(gguf.cols))
            default:
                throw ConversionPlanError.unsupported("\(gguf.name): \(gguf.type) rank \(gguf.dims.count)")
            }
            let sizes = try FieldfareTensor.regionSizes(encoding: encoding, shape: shape, field: name)
            let weights = reserve(sizes.weights)
            let scales = sizes.scales > 0 ? reserve(sizes.scales) : nil
            let biases = sizes.biases > 0 ? reserve(sizes.biases) : nil
            let tensor = FieldfareTensor(name: name, encoding: encoding, shape: shape,
                                         weights: weights, scales: scales, biases: biases)

            // Split into segments of whole rows; each output advances by the
            // per-row size of its region.
            let rows = Int(gguf.rows)
            let rowBytes = Int(gguf.rowBytes)
            let perSegment = Swift.max(1, Gemma4Plan.segmentBytes / rowBytes)
            let regions = [weights, scales, biases].compactMap { $0 }
            var row = 0
            while row < rows {
                let n = Swift.min(perSegment, rows - row)
                segments.append(Segment(
                    source: gguf.offset + UInt64(row * rowBytes), rows: n, rowBytes: rowBytes,
                    operation: operation,
                    outputs: regions.map { Segment.Output(file: file, offset: $0.offset + $0.size / UInt64(rows) * UInt64(row)) }))
                row += n
            }
            return tensor
        }
    }

    /// Validates the GGUF metadata against the one architecture this runtime
    /// runs, and returns it.
    static func architecture(_ header: GGUFHeader) throws -> FieldfareArch {
        let m = header.metadata
        func int(_ key: String) throws -> Int {
            guard let v = m[key]?.intValue else { throw ConversionPlanError.unsupported("missing \(key)") }
            return v
        }
        func double(_ key: String) throws -> Double {
            guard let v = m[key]?.doubleValue else { throw ConversionPlanError.unsupported("missing \(key)") }
            return v
        }
        guard case .string("gemma4")? = m["general.architecture"] else {
            throw ConversionPlanError.unsupported("architecture is not gemma4")
        }
        let layers = try int("gemma4.block_count")
        guard case let .array(kvHeads)? = m["gemma4.attention.head_count_kv"],
              case let .array(pattern)? = m["gemma4.attention.sliding_window_pattern"],
              kvHeads.count == layers, pattern.count == layers else {
            throw ConversionPlanError.unsupported("per-layer attention metadata")
        }
        var full: [Int] = []
        var slidingKV: Set<Int> = [], fullKV: Set<Int> = []
        for layer in 0..<layers {
            guard case let .bool(sliding) = pattern[layer], let heads = kvHeads[layer].intValue else {
                throw ConversionPlanError.unsupported("attention metadata of layer \(layer)")
            }
            if sliding { slidingKV.insert(heads) } else { full.append(layer); fullKV.insert(heads) }
        }
        guard slidingKV.count == 1, fullKV.count == 1, let numKV = slidingKV.first,
              let numFullKV = fullKV.first else {
            throw ConversionPlanError.unsupported("mixed KV head counts")
        }
        let vocab: UInt64
        switch m["tokenizer.ggml.tokens"] {
        case let .longArray(count)?: vocab = count
        case let .array(tokens)?: vocab = UInt64(tokens.count)
        default: throw ConversionPlanError.unsupported("missing vocabulary")
        }
        let arch = FieldfareArch(
            hiddenSize: try int("gemma4.embedding_length"),
            intermediateSize: try int("gemma4.feed_forward_length"),
            moeIntermediateSize: try int("gemma4.expert_feed_forward_length"),
            numHeads: try int("gemma4.attention.head_count"),
            numKVHeads: numKV, numFullKVHeads: numFullKV,
            headDim: try int("gemma4.attention.key_length_swa"),
            fullHeadDim: try int("gemma4.attention.key_length"),
            vocabSize: Int(vocab),
            slidingWindow: try int("gemma4.attention.sliding_window"),
            finalLogitSoftcap: try double("gemma4.final_logit_softcapping"),
            ropeTheta: try double("gemma4.rope.freq_base_swa"),
            fullRopeTheta: try double("gemma4.rope.freq_base"),
            partialRotaryFactor: 0.25,
            numLayers: layers,
            numExperts: try int("gemma4.expert_count"),
            topKExperts: try int("gemma4.expert_used_count"),
            tieWordEmbeddings: header.tensor("output.weight") == nil,
            attentionKEqV: true,
            hiddenActivation: "gelu_pytorch_tanh",
            fullAttentionLayers: full)
        guard try int("gemma4.attention.value_length") == arch.fullHeadDim,
              try int("gemma4.attention.value_length_swa") == arch.headDim,
              try int("gemma4.rope.dimension_count") == arch.fullHeadDim,
              try int("gemma4.rope.dimension_count_swa") == arch.headDim else {
            throw ConversionPlanError.unsupported("head dimensions")
        }
        return arch
    }

    /// Proportional RoPE: the full-attention layers rotate the first quarter
    /// of their frequency pairs (factor 1) and leave the rest unrotated
    /// (factor 1e30). The runtime implements exactly this, so the converter
    /// only checks the table.
    static func ropeFrequencies(_ tensor: GGUFHeader.Tensor, arch: FieldfareArch) throws -> Segment {
        let pairs = arch.fullHeadDim / 2
        guard tensor.type == .f32, tensor.dims == [UInt64(pairs)] else {
            throw ConversionPlanError.unsupported("rope_freqs shape")
        }
        let rotated = Int(Double(pairs) * arch.partialRotaryFactor)
        var expected: [UInt8] = []
        for i in 0..<pairs {
            let bits = Float(i < rotated ? 1 : 1e30).bitPattern
            for b in 0..<4 { expected.append(UInt8(truncatingIfNeeded: bits >> (8 * b))) }
        }
        return Segment(source: tensor.offset, rows: 1, rowBytes: Int(tensor.size),
                       operation: .expect(expected), outputs: [])
    }
}

extension FieldfareTensor {
    /// The segment outputs of an expert tensor in the blob at `base`.
    func outputs(file: Int, base: UInt64) -> [Segment.Output] {
        [weights, scales, biases].compactMap { $0 }.map { Segment.Output(file: file, offset: base + $0.offset) }
    }
}


/// The vision tower and projector of Gemma 4 26B-A4B from Google's QAT
/// release's `mmproj` GGUF. The tower stays BF16 (the GGUF's F32 norms and
/// tables are BF16 values widened) and the projector is quantized to Q4_0.
package enum Gemma4VisionPlan {
    package static let file = "vision.bin"

    package struct Plan: Sendable {
        package let files: [PlannedFile]
        package let vision: FieldfareTensorFile
        package let segments: [Segment]
    }

    static let layerTensors: [(gguf: String, name: String)] = [
        ("ln1.weight", "input_layernorm.weight"),
        ("attn_q.weight", "self_attn.q_proj.linear.weight"),
        ("attn_k.weight", "self_attn.k_proj.linear.weight"),
        ("attn_v.weight", "self_attn.v_proj.linear.weight"),
        ("attn_out.weight", "self_attn.o_proj.linear.weight"),
        ("attn_q_norm.weight", "self_attn.q_norm.weight"),
        ("attn_k_norm.weight", "self_attn.k_norm.weight"),
        ("attn_post_norm.weight", "post_attention_layernorm.weight"),
        ("ln2.weight", "pre_feedforward_layernorm.weight"),
        ("ffn_gate.weight", "mlp.gate_proj.linear.weight"),
        ("ffn_up.weight", "mlp.up_proj.linear.weight"),
        ("ffn_down.weight", "mlp.down_proj.linear.weight"),
        ("ffn_post_norm.weight", "post_feedforward_layernorm.weight"),
    ]

    package static func plan(_ header: GGUFHeader, textHiddenSize: Int) throws -> Plan {
        let m = header.metadata
        func int(_ key: String) throws -> Int {
            guard let v = m[key]?.intValue else { throw ConversionPlanError.unsupported("missing \(key)") }
            return v
        }
        guard case .string("gemma4v")? = m["clip.vision.projector_type"] else {
            throw ConversionPlanError.unsupported("projector is not gemma4v")
        }
        let hidden = try int("clip.vision.embedding_length"), layers = try int("clip.vision.block_count")
        let patch = try int("clip.vision.patch_size")
        guard try int("clip.vision.projection_dim") == textHiddenSize else {
            throw ConversionPlanError.unsupported("projector width does not match the text model")
        }

        var builder = Gemma4Plan.Builder()
        let file = builder.addFile(Self.file)
        var tensors: [FieldfareTensor] = []
        var consumed: Set<String> = []
        func source(_ name: String) throws -> GGUFHeader.Tensor {
            guard let tensor = header.tensor(name) else { throw ConversionPlanError.unsupported("missing \(name)") }
            consumed.insert(name)
            return tensor
        }
        func add(_ gguf: GGUFHeader.Tensor, as name: String, operation: Segment.Operation? = nil) throws {
            tensors.append(try builder.addVision(gguf, name: name, file: file, operation: operation))
        }

        let patchKernel = try source("v.patch_embd.weight")
        guard patchKernel.type == .f32, patchKernel.dims == [UInt64(patch), UInt64(patch), 3, UInt64(hidden)] else {
            throw ConversionPlanError.unsupported("patch embedding shape \(patchKernel.dims)")
        }
        try add(patchKernel, as: "vision_tower.patch_embedder.input_proj.weight",
                operation: .f32PatchToBF16(channels: 3, size: patch))
        try add(try source("v.position_embd.weight"), as: "vision_tower.patch_embedder.position_embedding_table")
        for layer in 0..<layers {
            for (ggufName, name) in layerTensors {
                try add(try source("v.blk.\(layer).\(ggufName)"), as: "vision_tower.encoder.layers.\(layer).\(name)")
            }
        }
        try add(try source("v.std_bias"), as: "vision_tower.std_bias")
        try add(try source("v.std_scale"), as: "vision_tower.std_scale")
        let projector = try source("mm.input_projection.weight")
        guard projector.type == .bf16, projector.dims == [UInt64(hidden), UInt64(textHiddenSize)] else {
            throw ConversionPlanError.unsupported("projector shape \(projector.dims)")
        }
        try add(projector, as: "embed_vision.embedding_projection.weight",
                operation: .bf16ToQ4_0(cols: hidden))
        builder.files[file] = PlannedFile(path: Self.file, size: builder.cursor)

        let unknown = header.tensors.map(\.name).filter { !consumed.contains($0) }
        guard unknown.isEmpty else {
            throw ConversionPlanError.unsupported("unexpected vision tensors \(unknown.prefix(5))")
        }
        builder.segments.sort { $0.source < $1.source }
        return Plan(files: builder.files, vision: FieldfareTensorFile(file: Self.file, tensors: tensors),
                    segments: builder.segments)
    }
}

extension Gemma4Plan.Builder {
    /// Lays out one vision tensor: BF16 unless `operation` quantizes it.
    mutating func addVision(_ gguf: GGUFHeader.Tensor, name: String, file: Int,
                            operation explicit: Segment.Operation?) throws -> FieldfareTensor {
        let operation: Segment.Operation
        switch (explicit, gguf.type) {
        case let (op?, _): operation = op
        case (nil, .f32): operation = .f32ToBF16
        case (nil, .bf16): operation = .copy
        default: throw ConversionPlanError.unsupported("\(gguf.name): \(gguf.type)")
        }
        let encoding: FieldfareEncoding
        let shape: [UInt64]
        switch operation {
        case .bf16ToQ4_0:
            encoding = .q4_0
            shape = [gguf.rows, gguf.cols]
        case let .f32PatchToBF16(channels, size):
            encoding = .bf16
            shape = [gguf.dims[3], UInt64(channels * size * size)]
        default:
            encoding = .bf16
            shape = Array(gguf.dims.reversed())
        }
        let sizes = try FieldfareTensor.regionSizes(encoding: encoding, shape: shape, field: name)
        let weights = reserve(sizes.weights)
        let scales = sizes.scales > 0 ? reserve(sizes.scales) : nil
        let regions = [weights, scales].compactMap { $0 }
        // One kernel row of the patch embedding is all its input channels.
        let rows = Int(gguf.rows)
        let rowBytes = Int(gguf.rowBytes)
        let (unitRows, unitBytes): (Int, Int) = if case .f32PatchToBF16 = operation {
            (Int(gguf.dims[3]), Int(gguf.size) / Int(gguf.dims[3]))
        } else {
            (rows, rowBytes)
        }
        let perSegment = Swift.max(1, Gemma4Plan.segmentBytes / unitBytes)
        var row = 0
        while row < unitRows {
            let n = Swift.min(perSegment, unitRows - row)
            segments.append(Segment(
                source: gguf.offset + UInt64(row * unitBytes), rows: n, rowBytes: unitBytes,
                operation: operation,
                outputs: regions.map { Segment.Output(file: file, offset: $0.offset + $0.size / UInt64(unitRows) * UInt64(row)) }))
            row += n
        }
        return FieldfareTensor(name: name, encoding: encoding, shape: shape, weights: weights, scales: scales)
    }
}
