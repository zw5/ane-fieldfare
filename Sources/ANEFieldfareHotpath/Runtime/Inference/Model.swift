public import Foundation
public import Metal
import Darwin
import ANEFieldfareModelFormat

/// Loaded `.fieldfare` model. Resident weights live behind one mmap'd
/// `MTLBuffer`; routed expert weights are per-layer mappings, opened lazily
/// on first touch and served from the OS page cache (`MappedExpertLayer`).
public struct Model {
    public let device: any MTLDevice
    public let config: ArchConfig
    public var modelID: String { manifest.modelID }
    /// SHA-256 of the weights file the model was converted from.
    public var sourceDigest: String? { manifest.source.files.first?.sha256 }

    let residentBuffer: ResidentBuffer
    let residentTensors: [String: FieldfareTensor]
    let manifest: FieldfareManifest
    let directoryURL: URL
    let modelDirectory: ModelDirectory

    /// Routed-expert layer mappings, shared by every copy of this value.
    let routedLayers: RoutedExpertLayers

    init(device: any MTLDevice,
         config: ArchConfig,
         residentBuffer: ResidentBuffer,
         manifest: FieldfareManifest,
         directoryURL: URL,
         modelDirectory: ModelDirectory) throws {
        let residentTensors = Dictionary(uniqueKeysWithValues: manifest.resident.tensors.map { ($0.name, $0) })
        self.embedding = try Self.resident(name: "embed_tokens.weight",
                                           tensors: residentTensors, buffer: residentBuffer)
        self.finalNorm = try Self.resident(name: "norm.weight",
                                           tensors: residentTensors, buffer: residentBuffer)
        self.device = device
        self.config = config
        self.residentBuffer = residentBuffer
        self.residentTensors = residentTensors
        self.manifest = manifest
        self.directoryURL = directoryURL
        self.modelDirectory = modelDirectory
        self.routedLayers = RoutedExpertLayers(
            device: device,
            experts: manifest.experts,
            files: manifest.files,
            modelDirectory: modelDirectory)
    }

    // MARK: - Resident accessors

    /// Resolved at load, so a loaded model always has one.
    public let embedding: TensorView

    /// Gemma 4 ties lm_head to the embedding. The transpose for the lm_head
    /// GEMV path is the kernel's job, not the loader's.
    public var lmHead: TensorView { embedding }

    public func qProj(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).self_attn.q_proj.weight")
    }
    public func kProj(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).self_attn.k_proj.weight")
    }
    public func vProj(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).self_attn.v_proj.weight")
    }
    public func oProj(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).self_attn.o_proj.weight")
    }
    /// Writer emits `.router.proj.weight` (no `.mlp.` segment).
    public func router(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).router.proj.weight")
    }
    /// Writer emits the shared-expert FFN as `.mlp.{gate,up,down}_proj.weight`
    /// without a `.shared_expert.` segment.
    public func sharedExpertGate(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).mlp.gate_proj.weight")
    }
    public func sharedExpertUp(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).mlp.up_proj.weight")
    }
    public func sharedExpertDown(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).mlp.down_proj.weight")
    }
    public func inputNorm(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).input_layernorm.weight")
    }
    public func postAttnNorm(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).post_attention_layernorm.weight")
    }
    public let finalNorm: TensorView

    // MARK: - Per-head attention norms (Q/K only)
    //
    // `q_norm` and `k_norm` are RMSNorm with learnable scale, applied per head
    // before RoPE. `v_norm` has **no learnable weight** (no-scale RMSNorm) and
    // is therefore not stored as a tensor — the runtime uses an
    // explicit no-scale variant rather than consuming a unit-weight buffer.

    public func qNorm(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).self_attn.q_norm.weight")
    }
    public func kNorm(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).self_attn.k_norm.weight")
    }

    // MARK: - Feed-forward norms
    //
    // The Gemma 4 sandwich wraps two parallel FFN branches:
    //   pre_feedforward_layernorm        -> dense MLP input
    //   pre_feedforward_layernorm_2      -> routed expert input
    //   post_feedforward_layernorm_1     -> dense MLP output
    //   post_feedforward_layernorm_2     -> routed expert output
    //   post_feedforward_layernorm       -> combined (h1+h2) output

    public func preFFN(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).pre_feedforward_layernorm.weight")
    }
    public func preFFN2(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).pre_feedforward_layernorm_2.weight")
    }
    public func postFFN1(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).post_feedforward_layernorm_1.weight")
    }
    public func postFFN2(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).post_feedforward_layernorm_2.weight")
    }
    public func postFFN(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).post_feedforward_layernorm.weight")
    }

    // MARK: - Router auxiliaries
    //
    // `router.scale` is a per-feature multiplier on the router's input
    // (post-RMSNorm), fused with 1/sqrt(hidden_size). `per_expert_scale` is
    // applied to the top-k routing weights after softmax over top-k.

    public func routerScale(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).router.scale")
    }
    public func routerPerExpertScale(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).router.per_expert_scale")
    }

    /// Per-layer scalar gain applied to the entire residual stream at the end
    /// of the layer; shape `[1]`, BF16.
    public func layerScalar(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).layer_scalar")
    }

    /// Resolve a tensor name to a `TensorView` against the resident buffer.
    /// The manifest codec already checked every region against the file.
    func resident(name: String) throws -> TensorView {
        try Self.resident(name: name, tensors: residentTensors, buffer: residentBuffer)
    }

    private static func resident(name: String,
                                 tensors: [String: FieldfareTensor],
                                 buffer residentBuffer: ResidentBuffer) throws -> TensorView {
        guard let tensor = tensors[name] else {
            throw ModelError.tensorNotFound(name: name)
        }
        let shape = tensor.shape.map { UInt32(truncatingIfNeeded: $0) } + [0, 0, 0]
        return TensorView(
            buffer: residentBuffer.buffer,
            offset: tensor.weights.offset,
            length: tensor.weights.size,
            scaleOffset: tensor.scales?.offset ?? 0, scaleLength: tensor.scales?.size ?? 0,
            biasOffset: tensor.biases?.offset ?? 0, biasLength: tensor.biases?.size ?? 0,
            shape: (shape[0], shape[1], shape[2], shape[3]))
    }

    // MARK: - Routed expert (lazy)

    /// Layer L's mapping, opening its file on first touch.
    func expertLayer(_ L: Int) throws -> MappedExpertLayer {
        try routedLayers.layer(L)
    }

    /// Best-effort overlap hook for prefill: starts the same lazy layer open in
    /// the background without waiting for the first expert fetch. A failure is
    /// dropped here and resurfaces from the first real fetch of the layer.
    public func beginOpeningRoutedExpertLayer(layer L: Int) {
        BlockingIO.detach { [routedLayers] in
            _ = try? routedLayers.layer(L)
        }
    }

}

extension Model {

    /// Open a `.fieldfare/` directory and return a typed handle. The manifest
    /// is checked against the runtime's architecture and tensor schema and
    /// every file size against the manifest; layer files are opened lazily
    /// on first touch. File contents are not hashed here: the installer
    /// verified them before committing the directory, and
    /// `ANEFieldfareInstall --verify` re-checks them on demand.
    public static func load(directoryURL: URL,
                            device: any MTLDevice,
                            expecting: ArchConfig = .gemma4_26B_A4B) throws -> Model {
        let modelDirectory = try ModelDirectory(rootURL: directoryURL)
        let manifest = try ModelManifest.load(modelDirectory, path: directoryURL.path, expecting: expecting)
        try validateRuntimeSchema(manifest: manifest, config: expecting)

        let residentPath = manifest.resident.file
        let residentFD = try modelDirectory.openFile(residentPath)
        defer { close(residentFD) }
        let residentSize = try modelDirectory.fileSize(fileDescriptor: residentFD, relativePath: residentPath)
        let expectedSize = manifest.files[residentPath]?.size ?? 0
        guard residentSize == expectedSize else {
            throw ModelError.tensorSizeMismatch(name: residentPath, expected: expectedSize, actual: residentSize)
        }
        let residentBuffer = try ResidentBuffer(
            fileURL: directoryURL.appendingPathComponent(residentPath),
            fileOffset: 0,
            residentSize: residentSize,
            device: device,
            fileDescriptor: residentFD)

        return try Model(
            device: device,
            config: expecting,
            residentBuffer: residentBuffer,
            manifest: manifest,
            directoryURL: directoryURL,
            modelDirectory: modelDirectory)
    }

    /// Every tensor the forward pass binds, with the encoding and shape its
    /// kernel requires.
    static func validateRuntimeSchema(manifest: FieldfareManifest, config: ArchConfig) throws {
        let tensors = Dictionary(uniqueKeysWithValues: manifest.resident.tensors.map { ($0.name, $0) })
        func require(_ name: String, _ encoding: FieldfareEncoding, _ shape: [Int],
                     in table: [String: FieldfareTensor] = tensors, field: String = "resident") throws {
            guard let tensor = table[name] else {
                throw ModelError.invalidManifest(detail: "\(field) is missing \(name)")
            }
            guard tensor.encoding == encoding, tensor.shape == shape.map(UInt64.init) else {
                throw ModelError.invalidManifest(detail: """
                    \(field) \(name) is \(tensor.encoding) \(tensor.shape); the runtime needs \(encoding) \(shape)
                    """)
            }
        }

        let d = config.hiddenSize
        try require("embed_tokens.weight", .q4_0, [config.vocabSize, d])
        try require("norm.weight", .bf16, [d])
        for layer in 0..<config.numLayers {
            let prefix = "layers.\(layer)"
            let isFull = config.fullAttentionLayerMask[layer] != 0
            let headDimension = isFull ? config.fullHeadDim : config.headDim
            let kvHeads = isFull ? config.numFullKVHeads : config.numKVHeads
            let q = config.numHeads * headDimension, kv = kvHeads * headDimension
            for name in ["input_layernorm.weight", "post_attention_layernorm.weight",
                         "pre_feedforward_layernorm.weight", "pre_feedforward_layernorm_2.weight",
                         "post_feedforward_layernorm_1.weight", "post_feedforward_layernorm_2.weight",
                         "post_feedforward_layernorm.weight", "router.scale"] {
                try require("\(prefix).\(name)", .bf16, [d])
            }
            try require("\(prefix).self_attn.q_norm.weight", .bf16, [headDimension])
            try require("\(prefix).self_attn.k_norm.weight", .bf16, [headDimension])
            try require("\(prefix).router.per_expert_scale", .bf16, [config.numExperts])
            try require("\(prefix).layer_scalar", .bf16, [1])
            try require("\(prefix).self_attn.q_proj.weight", .q4_0, [q, d])
            try require("\(prefix).self_attn.k_proj.weight", .q4_0, [kv, d])
            // Full-attention layers reuse K as V.
            if !isFull { try require("\(prefix).self_attn.v_proj.weight", .q4_0, [kv, d]) }
            try require("\(prefix).self_attn.o_proj.weight", .q4_0, [d, q])
            try require("\(prefix).mlp.gate_proj.weight", .q4_0, [config.intermediateSize, d])
            try require("\(prefix).mlp.up_proj.weight", .q4_0, [config.intermediateSize, d])
            try require("\(prefix).mlp.down_proj.weight", .q4_0, [d, config.intermediateSize])
            try require("\(prefix).router.proj.weight", .affine8, [config.numExperts, d])
        }

        // Routed experts: zero-copy page buffers, with region offsets the
        // kernels take as UInt32 and load 4 bytes at a time.
        let experts = manifest.experts
        guard experts.stride % UInt64(getpagesize()) == 0 else {
            throw ModelError.invalidManifest(detail: "expert stride \(experts.stride) is not page aligned")
        }
        let expertTensors = Dictionary(uniqueKeysWithValues: experts.tensors.map { ($0.name, $0) })
        let f = config.moeIntermediateSize
        try require("gate", .q4_0, [f, d], in: expertTensors, field: "experts")
        try require("up", .q4_0, [f, d], in: expertTensors, field: "experts")
        try require("down", .q4_0, [d, f], in: expertTensors, field: "experts")
        guard experts.stride <= UInt64(UInt32.max) + 1 else {
            throw ModelError.invalidManifest(detail: "expert regions do not fit UInt32 offsets")
        }
    }

    /// Unmaps routed-expert layers before vision encoding under `.onDemand`.
    /// Expert pages live in the page cache, which the OS reclaims under
    /// pressure either way, so this only drops the mappings; layers reopen on
    /// next touch.
    ///
    /// The caller must serialize this transition against forward execution.
    public func prepareExpertResidencyForVision(
        _ policy: VisionResidencyPolicy,
        gpuDrainNanoseconds: UInt64 = 0
    ) -> VisionExpertResidencyTransition {
        let start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let released = policy == .onDemand ? routedLayers.release() : 0
        return VisionExpertResidencyTransition(
            policy: policy,
            gpuDrainNanoseconds: gpuDrainNanoseconds,
            releasedLayerCount: released,
            remainingOpenLayerCount: routedLayers.openLayerCount,
            wallNanoseconds: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start)
    }

}
