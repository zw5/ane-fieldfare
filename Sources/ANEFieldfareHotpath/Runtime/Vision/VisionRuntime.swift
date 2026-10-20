public import Foundation
public import Metal
import ANEFieldfareImportedFormat

public enum VisionRuntimeError: Error, CustomStringConvertible {
    case invalidInput(String)
    case commandFailed(String)
    case unsupportedKernel(String)

    public var description: String {
        switch self {
        case .invalidInput(let detail): "invalid vision input: \(detail)"
        case .commandFailed(let detail): "vision Metal command failed: \(detail)"
        case .unsupportedKernel(let detail): "vision kernel unavailable: \(detail)"
        }
    }
}

public struct VisionFeatures {
    public let buffer: any MTLBuffer
    public let tokenCount: Int
    public let hiddenSize: Int
}

public final class VisionRuntime {

    private let context: MetalContext
    private let store: VisionWeightStore
    private let useLease: VisionPackUseLease
    private let linear: VisionLinearBF16
    private let primitives: VisionPrimitives
    private let attention: VisionAttention
    private let usesBF16Projector: Bool
    private let projectorKernel: MPPPrefillInt4QMM
    private let imageResize: VisionResize?
    /// Mapped weight regions retained across images under `.keepReady`, so
    /// mapping and first-touching the tower's weights is paid once. Pages are
    /// clean, file-backed and evictable, like the routed expert pool.
    private var cachedRegions: [String: VisionMappedWeightRegion] = [:]

    public static func open(
        textModelURL: URL,
        context: MetalContext,
        visionPackURL: URL?
    ) throws -> VisionRuntime {
        try requireSupportedDevice(context.device)
        let manifest = try ManifestReader.load(
            directoryURL: textModelURL,
            expecting: .gemma4_26B_A4B)
        guard let source = manifest.sourceSnapshotHash else {
            throw VisionRuntimeError.invalidInput("text model has no source identity")
        }
        let companion = try visionPackURL
            ?? VisionPackLocation.companionURL(forTextModel: textModelURL)
        // The lease creates its lock file beside the pack (O_CREAT), so the
        // pack and its manifest must be confirmed first: a missing pack or a
        // stray directory must not get a lock file, and a missing pack is
        // `packNotFound`, not invalid metadata.
        guard FileManager.default.fileExists(atPath: companion.path) else {
            throw VisionPackError.packNotFound(companion.path)
        }
        let manifestPath = companion
            .appendingPathComponent(GTurboVisionFormatV1.manifestFile).path
        guard FileManager.default.fileExists(atPath: manifestPath) else {
            throw VisionPackError.invalidMetadata(
                "no \(GTurboVisionFormatV1.manifestFile) in \(companion.path)")
        }
        let useLease = try VisionPackUseLease.acquireShared(companionURL: companion)
        let store = try VisionWeightStore.open(
            directoryURL: companion,
            compatibleTextSourceSnapshotHash: source)
        return try VisionRuntime(context: context, store: store, useLease: useLease)
    }

    public static func requireSupportedDevice(_ device: any MTLDevice) throws {
        guard device.supportsFamily(.apple8) else {
            throw VisionRuntimeError.unsupportedKernel(
                "the image tower requires an M2 or newer Apple Silicon Mac")
        }
    }

    private init(context: MetalContext, store: VisionWeightStore,
                 useLease: consuming VisionPackUseLease) throws {
        self.context = context
        self.store = store
        self.useLease = useLease
        self.linear = try VisionLinearBF16(context: context)
        self.primitives = try VisionPrimitives(context: context)
        self.attention = try VisionAttention(context: context)
        // Pre-Apple10 GPUs convert the pooled features to FP16 for the projector.
        self.usesBF16Projector = context.device.supportsFamily(.apple10)
        self.imageResize = try VisionResize(context: context)
        self.projectorKernel = try MPPPrefillInt4QMM(
            context: context, variant: usesBF16Projector ? .apple10BF16 : .control)
    }

    public func encodeImage(
        at fileURL: URL,
        languageModel: Model,
        residencyPolicy: VisionResidencyPolicy,
        checkCancellation: () throws -> Void
    ) throws -> VisionFeatures {
        try encodeImage(
            plan: try makePreprocessor().plan(fileURL: fileURL),
            languageModel: languageModel,
            residencyPolicy: residencyPolicy,
            checkCancellation: checkCancellation)
    }

    /// Encodes an image the caller has already planned, without reopening the
    /// file, so the encode matches the metadata the plan was validated against.
    public func encodeImage(
        plan: VisionImagePlan,
        languageModel: Model,
        residencyPolicy: VisionResidencyPolicy,
        checkCancellation: () throws -> Void
    ) throws -> VisionFeatures {
        try checkCancellation()
        if residencyPolicy == .onDemand { try drainGPU() }
        try checkCancellation()
        languageModel.prepareExpertResidencyForVision(residencyPolicy)
        try checkCancellation()
        return try encodePatches(
            try makePreprocessor().preprocess(plan),
            retainsWeightRegions: residencyPolicy == .keepReady,
            checkCancellation: checkCancellation)
    }

    private func makePreprocessor() -> Gemma4ImagePreprocessor {
        Gemma4ImagePreprocessor(device: context.device, gpuResize: imageResize)
    }

    private func encodePatches(
        _ input: VisionPixelBuffer,
        retainsWeightRegions: Bool,
        checkCancellation: () throws -> Void
    ) throws -> VisionFeatures {
        if !retainsWeightRegions { cachedRegions.removeAll(keepingCapacity: false) }
        try checkCancellation()
        let patchGridWidth = input.geometry.patchGridWidth
        let patchGridHeight = input.geometry.patchGridHeight
        let rows = patchGridWidth * patchGridHeight

        // The QKV projections write attention's padded head-major layout
        // directly, and the up projection folds GeGLU into its epilogue, so
        // there is no pad/unpad dispatch, no K/V scratch and no `up` scratch.
        let scratch = try VisionScratch(device: context.device, rows: rows)
        let paddedHeadStore = VisionLinearBF16.PaddedHeadStore(
            rowStride: scratch.paddedRows,
            headDim: VisionAttention.headDimension,
            paddedHeadDim: VisionAttention.paddedHeadDimension)
        func mapRegion(_ tensorNames: [String]) throws -> VisionMappedWeightRegion {
            let key = tensorNames.first ?? ""
            if retainsWeightRegions, let cached = cachedRegions[key] {
                return cached
            }
            let region = try store.mapRegion(
                tensorNames: tensorNames, device: context.device)
            if retainsWeightRegions { cachedRegions[key] = region }
            return region
        }
        // Command buffers are pipelined: the CPU encodes the next layer while
        // the GPU executes the previous one. An in-flight entry keeps its
        // mapped weight region alive until its buffer completes, and the
        // window bounds `.onDemand` residency to `pipelineDepth + 1` regions
        // instead of the whole tower.
        var inFlight: [(buffer: any MTLCommandBuffer, region: VisionMappedWeightRegion?)] = []
        let pipelineDepth = 2
        func drainOldest() throws {
            let entry = inFlight.removeFirst()
            entry.buffer.waitUntilCompleted()
            if let detail = metalCommandBufferFailureDetail(label: entry.buffer.label,
                                                            status: entry.buffer.status,
                                                            error: entry.buffer.error) {
                throw VisionRuntimeError.commandFailed(detail)
            }
        }
        func commitPipelined(_ buffer: any MTLCommandBuffer,
                             retaining region: VisionMappedWeightRegion?) throws {
            buffer.commit()
            inFlight.append((buffer, region))
            while inFlight.count > pipelineDepth { try drainOldest() }
        }
        // A throw mid-encode must not release mapped regions while the GPU is
        // still reading them; defers run before locals, so drain first.
        defer {
            for entry in inFlight { entry.buffer.waitUntilCompleted() }
        }

        let patchWeight = "vision_tower.patch_embedder.input_proj.weight"
        let positionTable = "vision_tower.patch_embedder.position_embedding_table"
        do {
            let fixed = try mapRegion([patchWeight, positionTable])
            let commandBuffer = try makeCommandBuffer()
            try scratch.encodeClear(commandBuffer: commandBuffer)
            try attention.encodeClearPadding(commandBuffer: commandBuffer)
            try primitives.encodeNormalize(commandBuffer: commandBuffer,
                                           input: input.patchesBF16,
                                           output: scratch.normalizedPatches,
                                           rows: rows, paddedRows: scratch.paddedRows)
            try linear.encode(
                commandBuffer: commandBuffer,
                input: scratch.normalizedPatches,
                weights: fixed.buffer, weightsOffset: try fixed.offset(of: patchWeight),
                output: scratch.hiddenA,
                m: scratch.paddedRows, n: VisionConfig.hiddenSize, k: VisionConfig.patchDimension)
            try primitives.encodeAddPosition(
                commandBuffer: commandBuffer,
                hidden: scratch.hiddenA,
                table: fixed.buffer, tableOffset: try fixed.offset(of: positionTable),
                positions: input.positionsInt32x2, rows: rows)
            try commitPipelined(commandBuffer, retaining: fixed)
        }

        for layer in 0..<VisionConfig.numLayers {
            try checkCancellation()
            let prefix = "vision_tower.encoder.layers.\(layer)."
            let names = store.tensorNames(withPrefix: prefix)
            guard names.count == 13 else {
                throw VisionRuntimeError.invalidInput(
                    "layer \(layer) expected 13 tensors, found \(names.count)")
            }
            let weights = try mapRegion(names)
            let commandBuffer = try makeCommandBuffer()
            func offset(_ suffix: String) throws -> Int {
                try weights.offset(of: prefix + suffix)
            }

            try primitives.encodeRMSNorm(
                commandBuffer: commandBuffer,
                input: scratch.hiddenA, output: scratch.hiddenB,
                weightBuffer: weights.buffer,
                weightOffset: try offset("input_layernorm.weight"), rows: rows)
            for (suffix, output) in [
                ("self_attn.q_proj.linear.weight", attention.fusedQ),
                ("self_attn.k_proj.linear.weight", attention.fusedK),
                ("self_attn.v_proj.linear.weight", attention.fusedV),
            ] {
                try linear.encode(
                    commandBuffer: commandBuffer,
                    input: scratch.hiddenB,
                    weights: weights.buffer, weightsOffset: try offset(suffix),
                    output: output,
                    m: scratch.paddedRows, n: VisionConfig.hiddenSize, k: VisionConfig.hiddenSize,
                    paddedHeadStore: paddedHeadStore)
            }
            try primitives.encodeQKV(
                commandBuffer: commandBuffer,
                q: attention.fusedQ, k: attention.fusedK, v: attention.fusedV,
                weights: weights.buffer,
                qWeightOffset: try offset("self_attn.q_norm.weight"),
                kWeightOffset: try offset("self_attn.k_norm.weight"),
                positions: input.positionsInt32x2,
                rows: rows, heads: VisionConfig.numHeads,
                paddedRowStride: scratch.paddedRows)
            try attention.encode(commandBuffer: commandBuffer,
                             q: attention.fusedQ, k: attention.fusedK, v: attention.fusedV,
                             output: scratch.attention,
                             sequenceLength: rows, numHeads: VisionConfig.numHeads,
                             inputRowStride: scratch.paddedRows)
            try linear.encode(
                commandBuffer: commandBuffer,
                input: scratch.attention,
                weights: weights.buffer,
                weightsOffset: try offset("self_attn.o_proj.linear.weight"),
                output: scratch.q,
                m: scratch.paddedRows, n: VisionConfig.hiddenSize, k: VisionConfig.hiddenSize)
            try primitives.encodePostnormResidual(
                commandBuffer: commandBuffer,
                residual: scratch.hiddenA, branch: scratch.q,
                weightBuffer: weights.buffer,
                weightOffset: try offset("post_attention_layernorm.weight"),
                output: scratch.hiddenB, rows: rows)
            try primitives.encodeRMSNorm(
                commandBuffer: commandBuffer,
                input: scratch.hiddenB, output: scratch.q,
                weightBuffer: weights.buffer,
                weightOffset: try offset("pre_feedforward_layernorm.weight"), rows: rows)
            try linear.encode(
                commandBuffer: commandBuffer,
                input: scratch.q,
                weights: weights.buffer,
                weightsOffset: try offset("mlp.gate_proj.linear.weight"),
                output: scratch.gate,
                m: scratch.paddedRows, n: VisionConfig.intermediateSize,
                k: VisionConfig.hiddenSize)
            try linear.encode(
                commandBuffer: commandBuffer,
                input: scratch.q,
                weights: weights.buffer,
                weightsOffset: try offset("mlp.up_proj.linear.weight"),
                output: scratch.gate,
                m: scratch.paddedRows, n: VisionConfig.intermediateSize,
                k: VisionConfig.hiddenSize,
                fusesGeGLU: true)
            try linear.encode(
                commandBuffer: commandBuffer,
                input: scratch.gate,
                weights: weights.buffer,
                weightsOffset: try offset("mlp.down_proj.linear.weight"),
                output: scratch.q,
                m: scratch.paddedRows, n: VisionConfig.hiddenSize, k: VisionConfig.intermediateSize)
            try primitives.encodePostnormResidual(
                commandBuffer: commandBuffer,
                residual: scratch.hiddenB, branch: scratch.q,
                weightBuffer: weights.buffer,
                weightOffset: try offset("post_feedforward_layernorm.weight"),
                output: scratch.hiddenA, rows: rows)
            try commitPipelined(commandBuffer, retaining: weights)
        }

        let outputRows = (patchGridWidth / VisionConfig.poolingKernel)
            * (patchGridHeight / VisionConfig.poolingKernel)
        try checkCancellation()
        let projector = try mapRegion([
            "vision_tower.std_bias",
            "vision_tower.std_scale",
            "embed_vision.embedding_projection.weight",
            "embed_vision.embedding_projection.scales",
            "embed_vision.embedding_projection.biases",
        ])
        guard let features = context.device.makeBuffer(
            length: outputRows * VisionConfig.textHiddenSize * MemoryLayout<Float16>.stride,
            options: .storageModeShared) else {
            throw MetalError.allocationFailed("vision features")
        }
        let commandBuffer = try makeCommandBuffer()
        try primitives.encodePool(
            commandBuffer: commandBuffer,
            input: scratch.hiddenA,
            weights: projector.buffer,
            stdBiasOffset: try projector.offset(of: "vision_tower.std_bias"),
            stdScaleOffset: try projector.offset(of: "vision_tower.std_scale"),
            output: scratch.attention,
            patchWidth: patchGridWidth, patchHeight: patchGridHeight)
        if !usesBF16Projector {
            try primitives.encodeBFloatToHalf(
                commandBuffer: commandBuffer,
                input: scratch.attention, output: scratch.hiddenB,
                count: outputRows * VisionConfig.hiddenSize)
        }
        try projectorKernel.encode(
            commandBuffer: commandBuffer,
            weights: projector.buffer,
            weightsOffset: try projector.offset(
                of: "embed_vision.embedding_projection.weight"),
            scales: projector.buffer,
            scalesOffset: try projector.offset(
                of: "embed_vision.embedding_projection.scales"),
            biases: projector.buffer,
            biasesOffset: try projector.offset(
                of: "embed_vision.embedding_projection.biases"),
            x: usesBF16Projector ? scratch.attention : scratch.hiddenB,
            y: features,
            m: outputRows, n: VisionConfig.textHiddenSize, k: VisionConfig.hiddenSize)
        try commitPipelined(commandBuffer, retaining: projector)
        while !inFlight.isEmpty { try drainOldest() }

        return VisionFeatures(buffer: features, tokenCount: outputRows,
                              hiddenSize: VisionConfig.textHiddenSize)
    }

    private func makeCommandBuffer() throws -> any MTLCommandBuffer {
        guard let value = context.queue.makeCommandBuffer() else { throw MetalError.noQueue }
        return value
    }

    private func drainGPU() throws {
        let commandBuffer = try makeCommandBuffer()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let detail = metalCommandBufferFailureDetail(label: commandBuffer.label,
                                                        status: commandBuffer.status,
                                                        error: commandBuffer.error) {
            throw VisionRuntimeError.commandFailed(detail)
        }
    }
}
