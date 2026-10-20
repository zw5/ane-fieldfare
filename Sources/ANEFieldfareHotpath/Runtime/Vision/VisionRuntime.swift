public import Foundation
public import Metal
import ANEFieldfareModelFormat

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
    public let config = VisionConfig()

    private let context: MetalContext
    private let store: VisionWeightStore
    private let linear: VisionLinearBF16
    private let primitives: VisionPrimitives
    private let attention: VisionAttention
    private let usesBF16Projector: Bool
    private let projectorKernel: MPPPrefillInt4QMM
    private let imageResize: VisionResize?
    /// Mapped weight regions retained across images under `.keepReady`. Mapping
    /// and first-touching the 1.08 GB the tower reads costs the same every image
    /// otherwise; holding the mappings pays it once. Pages are clean, file-backed
    /// and evictable, like the routed expert pool.
    private var cachedRegions: [String: VisionMappedWeightRegion] = [:]

    /// Opens the vision weights of the model at `textModelURL`; throws
    /// `VisionPackError.notInstalled` for a model installed without them.
    public static func open(
        textModelURL: URL,
        context: MetalContext
    ) throws -> VisionRuntime {
        try requireSupportedDevice(context.device)
        let directory = try ModelDirectory(rootURL: textModelURL)
        let manifest = try ModelManifest.load(directory, path: textModelURL.path, expecting: .gemma4_26B_A4B)
        let store = try VisionWeightStore.open(directory: directory, directoryURL: textModelURL,
                                               manifest: manifest)
        return try VisionRuntime(context: context, store: store)
    }

    public static func requireSupportedDevice(_ device: any MTLDevice) throws {
        guard device.supportsFamily(.apple8) else {
            throw VisionRuntimeError.unsupportedKernel(
                "the image tower requires an M2 or newer Apple Silicon Mac")
        }
    }

    private init(context: MetalContext, store: VisionWeightStore) throws {
        self.context = context
        self.store = store
        self.linear = try VisionLinearBF16(context: context)
        self.primitives = try VisionPrimitives(context: context)
        self.attention = try VisionAttention(context: context)
        // Pre-Apple10 GPUs convert the pooled features to FP16 for the projector.
        self.usesBF16Projector = context.device.supportsFamily(.apple10)
        self.imageResize = try? VisionResize(context: context)
        self.projectorKernel = MPPPrefillInt4QMM(
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
        _ = languageModel.prepareExpertResidencyForVision(residencyPolicy)
        try checkCancellation()
        return try encodePatches(
            try makePreprocessor().preprocess(plan),
            retainsWeightRegions: residencyPolicy == .keepReady,
            checkCancellation: checkCancellation)
    }

    private func makePreprocessor() -> Gemma4ImagePreprocessor {
        Gemma4ImagePreprocessor(device: context.device, config: config, gpuResize: imageResize)
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
        let scratch = try VisionScratch(device: context.device, rows: rows, config: config)
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
            attention.encodeClearPadding(commandBuffer: commandBuffer)
            primitives.encodeNormalize(commandBuffer: commandBuffer,
                                       input: input.patchesBF16,
                                       output: scratch.normalizedPatches,
                                       rows: rows, paddedRows: scratch.paddedRows)
            linear.encode(
                commandBuffer: commandBuffer,
                input: scratch.normalizedPatches,
                weights: fixed.buffer, weightsOffset: try fixed.offset(of: patchWeight),
                output: scratch.hiddenA,
                m: scratch.paddedRows, n: config.hiddenSize, k: config.patchDimension)
            primitives.encodeAddPosition(
                commandBuffer: commandBuffer,
                hidden: scratch.hiddenA,
                table: fixed.buffer, tableOffset: try fixed.offset(of: positionTable),
                positions: input.positionsInt32x2, rows: rows)
            try commitPipelined(commandBuffer, retaining: fixed)
        }

        for layer in 0..<config.numLayers {
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

            primitives.encodeRMSNorm(
                commandBuffer: commandBuffer,
                input: scratch.hiddenA, output: scratch.hiddenB,
                weightBuffer: weights.buffer,
                weightOffset: try offset("input_layernorm.weight"), rows: rows)
            for (suffix, output) in [
                ("self_attn.q_proj.linear.weight", attention.fusedQ),
                ("self_attn.k_proj.linear.weight", attention.fusedK),
                ("self_attn.v_proj.linear.weight", attention.fusedV),
            ] {
                linear.encode(
                    commandBuffer: commandBuffer,
                    input: scratch.hiddenB,
                    weights: weights.buffer, weightsOffset: try offset(suffix),
                    output: output,
                    m: scratch.paddedRows, n: config.hiddenSize, k: config.hiddenSize,
                    paddedHeadStore: paddedHeadStore)
            }
            primitives.encodeQKV(
                commandBuffer: commandBuffer,
                q: attention.fusedQ, k: attention.fusedK, v: attention.fusedV,
                weights: weights.buffer,
                qWeightOffset: try offset("self_attn.q_norm.weight"),
                kWeightOffset: try offset("self_attn.k_norm.weight"),
                positions: input.positionsInt32x2,
                rows: rows, heads: config.numHeads,
                paddedRowStride: scratch.paddedRows)
            attention.encode(commandBuffer: commandBuffer,
                             q: attention.fusedQ, k: attention.fusedK, v: attention.fusedV,
                             output: scratch.attention,
                             sequenceLength: rows, numHeads: config.numHeads,
                             inputRowStride: scratch.paddedRows)
            linear.encode(
                commandBuffer: commandBuffer,
                input: scratch.attention,
                weights: weights.buffer,
                weightsOffset: try offset("self_attn.o_proj.linear.weight"),
                output: scratch.q,
                m: scratch.paddedRows, n: config.hiddenSize, k: config.hiddenSize)
            primitives.encodePostnormResidual(
                commandBuffer: commandBuffer,
                residual: scratch.hiddenA, branch: scratch.q,
                weightBuffer: weights.buffer,
                weightOffset: try offset("post_attention_layernorm.weight"),
                output: scratch.hiddenB, rows: rows)
            primitives.encodeRMSNorm(
                commandBuffer: commandBuffer,
                input: scratch.hiddenB, output: scratch.q,
                weightBuffer: weights.buffer,
                weightOffset: try offset("pre_feedforward_layernorm.weight"), rows: rows)
            linear.encode(
                commandBuffer: commandBuffer,
                input: scratch.q,
                weights: weights.buffer,
                weightsOffset: try offset("mlp.gate_proj.linear.weight"),
                output: scratch.gate,
                m: scratch.paddedRows, n: config.intermediateSize,
                k: config.hiddenSize)
            linear.encode(
                commandBuffer: commandBuffer,
                input: scratch.q,
                weights: weights.buffer,
                weightsOffset: try offset("mlp.up_proj.linear.weight"),
                output: scratch.gate,
                m: scratch.paddedRows, n: config.intermediateSize,
                k: config.hiddenSize,
                fusesGeGLU: true)
            linear.encode(
                commandBuffer: commandBuffer,
                input: scratch.gate,
                weights: weights.buffer,
                weightsOffset: try offset("mlp.down_proj.linear.weight"),
                output: scratch.q,
                m: scratch.paddedRows, n: config.hiddenSize, k: config.intermediateSize)
            primitives.encodePostnormResidual(
                commandBuffer: commandBuffer,
                residual: scratch.hiddenB, branch: scratch.q,
                weightBuffer: weights.buffer,
                weightOffset: try offset("post_feedforward_layernorm.weight"),
                output: scratch.hiddenA, rows: rows)
            try commitPipelined(commandBuffer, retaining: weights)
        }

        let outputRows = (patchGridWidth / config.poolingKernel)
            * (patchGridHeight / config.poolingKernel)
        try checkCancellation()
        let projector = try mapRegion([
            "vision_tower.std_bias",
            "vision_tower.std_scale",
            "embed_vision.embedding_projection.weight",
        ])
        guard let features = context.device.makeBuffer(
            length: outputRows * config.textHiddenSize * MemoryLayout<Float16>.stride,
            options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        guard projectorKernel.isAvailable else {
            throw VisionRuntimeError.unsupportedKernel(
                projectorKernel.unavailableReason ?? "INT4 projector unavailable")
        }
        let commandBuffer = try makeCommandBuffer()
        primitives.encodePool(
            commandBuffer: commandBuffer,
            input: scratch.hiddenA,
            weights: projector.buffer,
            stdBiasOffset: try projector.offset(of: "vision_tower.std_bias"),
            stdScaleOffset: try projector.offset(of: "vision_tower.std_scale"),
            output: scratch.attention,
            patchWidth: patchGridWidth, patchHeight: patchGridHeight)
        if !usesBF16Projector {
            primitives.encodeBFloatToHalf(
                commandBuffer: commandBuffer,
                input: scratch.attention, output: scratch.hiddenB,
                count: outputRows * config.hiddenSize)
        }
        let projectorPath = projectorKernel.encode(
            commandBuffer: commandBuffer,
            weights: projector.buffer,
            weightsOffset: try projector.offset(
                of: "embed_vision.embedding_projection.weight"),
            scales: projector.buffer,
            scalesOffset: try projector.scalesOffset(
                of: "embed_vision.embedding_projection.weight"),
            x: usesBF16Projector ? scratch.attention : scratch.hiddenB,
            y: features,
            m: outputRows, n: config.textHiddenSize, k: config.hiddenSize)
        guard projectorPath != .fallback else {
            throw VisionRuntimeError.unsupportedKernel("INT4 projector fell back")
        }
        try commitPipelined(commandBuffer, retaining: projector)
        while !inFlight.isEmpty { try drainOldest() }

        return VisionFeatures(buffer: features, tokenCount: outputRows,
                              hiddenSize: config.textHiddenSize)
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
