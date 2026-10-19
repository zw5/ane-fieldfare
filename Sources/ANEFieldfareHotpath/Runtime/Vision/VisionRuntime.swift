import Darwin
public import Foundation
public import Metal
import ANEFieldfareImportedFormat

public enum VisionRuntimeError: Error, CustomStringConvertible {
    case disabled
    case invalidInput(String)
    case commandFailed(String)
    case unsupportedKernel(String)

    public var description: String {
        switch self {
        case .disabled: "vision runtime is unavailable"
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
    public let gpuNanoseconds: UInt64
    public let scratchBytes: Int
    public let projectorPath: MPPPrefillInt4QMM.Path
    public let expertResidencyTransition: VisionExpertResidencyTransition?
    public let preprocessing: VisionPreprocessingReport?
}

public struct VisionPreprocessingReport: Sendable, Equatable {
    public let metadata: VisionImageMetadata
    public let geometry: Gemma4ImageGeometry
    public let wallNanoseconds: UInt64
    public let allocatedBytes: Int
}

public final class VisionRuntime {
    public let config = VisionConfig()

    private let context: MetalContext
    private let store: VisionWeightStore
    private let useLease: VisionPackUseLease
    private let linear: VisionLinearBF16
    private let primitives: VisionPrimitives
    private let attention: VisionAttention
    private let usesBF16Projector: Bool
    /// Mapped weight regions retained across images under `.keepReady`. Mapping
    /// and first-touching the 1.08 GB the tower reads costs the same every image
    /// otherwise; holding the mappings pays it once. Pages are clean, file-backed
    /// and evictable, like the routed expert pool.
    private var cachedRegions: [String: VisionMappedWeightRegion] = [:]


    public static func open(
        textModelURL: URL,
        context: MetalContext,
        visionPackURL: URL? = nil
    ) throws -> VisionRuntime {
        try requireSupportedDevice(context.device)
        let manifest = try ManifestReader.load(
            directoryURL: textModelURL,
            expecting: .gemma4_26B_A4B)
        guard let source = manifest.sourceSnapshotHash else {
            throw VisionRuntimeError.invalidInput("text model has no source identity")
        }
        let textManifestSHA = try Sha256Verifier.hashFile(
            at: textModelURL.appendingPathComponent("manifest.json"))
        let companion = try visionPackURL
            ?? VisionPackLocation.companionURL(forTextModel: textModelURL)
        // The pack has to exist before anything is created beside it. The lease
        // opens its lock with O_CREAT, so taking it first meant a text-only
        // install got a permanent stray dotfile, and a pack on a read-only
        // volume failed with "could not open use lock ... errno 30" — a lock
        // error standing in for a perfectly valid pack.
        guard FileManager.default.fileExists(atPath: companion.path) else {
            // Absence is not invalid metadata. Saying "metadata is invalid" for
            // a pack nobody installed sends the reader looking for corruption
            // in a file that is not there.
            throw VisionPackError.packNotFound(companion.path)
        }
        // ... and it has to look like a pack, for the same reason the existence
        // check above exists. A directory that is merely *there* — a `.partial`
        // download, a mistyped path — also got a permanent stray lock file
        // beside it, because the lease is created before the store validates
        // anything. One `lstat` moves that decision ahead of the `O_CREAT`.
        let manifestPath = companion
            .appendingPathComponent(GTurboVisionFormatV1.manifestFile).path
        guard FileManager.default.fileExists(atPath: manifestPath) else {
            throw VisionPackError.invalidMetadata(
                "no \(GTurboVisionFormatV1.manifestFile) in \(companion.path)")
        }
        let useLease = try VisionPackUseLease.acquireShared(companionURL: companion)
        let store = try VisionWeightStore.open(
            directoryURL: companion,
            compatibleTextSourceSnapshotHash: source,
            compatibleTextManifestSha256: textManifestSHA)
        return try VisionRuntime(context: context, store: store, useLease: useLease)
    }

    public static func isSupported(on device: any MTLDevice) -> Bool {
        device.supportsFamily(.apple8)
    }

    public static func requireSupportedDevice(_ device: any MTLDevice) throws {
        try requireSupportedDevice(supportsApple8: isSupported(on: device))
    }

    static func requireSupportedDevice(supportsApple8: Bool) throws {
        guard supportsApple8 else {
            throw VisionRuntimeError.unsupportedKernel(
                "the image tower requires an M2 or newer Apple Silicon Mac")
        }
    }

    init(context: MetalContext, store: VisionWeightStore,
         useLease: consuming VisionPackUseLease) throws {
        self.context = context
        self.store = store
        self.useLease = useLease
        self.linear = try VisionLinearBF16(context: context)
        self.primitives = try VisionPrimitives(context: context)
        self.attention = try VisionAttention(context: context)
        // Pre-Apple10 GPUs convert the pooled features to FP16 for the projector.
        self.usesBF16Projector = context.device.supportsFamily(.apple10)
        self.imageResize = try? VisionResize(context: context)
        // Built once per runtime; the tensorops library behind it is shared
        // with the linear and attention kernels through
        // `MetalContext.privateLibrary`'s cache instead of compiled per
        // constructor.
        self.projectorKernel = MPPPrefillInt4QMM(
            context: context,
            variant: usesBF16Projector
                ? .apple10BF16
                : context.device.supportsFamily(.apple10) ? .apple10V1 : .control)
    }

    private let projectorKernel: MPPPrefillInt4QMM
    private let imageResize: VisionResize?

    /// The tensor groups the encode path maps, named once so prewarm cannot
    /// map a different grouping into the same cache.
    static let patchEmbedderTensorNames = [
        "vision_tower.patch_embedder.input_proj.weight",
        "vision_tower.patch_embedder.position_embedding_table",
    ]
    static let projectorTensorNames = [
        "vision_tower.std_bias",
        "vision_tower.std_scale",
        "embed_vision.embedding_projection.weight",
        "embed_vision.embedding_projection.scales",
        "embed_vision.embedding_projection.biases",
    ]

    public func encodeImage(
        at fileURL: URL,
        languageModel: Model? = nil,
        residencyPolicy: VisionResidencyPolicy = .defaultPolicy,
        checkCancellation: () throws -> Void = {}
    ) throws -> VisionFeatures {
        let preprocessor = Gemma4ImagePreprocessor(device: context.device, config: config,
                                                   gpuResize: imageResize)
        return try encodeImage(
            plan: try preprocessor.plan(fileURL: fileURL),
            languageModel: languageModel,
            residencyPolicy: residencyPolicy,
            checkCancellation: checkCancellation)
    }

    /// Encodes an image the caller has already planned.
    ///
    /// Callers plan first to learn the token count, and re-planning here opened
    /// the file a second time for another ImageIO sniff, metadata parse and
    /// marker walk — per image, per turn. It also validated the encode against
    /// metadata that a file swapped between the two opens would no longer match.
    public func encodeImage(
        plan: VisionImagePlan,
        languageModel: Model? = nil,
        residencyPolicy: VisionResidencyPolicy = .defaultPolicy,
        checkCancellation: () throws -> Void = {}
    ) throws -> VisionFeatures {
        try encode(languageModel: languageModel,
                   residencyPolicy: residencyPolicy,
                   checkCancellation: checkCancellation) { preprocessor in
            try preprocessor.preprocess(plan)
        }
    }

    private func encode(
        languageModel: Model?,
        residencyPolicy: VisionResidencyPolicy,
        checkCancellation: () throws -> Void,
        pixels: (Gemma4ImagePreprocessor) throws -> VisionPixelBuffer
    ) throws -> VisionFeatures {
        try checkCancellation()
        let preprocessor = Gemma4ImagePreprocessor(device: context.device, config: config,
                                                   gpuResize: imageResize)
        try checkCancellation()
        let gpuDrainNanoseconds = try languageModel != nil && residencyPolicy == .onDemand
            ? drainGPU() : 0
        try checkCancellation()
        let transition = languageModel?.prepareExpertResidencyForVision(
            residencyPolicy, gpuDrainNanoseconds: gpuDrainNanoseconds)
        try checkCancellation()
        let input = try pixels(preprocessor)
        return try encodePreparedPatches(
            patchesBF16: input.patchesBF16,
            positionsInt32x2: input.positionsInt32x2,
            patchGridWidth: input.geometry.patchGridWidth,
            patchGridHeight: input.geometry.patchGridHeight,
            expertResidencyTransition: transition,
            preprocessing: VisionPreprocessingReport(
                metadata: input.metadata,
                geometry: input.geometry,
                wallNanoseconds: input.wallNanoseconds,
                allocatedBytes: input.allocatedBytes),
            retainsWeightRegions: residencyPolicy == .keepReady,
            checkCancellation: checkCancellation)
    }

    private func encodePreparedPatches(
        patchesBF16: any MTLBuffer,
        positionsInt32x2: any MTLBuffer,
        patchGridWidth: Int,
        patchGridHeight: Int,
        expertResidencyTransition: VisionExpertResidencyTransition?,
        preprocessing: VisionPreprocessingReport?,
        retainsWeightRegions: Bool,
        checkCancellation: () throws -> Void
    ) throws -> VisionFeatures {
        if !retainsWeightRegions { cachedRegions.removeAll(keepingCapacity: false) }
        try checkCancellation()
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
        var gpuNanoseconds: UInt64 = 0
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
            gpuNanoseconds += UInt64(
                max(0, entry.buffer.gpuEndTime - entry.buffer.gpuStartTime)
                    * 1_000_000_000)
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

        let patchWeight = Self.patchEmbedderTensorNames[0]
        let positionTable = Self.patchEmbedderTensorNames[1]
        do {
            let fixed = try mapRegion(Self.patchEmbedderTensorNames)
            let commandBuffer = try makeCommandBuffer()
            try scratch.encodeClear(commandBuffer: commandBuffer)
            attention.encodeClearPadding(commandBuffer: commandBuffer)
            primitives.encodeNormalize(commandBuffer: commandBuffer,
                                       input: patchesBF16,
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
                positions: positionsInt32x2, rows: rows)
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
                positions: positionsInt32x2,
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
        let projector = try mapRegion(Self.projectorTensorNames)
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
        let projectorMetadata = projectorKernel.encode(
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
            m: outputRows, n: config.textHiddenSize, k: config.hiddenSize)
        guard projectorMetadata.path != .fallback else {
            throw VisionRuntimeError.unsupportedKernel("INT4 projector fell back")
        }
        try commitPipelined(commandBuffer, retaining: projector)
        while !inFlight.isEmpty { try drainOldest() }

        return VisionFeatures(buffer: features,
                              tokenCount: outputRows,
                              hiddenSize: config.textHiddenSize,
                              gpuNanoseconds: gpuNanoseconds,
                              scratchBytes: scratch.allocatedBytes
                                  + attention.additionalScratchBytes,
                              projectorPath: projectorMetadata.path,
                              expertResidencyTransition: expertResidencyTransition,
                              preprocessing: preprocessing)
    }

    private func makeCommandBuffer() throws -> any MTLCommandBuffer {
        guard let value = context.queue.makeCommandBuffer() else { throw MetalError.noQueue }
        return value
    }

    private func drainGPU() throws -> UInt64 {
        let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let commandBuffer = try makeCommandBuffer()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let detail = metalCommandBufferFailureDetail(label: commandBuffer.label,
                                                        status: commandBuffer.status,
                                                        error: commandBuffer.error) {
            throw VisionRuntimeError.commandFailed(detail)
        }
        return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started
    }

}
