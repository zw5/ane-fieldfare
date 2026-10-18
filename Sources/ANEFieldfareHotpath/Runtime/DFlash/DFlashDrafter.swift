import Foundation
import Metal

public enum DFlashError: Error, CustomStringConvertible {
    case invalidCheckpoint(String)
    case invalidState(String)

    public var description: String {
        switch self {
        case .invalidCheckpoint(let detail): return "DFlash drafter checkpoint: \(detail)"
        case .invalidState(let detail): return "DFlash drafter: \(detail)"
        }
    }
}

/// Configuration read from a z-lab DFlash drafter's `config.json`.
public struct DFlashDrafterConfig: Sendable, Equatable {
    public let hiddenSize: Int
    public let layers: Int
    public let heads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let intermediateSize: Int
    public let blockSize: Int
    public let contextLayers: [Int]
    public let maskTokenID: Int32
    public let slidingWindow: Int
    public let slidingLayers: [Bool]
    public let ropeTheta: Float
    public let rmsEps: Float

    static func load(_ url: URL) throws -> DFlashDrafterConfig {
        guard let json = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any],
              let dflash = json["dflash_config"] as? [String: Any],
              let hidden = json["hidden_size"] as? Int, let layers = json["num_hidden_layers"] as? Int,
              let heads = json["num_attention_heads"] as? Int, let kvHeads = json["num_key_value_heads"] as? Int,
              let headDim = json["head_dim"] as? Int, let intermediate = json["intermediate_size"] as? Int,
              let contextLayers = dflash["target_layer_ids"] as? [Int], let mask = dflash["mask_token_id"] as? Int,
              let types = json["layer_types"] as? [String], types.count == layers else {
            throw DFlashError.invalidCheckpoint("config.json is missing DFlash fields")
        }
        guard headDim == 128, heads % kvHeads == 0, json["model_type"] as? String == "qwen3" else {
            throw DFlashError.invalidCheckpoint("unsupported drafter shape")
        }
        return DFlashDrafterConfig(
            hiddenSize: hidden, layers: layers, heads: heads, kvHeads: kvHeads, headDim: headDim,
            intermediateSize: intermediate, blockSize: json["block_size"] as? Int ?? 16,
            contextLayers: contextLayers, maskTokenID: Int32(mask),
            slidingWindow: json["sliding_window"] as? Int ?? Int.max,
            slidingLayers: types.map { $0 == "sliding_attention" },
            ropeTheta: Float((json["rope_theta"] as? Double) ?? 1_000_000),
            rmsEps: Float((json["rms_norm_eps"] as? Double) ?? 1e-6))
    }
}

/// The DFlash block-diffusion drafter on Metal. It shares the target's tied
/// INT4 embedding for its input and output head, keeps its own context K/V
/// built from the target's captured layer outputs, and drafts a block of
/// tokens after an anchor in one forward pass.
public final class DFlashDrafter {
    public let config: DFlashDrafterConfig
    /// Positions whose target features are in the drafter's context cache.
    public private(set) var contextLength = 0
    public let maxContext: Int
    public var vocabSize: Int { model.config.vocabSize }

    private let ctx: MetalContext
    private let model: Model
    private let weights: MTLBuffer
    private let tensorOffsets: [String: Int]
    private let gemm: VisionLinearBF16
    private let head: LMHeadChainInt4
    private let embedPSO, normF32PSO, normBF16PSO, headNormRopePSO, attentionPSO,
                siluPSO, residualPSO, toF16PSO, gatherPSO: MTLComputePipelineState
    private let kCache: [MTLBuffer]
    private let vCache: [MTLBuffer]

    private static let appendRows = 512
    private static let gemmRowPad = 64
    // Context-append scratch.
    private let staging: MTLBuffer
    private let feature, contextHidden, contextK, contextV, contextKRoped: MTLBuffer
    // Block scratch.
    private let residual, normed, q, kBlock, vBlock, attention, delta, gate, up, act, headRows,
                ids, draftTokens: MTLBuffer

    public init(directory: URL, model: Model, context: MetalContext, maxContext: Int) throws {
        let config = try DFlashDrafterConfig.load(directory.appendingPathComponent("config.json"))
        guard config.hiddenSize == model.config.hiddenSize,
              config.contextLayers.allSatisfy({ $0 >= 0 && $0 < model.config.numLayers }),
              config.blockSize <= LMHeadChainInt4.maxGreedyRows else {
            throw DFlashError.invalidCheckpoint("drafter does not match the target")
        }
        self.config = config
        self.ctx = context
        self.model = model
        self.maxContext = maxContext

        let (weights, offsets) = try Self.loadSafetensors(
            directory.appendingPathComponent("model.safetensors"), device: context.device)
        self.weights = weights
        self.tensorOffsets = offsets
        self.gemm = try VisionLinearBF16(context: context)
        self.head = try LMHeadChainInt4(context: context, maxD: config.hiddenSize,
                                        maxVocab: model.config.vocabSize)
        embedPSO = try context.pipeline("dflash_embed_int4")
        normF32PSO = try context.pipeline("dflash_rmsnorm_rows_f32")
        normBF16PSO = try context.pipeline("dflash_rmsnorm_rows_bf16")
        headNormRopePSO = try context.pipeline("dflash_head_norm_rope")
        attentionPSO = try context.pipeline("dflash_block_attention")
        siluPSO = try context.pipeline("dflash_silu_mul")
        residualPSO = try context.pipeline("dflash_residual_add")
        toF16PSO = try context.pipeline("dflash_f32_to_f16")
        gatherPSO = try context.pipeline("dflash_gather_context")

        let device = context.device
        func buffer(_ bytes: Int, _ label: String, shared: Bool = false) throws -> MTLBuffer {
            guard let made = device.makeBuffer(length: max(bytes, 16),
                                               options: shared ? .storageModeShared : .storageModePrivate) else {
                throw DFlashError.invalidState("allocation failed: \(label)")
            }
            made.label = "dflash.\(label)"
            return made
        }
        let D = config.hiddenSize, kvWidth = config.kvHeads * config.headDim
        let qWidth = config.heads * config.headDim, F = config.intermediateSize
        let bf = 2, pad = Self.gemmRowPad
        kCache = try (0..<config.layers).map { try buffer(maxContext * kvWidth * bf, "k\($0)") }
        vCache = try (0..<config.layers).map { try buffer(maxContext * kvWidth * bf, "v\($0)") }
        let rows = Self.appendRows
        staging = try buffer(config.contextLayers.count * rows * D * 2, "staging", shared: true)
        feature = try buffer(rows * config.contextLayers.count * D * bf, "feature")
        contextHidden = try buffer(rows * D * bf, "context")
        contextK = try buffer(rows * kvWidth * bf, "contextK")
        contextV = try buffer(rows * kvWidth * bf, "contextV")
        contextKRoped = try buffer(rows * kvWidth * bf, "contextKRoped")
        residual = try buffer(pad * D * 4, "residual")
        normed = try buffer(pad * D * bf, "normed")
        q = try buffer(pad * qWidth * bf, "q")
        kBlock = try buffer(pad * kvWidth * bf, "kBlock")
        vBlock = try buffer(pad * kvWidth * bf, "vBlock")
        attention = try buffer(pad * qWidth * bf, "attention")
        delta = try buffer(pad * D * bf, "delta")
        gate = try buffer(pad * F * bf, "gate")
        up = try buffer(pad * F * bf, "up")
        act = try buffer(pad * F * bf, "act")
        headRows = try buffer(pad * D * 2, "headRows")
        ids = try buffer(pad * 4, "ids", shared: true)
        draftTokens = try buffer(pad * 4, "draftTokens", shared: true)
        // GEMM inputs are read in 64-row tiles; keep the pad rows zero.
        guard let cb = context.queue.makeCommandBuffer(), let blit = cb.makeBlitCommandEncoder() else {
            throw DFlashError.invalidState("command buffer")
        }
        for b in [normed, attention, act, feature, contextHidden] { blit.fill(buffer: b, range: 0..<b.length, value: 0) }
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    /// Forget context past `length`, e.g. after a rewind.
    public func truncateContext(to length: Int) throws {
        guard length >= 0, length <= contextLength else {
            throw DFlashError.invalidState("cannot truncate context \(contextLength) to \(length)")
        }
        contextLength = length
    }

    public func reset() { contextLength = 0 }

    /// Appends target features for positions `[contextLength, contextLength + rows)`.
    /// `captured` is `[contextLayers][stride][hidden]` FP16 as `PrefillCapture`
    /// delivers it; only its first `rows` rows per layer are used.
    public func appendContext(captured: UnsafeBufferPointer<Float16>, stride: Int, rows: Int) throws {
        let D = config.hiddenSize, layers = config.contextLayers.count
        guard rows >= 0, rows <= stride, captured.count >= layers * stride * D,
              contextLength + rows <= maxContext else {
            throw DFlashError.invalidState("context append of \(rows) rows exceeds capacity")
        }
        var done = 0
        while done < rows {
            let n = min(Self.appendRows, rows - done)
            let dst = staging.contents().assumingMemoryBound(to: Float16.self)
            for layer in 0..<layers {
                (dst + layer * n * D).update(from: captured.baseAddress! + (layer * stride + done) * D,
                                             count: n * D)
            }
            try encodeAppend(rows: n)
            done += n
        }
    }

    private func encodeAppend(rows n: Int) throws {
        let D = config.hiddenSize, layers = config.contextLayers.count
        let kvWidth = config.kvHeads * config.headDim
        guard let cb = ctx.queue.makeCommandBuffer() else { throw DFlashError.invalidState("command buffer") }
        cb.label = "dflash append ctx=\(contextLength) rows=\(n)"
        dispatch(cb, gatherPSO, [staging, feature], [UInt32(n), UInt32(layers), UInt32(D)],
                 grid: MTLSize(width: layers * D, height: n, depth: 1))
        gemm.encode(commandBuffer: cb, input: feature, weights: weights, weightsOffset: try offset("fc.weight"),
                    output: contextHidden, m: n, n: D, k: layers * D)
        try encodeNorm(cb, normBF16PSO, x: contextHidden, weight: "hidden_norm.weight", out: contextHidden, rows: n)
        for l in 0..<config.layers {
            let p = "layers.\(l).self_attn."
            gemm.encode(commandBuffer: cb, input: contextHidden, weights: weights,
                        weightsOffset: try offset(p + "k_proj.weight"), output: contextK, m: n, n: kvWidth, k: D)
            gemm.encode(commandBuffer: cb, input: contextHidden, weights: weights,
                        weightsOffset: try offset(p + "v_proj.weight"), output: contextV, m: n, n: kvWidth, k: D)
            try encodeHeadNormRope(cb, x: contextK, weight: p + "k_norm.weight",
                                   out: kCache[l], outOffset: contextLength * kvWidth * 2,
                                   rows: n * config.kvHeads, heads: config.kvHeads, pos0: contextLength)
            guard let blit = cb.makeBlitCommandEncoder() else { throw DFlashError.invalidState("blit") }
            blit.copy(from: contextV, sourceOffset: 0, to: vCache[l],
                      destinationOffset: contextLength * kvWidth * 2, size: n * kvWidth * 2)
            blit.endEncoding()
        }
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw DFlashError.invalidState("append failed: \(error)") }
        contextLength += n
    }

    /// Drafts `count - 1` tokens after `anchor`, which sits at position
    /// `contextLength`. Greedy over the drafter's logits.
    public func propose(anchor: Int32, count: Int) throws -> [Int32] {
        guard count >= 2, count <= config.blockSize else {
            throw DFlashError.invalidState("block of \(count) outside 2...\(config.blockSize)")
        }
        guard contextLength + count <= maxContext else { throw DFlashError.invalidState("context full") }
        let D = config.hiddenSize, R = count, F = config.intermediateSize
        let kvWidth = config.kvHeads * config.headDim, qWidth = config.heads * config.headDim
        let idPtr = ids.contents().assumingMemoryBound(to: UInt32.self)
        idPtr[0] = UInt32(bitPattern: anchor)
        for r in 1..<R { idPtr[r] = UInt32(bitPattern: config.maskTokenID) }

        guard let cb = ctx.queue.makeCommandBuffer() else { throw DFlashError.invalidState("command buffer") }
        cb.label = "dflash propose ctx=\(contextLength) block=\(R)"
        let emb = model.embedding
        if let enc = cb.makeComputeCommandEncoder() {
            enc.setComputePipelineState(embedPSO)
            enc.setBuffer(emb.buffer, offset: Int(emb.offset), index: 0)
            enc.setBuffer(emb.buffer, offset: Int(emb.scaleOffset), index: 1)
            enc.setBuffer(emb.buffer, offset: Int(emb.biasOffset), index: 2)
            enc.setBuffer(ids, offset: 0, index: 3)
            enc.setBuffer(residual, offset: 0, index: 4)
            var d = UInt32(D), scale = Float(D).squareRoot()
            enc.setBytes(&d, length: 4, index: 5)
            enc.setBytes(&scale, length: 4, index: 6)
            enc.dispatchThreads(MTLSize(width: D, height: R, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }
        for l in 0..<config.layers {
            let p = "layers.\(l)."
            let a = p + "self_attn."
            try encodeNorm(cb, normF32PSO, x: residual, weight: p + "input_layernorm.weight", out: normed, rows: R)
            gemm.encode(commandBuffer: cb, input: normed, weights: weights, weightsOffset: try offset(a + "q_proj.weight"),
                        output: q, m: R, n: qWidth, k: D)
            gemm.encode(commandBuffer: cb, input: normed, weights: weights, weightsOffset: try offset(a + "k_proj.weight"),
                        output: kBlock, m: R, n: kvWidth, k: D)
            gemm.encode(commandBuffer: cb, input: normed, weights: weights, weightsOffset: try offset(a + "v_proj.weight"),
                        output: vBlock, m: R, n: kvWidth, k: D)
            try encodeHeadNormRope(cb, x: q, weight: a + "q_norm.weight", out: q, outOffset: 0,
                                   rows: R * config.heads, heads: config.heads, pos0: contextLength)
            try encodeHeadNormRope(cb, x: kBlock, weight: a + "k_norm.weight", out: kBlock, outOffset: 0,
                                   rows: R * config.kvHeads, heads: config.kvHeads, pos0: contextLength)
            if let enc = cb.makeComputeCommandEncoder() {
                enc.setComputePipelineState(attentionPSO)
                for (i, b) in [q, kCache[l], vCache[l], kBlock, vBlock, attention].enumerated() {
                    enc.setBuffer(b, offset: 0, index: i)
                }
                var params: [UInt32] = [UInt32(config.heads), UInt32(config.kvHeads), UInt32(contextLength),
                                        UInt32(R), config.slidingLayers[l] ? 1 : 0,
                                        UInt32(min(config.slidingWindow, Int(UInt32.max)))]
                for i in params.indices { enc.setBytes(&params[i], length: 4, index: 6 + i) }
                enc.dispatchThreadgroups(MTLSize(width: config.heads, height: R, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                enc.endEncoding()
            }
            gemm.encode(commandBuffer: cb, input: attention, weights: weights, weightsOffset: try offset(a + "o_proj.weight"),
                        output: delta, m: R, n: D, k: qWidth)
            dispatch(cb, residualPSO, [residual, delta], [UInt32(R * D)], grid: MTLSize(width: R * D, height: 1, depth: 1))
            try encodeNorm(cb, normF32PSO, x: residual, weight: p + "post_attention_layernorm.weight", out: normed, rows: R)
            let m = p + "mlp."
            gemm.encode(commandBuffer: cb, input: normed, weights: weights, weightsOffset: try offset(m + "gate_proj.weight"),
                        output: gate, m: R, n: F, k: D)
            gemm.encode(commandBuffer: cb, input: normed, weights: weights, weightsOffset: try offset(m + "up_proj.weight"),
                        output: up, m: R, n: F, k: D)
            dispatch(cb, siluPSO, [gate, up, act], [UInt32(R * F)], grid: MTLSize(width: R * F, height: 1, depth: 1))
            gemm.encode(commandBuffer: cb, input: act, weights: weights, weightsOffset: try offset(m + "down_proj.weight"),
                        output: delta, m: R, n: D, k: F)
            dispatch(cb, residualPSO, [residual, delta], [UInt32(R * D)], grid: MTLSize(width: R * D, height: 1, depth: 1))
        }
        // Rows 1..<R predict the drafts; the head applies the final norm.
        guard let enc = cb.makeComputeCommandEncoder() else { throw DFlashError.invalidState("encoder") }
        enc.setComputePipelineState(toF16PSO)
        enc.setBuffer(residual, offset: D * 4, index: 0)
        enc.setBuffer(headRows, offset: 0, index: 1)
        var n = UInt32((R - 1) * D)
        enc.setBytes(&n, length: 4, index: 2)
        enc.dispatchThreads(MTLSize(width: Int(n), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
        head.encodeGreedyRows(commandBuffer: cb, hidden: headRows, rowCount: R - 1,
                              normWeight: weights, normOffset: try offset("norm.weight"),
                              weights: emb.buffer, weightsOffset: Int(emb.offset),
                              scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                              biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                              outTokens: draftTokens, d: UInt32(D), vocab: UInt32(model.config.vocabSize),
                              rmsEps: config.rmsEps)
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw DFlashError.invalidState("propose failed: \(error)") }
        let out = draftTokens.contents().assumingMemoryBound(to: UInt32.self)
        return (0..<(R - 1)).map { Int32(bitPattern: out[$0]) }
    }

    // MARK: - Encoding helpers

    private func offset(_ name: String) throws -> Int {
        guard let value = tensorOffsets[name] else { throw DFlashError.invalidCheckpoint("missing tensor \(name)") }
        return value
    }

    private func dispatch(_ cb: MTLCommandBuffer, _ pso: MTLComputePipelineState, _ buffers: [MTLBuffer],
                          _ constants: [UInt32], grid: MTLSize) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        for (i, b) in buffers.enumerated() { enc.setBuffer(b, offset: 0, index: i) }
        var values = constants
        for i in values.indices { enc.setBytes(&values[i], length: 4, index: buffers.count + i) }
        enc.dispatchThreads(grid, threadsPerThreadgroup: MTLSize(width: min(256, grid.width), height: 1, depth: 1))
        enc.endEncoding()
    }

    private func encodeNorm(_ cb: MTLCommandBuffer, _ pso: MTLComputePipelineState, x: MTLBuffer,
                            weight: String, out: MTLBuffer, rows: Int) throws {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(weights, offset: try offset(weight), index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
        var d = UInt32(config.hiddenSize), eps = config.rmsEps
        enc.setBytes(&d, length: 4, index: 3)
        enc.setBytes(&eps, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func encodeHeadNormRope(_ cb: MTLCommandBuffer, x: MTLBuffer, weight: String, out: MTLBuffer,
                                    outOffset: Int, rows: Int, heads: Int, pos0: Int) throws {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(headNormRopePSO)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(weights, offset: try offset(weight), index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        var h = UInt32(heads), p = UInt32(pos0), theta = config.ropeTheta, eps = config.rmsEps
        enc.setBytes(&h, length: 4, index: 3)
        enc.setBytes(&p, length: 4, index: 4)
        enc.setBytes(&theta, length: 4, index: 5)
        enc.setBytes(&eps, length: 4, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Reads every BF16 tensor of a safetensors file into one shared buffer.
    static func loadSafetensors(_ url: URL, device: MTLDevice) throws -> (MTLBuffer, [String: Int]) {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        guard let lengthData = try handle.read(upToCount: 8), lengthData.count == 8 else {
            throw DFlashError.invalidCheckpoint("short safetensors file")
        }
        let headerLength = Int(lengthData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) })
        guard let headerData = try handle.read(upToCount: headerLength), headerData.count == headerLength,
              let header = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw DFlashError.invalidCheckpoint("bad safetensors header")
        }
        var offsets: [String: Int] = [:]
        var dataLength = 0
        for (name, value) in header where name != "__metadata__" {
            guard let entry = value as? [String: Any], entry["dtype"] as? String == "BF16",
                  let range = entry["data_offsets"] as? [Int], range.count == 2 else {
                throw DFlashError.invalidCheckpoint("tensor \(name) is not BF16")
            }
            offsets[name] = range[0]
            dataLength = max(dataLength, range[1])
        }
        guard let buffer = device.makeBuffer(length: dataLength, options: .storageModeShared) else {
            throw DFlashError.invalidState("weight allocation failed")
        }
        buffer.label = "dflash.weights"
        var read = 0
        let base = buffer.contents()
        while read < dataLength {
            guard let chunk = try handle.read(upToCount: min(64 << 20, dataLength - read)), !chunk.isEmpty else {
                throw DFlashError.invalidCheckpoint("truncated safetensors data")
            }
            chunk.withUnsafeBytes { (base + read).copyMemory(from: $0.baseAddress!, byteCount: chunk.count) }
            read += chunk.count
        }
        return (buffer, offsets)
    }
}
