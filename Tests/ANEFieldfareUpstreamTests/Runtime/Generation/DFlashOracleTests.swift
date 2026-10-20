import Foundation
import Metal
import Testing

@testable import ANEFieldfareHotpath
import ANEFieldfareValidationSupport

// Oracles behind PERFORMANCE.md's bitwise claims for GPU-routed verify, the
// chained draft, sampled verify and the drafter kernels.

/// Token IDs in the ordinary text range, fixed per seed.
private func syntheticTokens(_ count: Int, seed: String) -> [Int32] {
    var rng = SeedTree(0xD5_1A54).key(seed)
    return (0..<count).map { _ in Int32(rng.uniform(1000, 200_000)) }
}

/// The last verify block's final hidden rows, read back from the prefill scratch.
private func lastVerifyHidden(_ runner: RealForwardRunner, context: MetalContext,
                              rows: Int) throws -> [Float16] {
    let scratch = try #require(runner.prefillScratch)
    let count = rows * scratch.layout.hiddenSize
    let out = try #require(context.device.makeBuffer(length: count * MemoryLayout<Float16>.stride,
                                                     options: .storageModeShared))
    let cb = try context.makeCommandBuffer()
    let blit = try #require(cb.makeBlitCommandEncoder())
    blit.copy(from: scratch.hidden, sourceOffset: 0, to: out, destinationOffset: 0,
              size: count * MemoryLayout<Float16>.stride)
    blit.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return unsafe Array(UnsafeBufferPointer(start: out.contents().bindMemory(to: Float16.self, capacity: count),
                                            count: count))
}

/// A verify block's experts routed on the GPU from the previous layer's guess
/// (with host patches for the routes it did not bind) against every layer
/// planned on the host: the same expert math over the same routes, so the
/// greedy tokens and the final hidden rows must match bitwise.
@Suite(.serialized, .enabled(if: installedCheckpointURL != nil, "installed checkpoint required"))
struct GPURoutedVerifyTests {
    @Test(arguments: [1, 3, 8, 16])
    func gpuRoutedVerifyIsBitwiseTheHostRoutedVerify(rows: Int) async throws {
        let context = try MetalContext()
        let model = try Model.load(directoryURL: try #require(installedCheckpointURL), device: context.device)
        let runner = try RealForwardRunner(model: model, context: context, maxContext: 1024)
        let prompt = syntheticTokens(64, seed: "verify-prompt")
        let block = syntheticTokens(rows, seed: "verify-block-\(rows)")
        let logits = try #require(context.device.makeBuffer(length: model.config.vocabSize * 2,
                                                             options: .storageModeShared))
        _ = try await runner.prefillChunked(tokens: prompt[...], startPosition: 0, outputMode: .greedyIfAvailable,
                                            config: .production(chunkTokens: 64), into: logits,
                                            onProgress: { _ in })
        var results: [(tokens: [Int32], hidden: [Float16])] = []
        for gpuRouted in [true, false] {
            runner.gpuRoutedVerify = gpuRouted
            let boundBefore = runner.boundVerifyRoutes
            let tokens = try await runner.verify(tokens: block, startPosition: prompt.count)
            results.append((tokens, try lastVerifyHidden(runner, context: context, rows: rows)))
            #expect((runner.boundVerifyRoutes > boundBefore) == gpuRouted)
            try runner.rewind(to: prompt.count)
        }
        #expect(results[0].tokens == results[1].tokens)
        let differing = zip(results[0].hidden, results[1].hidden).filter { $0.0.bitPattern != $0.1.bitPattern }.count
        #expect(differing == 0, "\(differing) of \(results[0].hidden.count) hidden values differ")

}

/// The drafter's GPU-side step, queued behind a verify (context update from
/// the target's feature ring, acceptance, and a draft at the position the
/// GPU computed), against the same draft driven from the host.
@Suite(.serialized, .enabled(if: installedCheckpointURL != nil && installedDrafterURL != nil,
                             "installed checkpoint and drafter required"))
struct DFlashChainedDraftTests {
    @Test func chainedDraftIsTheHostDrivenDraft() async throws {
        let context = try MetalContext()
        let model = try Model.load(directoryURL: try #require(installedCheckpointURL), device: context.device)
        let runner = try RealForwardRunner(model: model, context: context, maxContext: 1024)
        let drafter = try await DFlashDrafter(directory: try #require(installedDrafterURL), model: model,
                                        context: context, maxContext: 1024)
        let decoder = try SpeculativeDecoder(runner: runner, drafter: drafter)
        let prompt = syntheticTokens(48, seed: "draft-prompt")
        let logits = try #require(context.device.makeBuffer(length: model.config.vocabSize * 2,
                                                             options: .storageModeShared))
        let prefill = try await decoder.prefillChunked(tokens: prompt[...], startPosition: 0,
                                                       outputMode: .greedyIfAvailable,
                                                       config: .production(chunkTokens: 64), into: logits,
                                                       onProgress: { _ in })
        guard case .greedyToken(let first) = prefill.seed else { Issue.record("no greedy seed"); return }
        let position = prompt.count, rows = 8
        #expect(drafter.contextLength == position)
        let drafts = try drafter.propose(anchor: Int32(bitPattern: first), count: rows)

        let verified = try #require(context.device.makeBuffer(length: 64, options: .storageModeShared))
        // Queued behind the verify as `SpeculativeDecoder` queues it: the
        // context update and acceptance, then the next draft.
        var chained: [any MTLCommandBuffer] = []
        try await runner.verifyBlock(rows: rows, startPosition: position, tokens: drafter.block, greedy: verified) {
            let update = try runner.makeCommandBuffer()
            try drafter.encodeContextUpdate(update, position: position, rows: rows)
            try drafter.encodeAccept(update, verified: verified, rows: rows)
            update.commit()
            let draft = try runner.makeCommandBuffer()
            try drafter.encodeDraft(draft, rows: rows, sampling: nil)
            draft.commit()
            chained = [update, draft]
            return draft
        }
        #expect(chained.count == 2)
        for cb in chained {
            await cb.completed()
            #expect(cb.error == nil)
        }
        let greedy = (0..<rows).map { unsafe Int32(bitPattern: verified.contents().load(fromByteOffset: 4 * $0,
                                                                                            as: UInt32.self)) }
        var accepted = 0
        while accepted < rows - 1, drafts[accepted] == greedy[accepted] { accepted += 1 }
        let (contextLength, gpuAccepted) = drafter.settle()
        #expect(gpuAccepted == accepted)
        #expect(contextLength == position + accepted + 1)
        #expect(drafter.anchor == greedy[accepted])
        let gpu = drafter.drafts(count: rows)

        // The same context and anchor, driven from the host: the verify's
        // feature rows are still in the ring. The update covers all `rows`
        // rows, as the chained one does, so it runs the same kernels.
        try drafter.truncateContext(to: position)
        try drafter.appendContext(rows: rows)
        try drafter.truncateContext(to: position + accepted + 1)
        let host = try drafter.propose(anchor: greedy[accepted], count: rows)
        #expect(gpu.tokens == host)
        #expect(gpu.confidences.map(\.bitPattern) == drafter.drafts(count: rows).confidences.map(\.bitPattern))
    }
}

@Suite(.serialized, .enabled(if: installedCheckpointURL != nil, "installed checkpoint required"))
struct SampledVerifyTests {
    /// Sampled verify rows draw with the noise of the positions they fill:
    /// teacher-forcing a plain sampled decode through 8-row verifies under the
    /// same rule reproduces its tokens (up to near-ties between batched and
    /// single-row arithmetic).
    @Test func sampledVerifyReproducesSampledDecode() async throws {
        let context = try MetalContext()
        let model = try Model.load(directoryURL: try #require(installedCheckpointURL), device: context.device)
        let runner = try RealForwardRunner(model: model, context: context, maxContext: 1024)
        let prompt: [Int32] = [2, 105, 2364, 107, 6974, 496, 17856, 1292, 600, 130450, 48490, 20312, 528, 496,
                               1694, 236764, 607, 496, 6484, 2383, 532, 496, 2822, 2591, 236761, 106, 107, 105,
                               4368, 107, 100, 45518, 107, 101]
        let logits = try #require(context.device.makeBuffer(length: model.config.vocabSize * 2,
                                                             options: .storageModeShared))
        runner.tokenSampling = TokenSampling(temperature: 0.8, topK: 64, topP: 0.95, seed: 5)
        func prefill() async throws -> Int32 {
            runner.reset()
            let result = try await runner.prefillChunked(
                tokens: prompt[...], startPosition: 0, outputMode: .greedyIfAvailable,
                config: .production(chunkTokens: 64), into: logits, onProgress: { _ in })
            guard case .greedyToken(let first) = result.seed else {
                Issue.record("prefill did not seed a greedy token")
                return -1
            }
            return Int32(bitPattern: first)
        }
        var plain = [try await prefill()]
        while plain.count < 65 {
            try await runner.produce(token: plain[plain.count - 1], position: prompt.count + plain.count - 1,
                                     into: logits)
            plain.append(Int32(bitPattern: runner.lastGreedyToken))
        }
        #expect(try await prefill() == plain[0])
        var mismatches: [Int] = []
        var i = 0
        while i + 8 <= 64 {
            let out = try await runner.verify(tokens: Array(plain[i..<(i + 8)]), startPosition: prompt.count + i)
            for r in 0..<8 where out[r] != plain[i + r + 1] { mismatches.append(i + r + 1) }
            i += 8
        }
        #expect(mismatches.isEmpty, "verify differs from plain sampled decode at \(mismatches)")
    }
}

@Suite(.serialized, .enabled(if: installedDrafterURL != nil, "installed drafter required"))
struct DFlashWeightsTests {
    /// The installed INT4 weights hold each source row where the fused
    /// layout puts it, within the INT4 rounding bound of the BF16 source.
    @Test func installedWeightsMatchTheCheckpoint() throws {
        let directory = try #require(installedDrafterURL)
        let context = try MetalContext()
        let config = try DFlashDrafterConfig.load(directory.appendingPathComponent("config.json"))
        let layout = try DFlashWeights.install(directory: directory, config: config)
        let weights = try DFlashWeights.open(directory: directory, layout: layout, config: config,
                                             device: context.device)
        let source = try Data(contentsOf: directory.appendingPathComponent("model.safetensors"),
                              options: .alwaysMapped)
        let headerLength = source.prefix(8).enumerated().reduce(0) { $0 | Int($1.element) << (8 * $1.offset) }
        let header = try #require(try JSONSerialization.jsonObject(with: source[8..<(8 + headerLength)])
            as? [String: [String: Any]])
        func sourceRow(_ name: String, row: Int, columns: Int) throws -> [Float] {
            let offsets = try #require(header[name]?["data_offsets"] as? [Int])
            let start = 8 + headerLength + offsets[0] + row * columns * 2
            return (0..<columns).map { i in
                Quantization.bf16ToFloat(UInt16(source[start + 2 * i]) | UInt16(source[start + 2 * i + 1]) << 8)
            }
        }
        func installedRow(_ m: DFlashMatrix, row: Int) -> (values: [Float], scales: [Float]) {
            let base = unsafe weights.buffer.contents()
            let groups = m.columns / Quantization.groupSize
            var values: [Float] = [], scales: [Float] = []
            for g in 0..<groups {
                let s = unsafe Quantization.bf16ToFloat(base.load(fromByteOffset: m.scales + 2 * (row * groups + g),
                                                                  as: UInt16.self))
                let b = unsafe Quantization.bf16ToFloat(base.load(fromByteOffset: m.biases + 2 * (row * groups + g),
                                                                  as: UInt16.self))
                scales.append(s)
                for i in 0..<Quantization.groupSize {
                    let k = g * Quantization.groupSize + i
                    let byte = unsafe base.load(fromByteOffset: m.weights + (row * m.columns + k) / 2, as: UInt8.self)
                    values.append(Float(k % 2 == 0 ? byte & 0x0F : byte >> 4) * s + b)
                }
            }
            return (values, scales)
        }
        let D = config.hiddenSize, F = config.intermediateSize, q = config.qWidth, kv = config.kvWidth
        let layer = weights.layers[config.layers - 1], l = config.layers - 1
        let a = "layers.\(l).self_attn.", mlp = "layers.\(l).mlp."
        let checks: [(DFlashMatrix, Int, String, Int)] = [
            (weights.fc, 7, "fc.weight", 7),
            (layer.qkv, 5, a + "q_proj.weight", 5), (layer.qkv, q + 3, a + "k_proj.weight", 3),
            (layer.qkv, q + kv + 9, a + "v_proj.weight", 9), (layer.o, 11, a + "o_proj.weight", 11),
            (layer.gateUp, 2, mlp + "gate_proj.weight", 2), (layer.gateUp, F + 4, mlp + "up_proj.weight", 4),
            (layer.down, D - 1, mlp + "down_proj.weight", D - 1),
        ]
        for (matrix, row, name, sourceIndex) in checks {
            let reference = try sourceRow(name, row: sourceIndex, columns: matrix.columns)
            let (values, scales) = installedRow(matrix, row: row)
            for (k, (x, y)) in zip(reference, values).enumerated() {
                let s = scales[k / Quantization.groupSize]
                // Half a step, plus the BF16 rounding of scale and bias.
                #expect(abs(x - y) <= 0.5 * s + 0.01 * (abs(x) + 15 * s), "\(name) row \(sourceIndex) col \(k)")
            }
        }
    }
}
}

/// `dflash_residual_add_norm` is the residual add followed by the row norm,
/// bitwise.
@Suite struct DFlashKernelTests {
    @Test func residualAddNormIsBitwiseTheAddThenTheNorm() throws {
        let context = try MetalContext()
        let rows = 5, d = 2816
        var rng = SeedTree(0xADD).key("add-norm")
        let residual: [Float] = (0..<(rows * d)).map { _ in Float(rng.uniform(-30, 30)) }
        let delta: [UInt16] = (0..<(rows * d)).map { _ in Quantization.bf16Bits(Float(rng.uniform(-4, 4))) }
        let weight: [UInt16] = (0..<d).map { _ in Quantization.bf16Bits(Float(rng.uniform(0.5, 1.5))) }
        func buffer<T: BitwiseCopyable>(_ values: [T]) throws -> any MTLBuffer {
            let made = values.withUnsafeBytes {
                unsafe context.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
            }
            return try #require(made)
        }
        let fused = try buffer(residual), separate = try buffer(residual.indices.map {
            residual[$0] + Quantization.bf16ToFloat(delta[$0])
        })
        let deltaBuffer = try buffer(delta), weightBuffer = try buffer(weight)
        let fusedOut = try buffer([UInt16](repeating: 0, count: rows * d))
        let separateOut = try buffer([UInt16](repeating: 0, count: rows * d))
        let cb = try context.makeCommandBuffer()
        for (name, buffers) in [("dflash_residual_add_norm", [fused, deltaBuffer, weightBuffer, fusedOut]),
                                ("dflash_rmsnorm_rows_f32", [separate, weightBuffer, separateOut])] {
            let enc = try #require(cb.makeComputeCommandEncoder())
            enc.setComputePipelineState(try context.pipeline(name))
            for (i, b) in buffers.enumerated() { enc.setBuffer(b, offset: 0, index: i) }
            enc.setValue(UInt32(d), index: buffers.count)
            enc.setValue(Float(1e-6), index: buffers.count + 1)
            enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }
        cb.commit()
        cb.waitUntilCompleted()
        func words(_ b: any MTLBuffer, _ count: Int) -> [UInt32] {
            unsafe Array(UnsafeBufferPointer(start: b.contents().bindMemory(to: UInt32.self, capacity: count),
                                             count: count))
        }
        #expect(words(fused, rows * d) == words(separate, rows * d))
        #expect(words(fusedOut, rows * d / 2) == words(separateOut, rows * d / 2))
    }
}

