import Testing
import Foundation
import Metal
@testable import ANEFieldfareHotpath
import ANEFieldfareValidationSupport

/// `Sampler` exercises: greedy=argmax, seeded per-position reproducibility,
/// top-k / top-p truncation, and the repetition penalty on saturated logits. Inputs are raw pre-softcap logits
/// (the sampler runs the softcap+softmax front-end itself).
@Suite struct SamplerTests {

    /// Reusable rig — one MetalContext + Sampler + buffers, shared across the
    /// many draws a single test makes (avoids recompiling the shader library
    /// per draw).
    private final class Rig {
        let ctx: MetalContext
        let sampler: Sampler
        let vocab: Int
        let logits: any MTLBuffer
        let probs: any MTLBuffer
        let outToken: any MTLBuffer

        init(vocab: Int) throws {
            self.ctx = try MetalContext()
            self.sampler = try Sampler(context: ctx, vocab: vocab)
            self.vocab = vocab
            guard let l = ctx.device.makeBuffer(length: vocab * MemoryLayout<Float16>.size,
                                                options: .storageModeShared),
                  let p = ctx.device.makeBuffer(length: vocab * MemoryLayout<Float16>.size,
                                                options: .storageModeShared),
                  let o = ctx.device.makeBuffer(length: MemoryLayout<UInt32>.size,
                                                options: .storageModeShared) else {
                throw MetalError.noDevice
            }
            self.logits = l; self.probs = p; self.outToken = o
        }

        func writeLogits(_ values: [Float]) {
            let ptr = unsafe logits.contents().bindMemory(to: Float16.self, capacity: vocab)
            for i in 0..<vocab { unsafe ptr[i] = Float16(values[i]) }
        }

        func draw(_ values: [Float], config: GenerationConfig,
                  position: Int = 0, history: [Int32] = []) -> UInt32 {
            writeLogits(values)
            let cmd = ctx.queue.makeCommandBuffer()!
            sampler.sample(commandBuffer: cmd, logits: logits, probs: probs,
                           history: history, config: config,
                           position: position, outToken: outToken)
            cmd.commit(); cmd.waitUntilCompleted()
            return unsafe outToken.contents().load(as: UInt32.self)
        }
    }

    @Test func greedy_picksArgmax() throws {
        let v = 2048
        let rig = try Rig(vocab: v)
        var logits = [Float](repeating: 0.1, count: v)
        logits[1337] = 9.0
        let id = rig.draw(logits, config: GenerationConfig(temperature: 0))
        #expect(id == 1337, "got \(id)")
    }

    @Test func seeded_reproducibleAcrossPositions() throws {
        let v = 1024
        let rig = try Rig(vocab: v)
        var rng = SeedTree(0x51A7_1006).key("sampler-position-replay")
        let logits = (0..<v).map { _ in rng.uniform(-2, 2) }
        let cfg = GenerationConfig(temperature: 1.0, seed: 42)
        let run1 = (0..<5).map { rig.draw(logits, config: cfg, position: $0) }
        let run2 = (0..<5).map { rig.draw(logits, config: cfg, position: $0) }
        #expect(run1 == run2, "seed=42 not reproducible: \(run1) vs \(run2)")

        // A different seed should diverge on at least one position.
        let cfg43 = GenerationConfig(temperature: 1.0, seed: 43)
        let run3 = (0..<5).map { rig.draw(logits, config: cfg43, position: $0) }
        #expect(run3 != run1, "seed 42 and 43 produced identical sequences")
    }

    /// Top-k keeps the four largest; top-p keeps two tokens holding ~99% of
    /// the mass after softmax. No draw may leave the kept set.
    @Test func truncationKeepsEveryDrawInsideTheKeptSet() throws {
        let v = 1024
        let rig = try Rig(vocab: v)
        var logits = [Float](repeating: -10.0, count: v)
        logits[10] = 6.0; logits[200] = 5.6; logits[500] = 2.0; logits[900] = 1.0
        for t in 0..<32 {
            let seed = UInt64(t) &+ 1
            let k = rig.draw(logits, config: GenerationConfig(temperature: 1.0, topK: 4, seed: seed), position: t)
            #expect([10, 200, 500, 900].contains(k), "trial \(t): id=\(k) outside top-4")
            let p = rig.draw(logits, config: GenerationConfig(temperature: 1.0, topP: 0.9, seed: seed), position: t)
            #expect([10, 200].contains(p), "trial \(t): id=\(p) outside nucleus")
        }
    }

    /// Saturated-logit suppression: real Gemma 4 raw logits reach the
    /// hundreds, deep in softcap-tanh saturation. The penalty must act on the
    /// post-softcap value — applied to the raw logit it moves the capped
    /// result by ~nothing and the penalty silently no-ops (the repetition-loop
    /// regression this pins).
    @Test func repetitionPenalty_bitesOnSaturatedLogits() throws {
        let v = 64
        let rig = try Rig(vocab: v)
        // All raw logits deep in tanh saturation; id 5 is the model's strong
        // favorite and also the repeated-history token.
        var logits = [Float](repeating: 300.0, count: v)
        logits[5] = 400.0
        let history: [Int32] = [5]
        var count5 = 0
        let trials = 64
        for t in 0..<trials {
            let cfg = GenerationConfig(temperature: 1.0, repetitionPenalty: 1.3, seed: UInt64(t) &+ 1)
            let id = rig.draw(logits, config: cfg, position: t, history: history)
            if id == 5 { count5 += 1 }
        }
        // Post-softcap both land near the 30 cap; penalty 1.3 drops id 5 to
        // ~23, ~e^-7 of the others — it should essentially never win. The raw
        // pre-fix math left id 5 the argmax favorite at >half the draws.
        #expect(count5 < trials / 8, "saturated id 5 drawn \(count5)/\(trials) despite penalty")
    }
}
