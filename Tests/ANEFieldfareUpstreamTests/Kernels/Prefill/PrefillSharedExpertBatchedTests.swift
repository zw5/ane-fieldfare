import Foundation
import Metal
import Testing

@testable import ANEFieldfareHotpath
import ANEFieldfareValidationSupport

/// The batched shared-FFN GEMM must be at least as close to an FP32 CPU
/// reference as the per-token GEMV it replaces, on real checkpoint weights.
/// Both kernels round intermediates to FP16; neither is the reference.
@Suite(.serialized, .enabled(if: installedCheckpointURL != nil, "installed checkpoint required"))
struct PrefillSharedExpertBatchedTests {
    static let tokens = 64

    struct Errors { let relativeL2: Double; let maxAbs: Double }

    static func errors(_ candidate: [Float], _ reference: [Float]) -> Errors {
        var num = 0.0, den = 0.0, maxAbs = 0.0
        for (c, r) in zip(candidate, reference) {
            let e = Double(c) - Double(r)
            num += e * e; den += Double(r) * Double(r); maxAbs = max(maxAbs, abs(e))
        }
        return Errors(relativeL2: (num / den).squareRoot(), maxAbs: maxAbs)
    }

    static func rows(_ view: TensorView, rows: Int, cols: Int) -> [Quantization.Int4AffineRow] {
        let base = view.buffer.contents()
        let groups = cols / Quantization.groupSize
        return (0..<rows).map { row in
            let packed = base.advanced(by: Int(view.offset) + row * cols / 2)
                .bindMemory(to: UInt8.self, capacity: cols / 2)
            let scales = base.advanced(by: Int(view.scaleOffset) + row * groups * 2)
                .bindMemory(to: UInt16.self, capacity: groups)
            let biases = base.advanced(by: Int(view.biasOffset) + row * groups * 2)
                .bindMemory(to: UInt16.self, capacity: groups)
            return Quantization.Int4AffineRow(
                packed: Array(UnsafeBufferPointer(start: packed, count: cols / 2)),
                scales: Array(UnsafeBufferPointer(start: scales, count: groups)),
                biases: Array(UnsafeBufferPointer(start: biases, count: groups)))
        }
    }

    @Test(arguments: [0, 14, 29])
    func batchedIsNoLessAccurateThanPerToken(layer: Int) throws {
        let context = try MetalContext()
        let model = try Model.load(directoryURL: try #require(installedCheckpointURL), device: context.device,
                                   integrityPolicy: .sizeCheckTrustedReceipt)
        let d = model.config.hiddenSize, f = model.config.intermediateSize, t = Self.tokens
        let views = (gate: try model.sharedExpertGate(layer: layer), up: try model.sharedExpertUp(layer: layer),
                     down: try model.sharedExpertDown(layer: layer))
        func projection(_ view: TensorView, rows: Int, cols: Int) -> SharedExpertProjection {
            SharedExpertProjection(weights: view.buffer, scales: view.buffer, biases: view.buffer,
                                   weightsOffset: Int(view.offset), scalesOffset: Int(view.scaleOffset),
                                   biasesOffset: Int(view.biasOffset), rows: UInt32(rows), cols: UInt32(cols))
        }
        var rng = SeedTree(0x5EED).key("shared-ffn-layer-\(layer)")
        let x: [Float16] = (0..<(t * d)).map { _ in Float16(rng.uniform(-2, 2)) }

        func run(batched: Bool) throws -> [Float] {
            let kernel = try PrefillSharedExpert(context: context, weightBits: model.sharedExpertWeightBits,
                                                 batched: batched)
            #expect(kernel.usesBatchedGEMM == batched)
            let xBuffer = try #require(context.device.makeBuffer(bytes: x, length: x.count * 2, options: .storageModeShared))
            let y = try #require(context.device.makeBuffer(length: t * d * 2, options: .storageModeShared))
            let scratch = try (0..<3).map { _ in try #require(context.device.makeBuffer(length: f * 2, options: .storageModePrivate)) }
            let cb = try #require(context.queue.makeCommandBuffer())
            try kernel.encodeBlock(commandBuffer: cb, x: xBuffer, y: y,
                                   gate: projection(views.gate, rows: f, cols: d),
                                   up: projection(views.up, rows: f, cols: d),
                                   down: projection(views.down, rows: d, cols: f),
                                   scratchGate: scratch[0], scratchUp: scratch[1], scratchAct: scratch[2],
                                   queryCount: t, d: d, intermediate: f, xStrideElements: d, yStrideElements: d)
            cb.commit(); cb.waitUntilCompleted()
            #expect(cb.error == nil)
            let out = y.contents().bindMemory(to: Float16.self, capacity: t * d)
            return (0..<(t * d)).map { Float(out[$0]) }
        }
        let perToken = try run(batched: false)
        let batched = try run(batched: true)

        let gateRows = Self.rows(views.gate, rows: f, cols: d)
        let upRows = Self.rows(views.up, rows: f, cols: d)
        let downRows = Self.rows(views.down, rows: d, cols: f)
        var reference: [Float] = []
        for token in 0..<t {
            let row = x[(token * d)..<((token + 1) * d)].map(Float.init)
            reference += MoeRef.runFFN(gateRows: gateRows, upRows: upRows, downRows: downRows, x: row, d: d, f: f)
        }
        let a = Self.errors(perToken, reference), b = Self.errors(batched, reference)
        print("shared FFN layer \(layer): per-token relL2=\(a.relativeL2) maxAbs=\(a.maxAbs); batched relL2=\(b.relativeL2) maxAbs=\(b.maxAbs)")
        #expect(b.relativeL2 <= a.relativeL2 * 1.25 + 1e-6)
        #expect(b.relativeL2 < 5e-3)
    }
}
