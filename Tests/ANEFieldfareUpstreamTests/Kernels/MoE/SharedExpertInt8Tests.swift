import Testing
import Foundation
import Metal
@testable import ANEFieldfareHotpath
import ANEFieldfareValidationSupport

/// Validates the standalone 8-bit dense MLP wrapper used by Gemma 4 as the
/// "shared expert" parallel branch of its MoE, and with it the INT8 GEMV it
/// runs three times:
///
///     y = down(gelu_pytorch_tanh(gate(x)) * up(x))
///
/// Reference composes DequantInt8GemvRef + gelu_pytorch_tanh + multiply,
/// matching the math but with a different staging (bulk dequant rows, then
/// dotpr) so a bug in either side won't replicate.
@Suite struct SharedExpertInt8Tests {
    private static let d = 128
    private static let f = 64

    @Test func sharedExpertInt8_matchesReference() throws {
        var rng = SeedTree(0x601).key("shared-expert-int8")
        let x = (0..<Self.d).map { _ in rng.uniform(-0.4, 0.4) }
        let gate = (0..<Self.f).map { _ in (0..<Self.d).map { _ in rng.uniform(-0.4, 0.4) } }
        let up = (0..<Self.f).map { _ in (0..<Self.d).map { _ in rng.uniform(-0.4, 0.4) } }
        let down = (0..<Self.d).map { _ in (0..<Self.f).map { _ in rng.uniform(-0.4, 0.4) } }
        let gateRows = gate.map(Quantization.quantizeInt8Affine)
        let upRows = up.map(Quantization.quantizeInt8Affine)
        let downRows = down.map(Quantization.quantizeInt8Affine)

        let x16 = x.map { Float(Float16($0)) }
        let gateOut = DequantInt8GemvRef.apply(weightRows: gateRows, x: x16, n: Self.d)
        let upOut = DequantInt8GemvRef.apply(weightRows: upRows, x: x16, n: Self.d)
        // gelu_pytorch_tanh + multiply, rounded through FP16 once as the kernel stores it.
        let act = zip(gateOut, upOut).map { gateValue, upValue in
            let cube = gateValue * gateValue * gateValue
            let inner = 0.7978845608028654 * Double(gateValue + 0.044715 * cube)
            return Float(Float16(Float(0.5 * Double(gateValue) * (1 + tanh(inner))) * upValue))
        }
        let reference = DequantInt8GemvRef.apply(weightRows: downRows, x: act, n: Self.f)

        let ctx = try MetalContext()
        let wrapper = try SharedExpertInt8(context: ctx)
        func projection(_ rows: [Quantization.Int8AffineRow], cols: Int) throws -> SharedExpertProjection {
            func buffer<T: BitwiseCopyable>(_ values: [T]) throws -> any MTLBuffer {
                try #require(unsafe ctx.device.makeBuffer(bytes: values,
                                                          length: values.count * MemoryLayout<T>.stride,
                                                          options: .storageModeShared))
            }
            let weights = try buffer(rows.flatMap(\.packed))
            let scales = try buffer(rows.flatMap(\.scales))
            let biases = try buffer(rows.flatMap(\.biases))
            return SharedExpertProjection(weights: weights, scales: scales, biases: biases,
                                        rows: UInt32(rows.count), cols: UInt32(cols))
        }
        let xBuf = try #require(Fp16Buffer.make(ctx.device, values: x))
        let yBuf = try #require(Fp16Buffer.make(ctx.device, count: Self.d))
        let actScratch = try #require(Fp16Buffer.make(ctx.device, count: Self.f))
        let gateProj = try projection(gateRows, cols: Self.d)
        let upProj = try projection(upRows, cols: Self.d)
        let downProj = try projection(downRows, cols: Self.f)

        let cb = try #require(ctx.queue.makeCommandBuffer())
        try wrapper.encode(commandBuffer: cb,
                           x: xBuf, gate: gateProj, up: upProj, down: downProj,
                           y: yBuf,
                           scratchAct: actScratch)
        cb.commit(); cb.waitUntilCompleted()

        let actual = Fp16Buffer.read(yBuf, count: Self.d)
        let rel = RelError.compute(actual: actual, reference: reference)
        #expect(rel < Tolerance.quantInt8 * 4, "shared-expert int8 rel=\(rel)")
    }
}
