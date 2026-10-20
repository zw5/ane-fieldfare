import Foundation
import Metal
import Testing
@testable import ANEFieldfareHotpath
import ANEFieldfareValidationSupport

private let mppTensorOpsAvailable: Bool = {
    guard let context = try? MetalContext() else { return false }
    return MPPPrefillInt4QMM(context: context).isAvailable
}()

@Suite struct MPPPrefillInt4QMMTests {
    private struct Inputs {
        let packed: [UInt8]
        let scales: [UInt16]  // FP16 bits, one per 32 weights
        let x: [Float16]
    }

    private static func makeInputs(m: Int,
                                   n: Int,
                                   k: Int,
                                   adversarialScales adversarial: Bool = false) -> Inputs {
        let groups = k / Quantization.q4GroupSize
        var packed = [UInt8](repeating: 0, count: n * k / 2)
        for index in packed.indices {
            packed[index] = UInt8(truncatingIfNeeded: index &* 37 &+ 0x29)
        }
        // Negative scales and FP16 subnormals.
        let adversarialScales: [UInt16] = [
            Float16(0.001).bitPattern,
            Float16(-0.0015).bitPattern,
            0x0001,
            0x8001,
        ]
        var scales = [UInt16](repeating: 0, count: n * groups)
        for row in 0..<n {
            for group in 0..<groups {
                scales[row * groups + group] = adversarial
                    ? adversarialScales[(row + group) % adversarialScales.count]
                    : Float16(0.001 + Float((row + group) % 5) * 0.00025).bitPattern
            }
        }
        var x = [Float16](repeating: 0, count: m * k)
        for index in x.indices {
            x[index] = Float16(Float((index * 11) % 29 - 14) / 64.0)
        }
        return Inputs(packed: packed, scales: scales, x: x)
    }

    private static func makeBuffer<T>(device: any MTLDevice,
                                      values: [T],
                                      prefixBytes: Int = 0) -> any MTLBuffer? {
        let byteCount = values.count * MemoryLayout<T>.stride
        guard let buffer = device.makeBuffer(length: prefixBytes + byteCount,
                                             options: .storageModeShared) else {
            return nil
        }
        values.withUnsafeBytes { bytes in
            guard let baseAddress = bytes.baseAddress else { return }
            unsafe buffer.contents().advanced(by: prefixBytes)
                .copyMemory(from: baseAddress, byteCount: byteCount)
        }
        return buffer
    }

    private static func cpuReference(_ inputs: Inputs,
                                     m: Int,
                                     n: Int,
                                     k: Int) -> [Float] {
        let group = Quantization.q4GroupSize
        let groups = k / group
        let rowBytes = k / 2
        var output = [Float](repeating: 0, count: m * n)
        for token in 0..<m {
            for row in 0..<n {
                var accumulator: Float = 0
                for g in 0..<groups {
                    let scale = Float(Float16(bitPattern: inputs.scales[row * groups + g]))
                    for localK in 0..<group {
                        let column = g * group + localK
                        let byte = inputs.packed[row * rowBytes + column / 2]
                        let q = column.isMultiple(of: 2) ? byte & 0x0f : byte >> 4
                        let weight = Float(Int(q) - 8) * scale
                        accumulator.addProduct(weight, Float(inputs.x[token * k + column]))
                    }
                }
                output[token * n + row] = Float(Float16(accumulator))
            }
        }
        return output
    }

    /// One oracle for both prefill INT4 GEMMs: the `PrefillInt4QMM` fallback
    /// always, and the TensorOps candidate where the runtime supports it.
    /// Partial tiles in M and N, adversarial affine terms, unaligned weights.
    @Test func int4QMMAndTensorOpsCandidateMatchFP32Q4_0Reference() throws {
        let m = 17, n = 35, k = 128
        let weightOffset = 13, scaleOffset = 2
        let context = try MetalContext()
        let baseline = try PrefillInt4QMM(context: context)
        let candidate = MPPPrefillInt4QMM(context: context)
        let inputs = Self.makeInputs(m: m, n: n, k: k, adversarialScales: true)
        guard let weights = Self.makeBuffer(device: context.device,
                                            values: inputs.packed,
                                            prefixBytes: weightOffset),
              let scaleBuffer = Self.makeBuffer(device: context.device,
                                                values: inputs.scales,
                                                prefixBytes: scaleOffset),
              let input = Fp16Buffer.make(context.device, halves: inputs.x),
              let baselineBuffer = Fp16Buffer.make(context.device, count: m * n),
              let candidateBuffer = Fp16Buffer.make(context.device, count: m * n),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed")
            return
        }

        baseline.encode(commandBuffer: commandBuffer,
                        weights: weights, weightsOffset: weightOffset,
                        scales: scaleBuffer, scalesOffset: scaleOffset,
                        x: input, y: baselineBuffer, t: m, n: n, k: k)
        let path = mppTensorOpsAvailable
            ? candidate.encode(commandBuffer: commandBuffer,
                               weights: weights, weightsOffset: weightOffset,
                               scales: scaleBuffer, scalesOffset: scaleOffset,
                               x: input, y: candidateBuffer, m: m, n: n, k: k)
            : nil
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        let reference = Self.cpuReference(inputs, m: m, n: n, k: k)
        var outputs = [("PrefillInt4QMM", baselineBuffer)]
        if let path {
            #expect(path == .affineThreadgroupF16)
            outputs.append(("MPP candidate", candidateBuffer))
        }
        for (label, buffer) in outputs {
            let actual = Fp16Buffer.read(buffer, count: m * n)
            let maxAbs = RelError.maxAbsDiff(actual, reference)
            let relative = RelError.compute(actual: actual, reference: reference)
            #expect(maxAbs <= 0.03, "\(label) maxAbs=\(maxAbs) rel=\(relative)")
            #expect(relative <= 1e-3, "\(label) maxAbs=\(maxAbs) rel=\(relative)")
        }
    }

    @Test func apple10BF16ProjectorPreservesBF16Q4_0Semantics() throws {
        let context = try MetalContext()
        guard context.device.supportsFamily(.apple10) else { return }
        let m = 64, n = 64, k = 128
        let inputs = Self.makeInputs(m: m, n: n, k: k, adversarialScales: true)
        let activations = inputs.x.map { Quantization.bf16Bits(Float($0)) }
        guard let weights = Self.makeBuffer(device: context.device, values: inputs.packed),
              let scales = Self.makeBuffer(device: context.device, values: inputs.scales),
              let x = Self.makeBuffer(device: context.device, values: activations),
              let y = Fp16Buffer.make(context.device, count: m * n),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed")
            return
        }
        let candidate = MPPPrefillInt4QMM(context: context, variant: .apple10BF16)
        #expect(candidate.isAvailable, "\(candidate.unavailableReason ?? "unknown error")")
        // A short output must still project. The BF16 variant is the vision
        // projector and has no control kernel, so falling back here meant not
        // projecting at all — a 3x3 patch grid pools to a single row and failed
        // outright. The Apple10 kernel bound-checks against M, so a partial tile
        // is correct.
        let shortOutput = candidate.encode(
            commandBuffer: commandBuffer,
            weights: weights, scales: scales,
            x: x, y: y, m: 17, n: 35, k: k)
        #expect(shortOutput == .affineThreadgroupBF16Apple10)
        let singleRow = candidate.encode(
            commandBuffer: commandBuffer,
            weights: weights, scales: scales,
            x: x, y: y, m: 1, n: 35, k: k)
        #expect(singleRow == .affineThreadgroupBF16Apple10)
        let path = candidate.encode(
            commandBuffer: commandBuffer,
            weights: weights, scales: scales,
            x: x, y: y, m: m, n: n, k: k)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)
        #expect(path == .affineThreadgroupBF16Apple10)

        var reference = [Float](repeating: 0, count: m * n)
        let groups = k / Quantization.q4GroupSize
        for token in 0..<m {
            for row in 0..<n {
                var accumulator: Float = 0
                for column in 0..<k {
                    let byte = inputs.packed[row * k / 2 + column / 2]
                    let q = column.isMultiple(of: 2) ? byte & 0x0f : byte >> 4
                    let group = row * groups + column / Quantization.q4GroupSize
                    let scale = Float(Float16(bitPattern: inputs.scales[group]))
                    let weight = Quantization.bf16ToFloat(Quantization.bf16Bits(
                        Float(q) * scale + -8 * scale))
                    accumulator.addProduct(
                        weight, Quantization.bf16ToFloat(activations[token * k + column]))
                }
                reference[token * n + row] = Float(Float16(accumulator))
            }
        }
        let actual = Fp16Buffer.read(y, count: m * n)
        #expect(RelError.maxAbsDiff(actual, reference) <= 0.03)
        #expect(RelError.compute(actual: actual, reference: reference) <= 1e-3)
    }

    @Test func unsupportedOrUnalignedInputsReportFallback() throws {
        let context = try MetalContext()
        let candidate = MPPPrefillInt4QMM(context: context)
        guard let buffer = context.device.makeBuffer(length: 4096,
                                                     options: .storageModeShared),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed")
            return
        }
        let unsupportedShape = candidate.encode(commandBuffer: commandBuffer,
                                                weights: buffer,
                                                scales: buffer,
                                                x: buffer,
                                                y: buffer,
                                                m: 1,
                                                n: 1,
                                                k: 65)
        let unalignedScale = candidate.encode(commandBuffer: commandBuffer,
                                              weights: buffer,
                                              weightsOffset: 1,
                                              scales: buffer,
                                              scalesOffset: 1,
                                              x: buffer,
                                              y: buffer,
                                              m: 1,
                                              n: 1,
                                              k: 64)
        #expect(unsupportedShape == .fallback)
        #expect(unalignedScale == .fallback)
    }

}
