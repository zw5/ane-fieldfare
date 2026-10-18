import Foundation
import Metal
import Testing
@testable import ANEFieldfareHotpath

private let visionMPPTensorOpsAvailable = MTLCreateSystemDefaultDevice()?
    .supportsFamily(.apple8) == true

@Suite struct VisionLinearBF16Tests {
    @Test(.enabled(if: visionMPPTensorOpsAvailable,
                   "vision linear kernels require Apple8 or newer"))
    func registerPathProducesExpectedConstantProduct() throws {
        let context = try MetalContext()
        let m = 128, n = 256, k = 1_152
        func buffer(count: Int, value: Float) throws -> MTLBuffer {
            let bits = [UInt16](repeating: Quantization.bf16Bits(value), count: count)
            return try #require(context.device.makeBuffer(
                bytes: bits, length: bits.count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared))
        }
        let input = try buffer(count: m * k, value: 0.125)
        let weights = try buffer(count: n * k, value: 0.25)

        // Poisoned rather than zeroed, so a kernel that dispatched nothing fails.
        let output = try buffer(count: m * n, value: -1)
        let linear = try VisionLinearBF16(context: context)
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        linear.encode(commandBuffer: commandBuffer,
                      input: input, weights: weights, output: output,
                      m: m, n: n, k: k)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)
        let values = output.contents().bindMemory(to: UInt16.self, capacity: m * n)
        var maximumError: Float = 0
        for index in 0..<(m * n) {
            maximumError = max(
                maximumError,
                abs(Quantization.bf16ToFloat(values[index]) - 36))
        }
        #expect(maximumError == 0)
    }

    @Test(.enabled(if: visionMPPTensorOpsAvailable,
                   "vision linear kernels require Apple8 or newer"))
    func registerPathHandlesNonconstantTailShape() throws {
        let context = try MetalContext()
        let m = 79, n = 272, k = 23
        let inputBits = (0..<(m * k)).map { index in
            Quantization.bf16Bits(Float((index * 17) % 31 - 15) / 32)
        }
        let weightBits = (0..<(n * k)).map { index in
            Quantization.bf16Bits(Float((index * 13) % 29 - 14) / 64)
        }
        let inputValues = inputBits.map(Quantization.bf16ToFloat)
        let weightValues = weightBits.map(Quantization.bf16ToFloat)
        let input = try #require(context.device.makeBuffer(
            bytes: inputBits,
            length: inputBits.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared))
        let weights = try #require(context.device.makeBuffer(
            bytes: weightBits,
            length: weightBits.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared))
        let output = try #require(context.device.makeBuffer(
            length: m * n * MemoryLayout<UInt16>.stride,
            options: .storageModeShared))
        let linear = try VisionLinearBF16(context: context)
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        linear.encode(commandBuffer: commandBuffer,
                      input: input, weights: weights, output: output,
                      m: m, n: n, k: k)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        let actual = output.contents().bindMemory(to: UInt16.self, capacity: m * n)
        var maximumWideError: Float = 0
        var maximumTailError: Float = 0
        for row in 0..<m {
            for column in 0..<n {
                var expected: Float = 0
                for inner in 0..<k {
                    expected += inputValues[row * k + inner] * weightValues[column * k + inner]
                }
                let error = abs(
                    Quantization.bf16ToFloat(actual[row * n + column]) - expected)
                if column < 256 {
                    maximumWideError = max(maximumWideError, error)
                } else {
                    maximumTailError = max(maximumTailError, error)
                }
            }
        }
        #expect(maximumWideError < 0.016)
        #expect(maximumTailError < 0.016)
    }
}
