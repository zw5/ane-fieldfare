import Foundation
import Metal

/// GPU micro-benchmarks for decode kernels, with weights rotated through a
/// buffer larger than the system cache so every dispatch reads DRAM.
public enum KernelBench {
    public static func coldGEMV(context: MetalContext, m: Int, n: Int, dispatches: Int = 64) throws -> String {
        let device = context.device
        let wBytes = m * n / 2, sBytes = m * (n / 64) * 2
        let copies = max(2, (768 << 20) / wBytes)
        guard let w = device.makeBuffer(length: wBytes * copies, options: .storageModePrivate),
              let sc = device.makeBuffer(length: sBytes * copies, options: .storageModePrivate),
              let x = device.makeBuffer(length: n * 2, options: .storageModeShared),
              let y = device.makeBuffer(length: m * 2, options: .storageModeShared) else {
            throw DFlashError.invalidState("alloc")
        }
        let gemv = try DequantInt4GEMV(context: context)
        var best = Double.infinity
        for _ in 0..<3 {
            let cb = try context.makeCommandBuffer()
            for i in 0..<dispatches {
                let c = i % copies
                gemv.encode(commandBuffer: cb, weights: w, weightsOffset: c * wBytes, scales: sc, scalesOffset: c * sBytes,
                            biases: sc, biasesOffset: c * sBytes, x: x, y: y, m: UInt32(m), n: UInt32(n))
            }
            cb.commit(); cb.waitUntilCompleted()
            best = min(best, (cb.gpuEndTime - cb.gpuStartTime) / Double(dispatches))
        }
        let bytes = Double(wBytes + 2 * sBytes)
        return unsafe String(format: "cold gemv m=%d n=%d: %.1f us/dispatch, %.0f GB/s", m, n, best * 1e6, bytes / best / 1e9)
    }
}
