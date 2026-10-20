import Testing
import Foundation
import Darwin
import Metal
@testable import ANEFieldfareHotpath
import ANEFieldfareValidationSupport

/// End-to-end: write one synthetic Q4_0 expert blob (packed nibbles + FP16
/// scales), open it through `MappedExpertLayer`,
/// read it into the page cache, run `dequant_int4_gemv`, and compare against
/// an FP32 CPU reference. Covers the streaming -> kernel data path.
@Suite struct StreamingKernelIntegrationTests {

    /// Layout for a single "expert blob" used in this test.
    ///   M = 64  output rows
    ///   N = 128 input columns          (4 groups of 32 per row)
    /// Packed:   M * N/2                 = 4096 bytes
    /// Scales:   M * N/32 * 2 B          =  512 bytes FP16
    /// We page-align both regions so MTLBuffer offsets are legal.
    private struct Sizes {
        static let M = 64
        static let N = 128
        static let groupsPerRow = N / Quantization.q4GroupSize  // 4
        static let packedBytes  = M * (N / 2)                   // 4096
        static let scalesBytes  = M * groupsPerRow * MemoryLayout<UInt16>.size  // 512
    }

    /// Build the in-memory bytes that we'll write to the fake layer file.
    /// Layout inside one expert blob (page-aligned externally):
    ///   [0,        packedBytes)              packed nibbles, row-major
    ///   [scalesOff, scalesOff + scalesBytes) FP16 scales, row-major
    private static func buildExpertBlob(weightsFp32: [[Float]],
                                        pageSize: Int)
        -> (blob: [UInt8], scalesOffset: Int, blobSize: Int)
    {
        precondition(weightsFp32.count == Sizes.M)
        precondition(weightsFp32[0].count == Sizes.N)

        var packed: [UInt8] = []
        var scales: [UInt16] = []
        for row in weightsFp32 {
            let q = Quantization.quantizeQ4_0(row)
            packed += q.packed
            scales += q.scales
        }

        let scalesOffset = roundUp(Sizes.packedBytes, to: pageSize)
        let blobSize     = roundUp(scalesOffset + Sizes.scalesBytes, to: pageSize)

        func littleEndian(_ values: [UInt16]) -> [UInt8] {
            values.flatMap { [UInt8(truncatingIfNeeded: $0), UInt8(truncatingIfNeeded: $0 >> 8)] }
        }
        var blob = [UInt8](repeating: 0, count: blobSize)
        blob.replaceSubrange(0..<Sizes.packedBytes, with: packed)
        blob.replaceSubrange(scalesOffset..<(scalesOffset + Sizes.scalesBytes),
                             with: littleEndian(scales))
        return (blob, scalesOffset, blobSize)
    }

    private static func roundUp(_ n: Int, to pageSize: Int) -> Int {
        (n + pageSize - 1) / pageSize * pageSize
    }

    private static func cpuReferenceGemv(weightsFp32: [[Float]], x: [Float]) -> [Float] {
        var y = [Float](repeating: 0, count: Sizes.M)
        for m in 0..<Sizes.M {
            let q = Quantization.quantizeQ4_0(weightsFp32[m])
            let w = Quantization.dequantizeQ4_0(q, n: Sizes.N)
            var acc: Float = 0
            for n in 0..<Sizes.N { acc += w[n] * x[n] }
            y[m] = acc
        }
        return y
    }

    @Test func endToEnd_streamFromFile_runKernel_matchesCpuReference() throws {
        var rng = SeedTree(0x51A7_1002).key("streaming-kernel-integration")
        var weights = [[Float]](repeating: [], count: Sizes.M)
        for m in 0..<Sizes.M {
            weights[m] = (0..<Sizes.N).map { _ in rng.uniform(-1.0, 1.0) }
        }
        let xFp32: [Float] = (0..<Sizes.N).map { _ in rng.uniform(-1.0, 1.0) }
        let xFp16: [Float16] = xFp32.map { Float16($0) }

        let pageSize = Int(getpagesize())
        let (blob, scalesOffset, blobSize) =
            Self.buildExpertBlob(weightsFp32: weights, pageSize: pageSize)

        let headerSize = pageSize
        let fileSize   = headerSize + blobSize
        var fileBytes  = [UInt8](repeating: 0, count: fileSize)
        fileBytes.replaceSubrange(headerSize..<(headerSize + blob.count), with: blob)

        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("turbo-fieldfare-smoke-\(UUID().uuidString).bin")
        try Data(fileBytes).write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let ctx = try MetalContext()
        let fd = unsafe open(tmp.path, O_RDONLY)
        #expect(fd >= 0)
        defer { close(fd) }
        let layer = try MappedExpertLayer(
            fileDescriptor: fd,
            path: tmp.path,
            device: ctx.device,
            expertCount: 1,
            expertBytes: blobSize,
            expertOffsets: [UInt64(headerSize)])
        try layer.read([0])
        let kernel = try DequantInt4GEMV(context: ctx)
        let weightsBuf = layer.buffers[0]

        guard let xBuf = ctx.device.makeBuffer(length: Sizes.N * MemoryLayout<Float16>.size,
                                               options: .storageModeShared),
              let yBuf = ctx.device.makeBuffer(length: Sizes.M * MemoryLayout<Float16>.size,
                                               options: .storageModeShared) else {
            Issue.record("Failed to allocate x / y buffers")
            return
        }
        unsafe xBuf.contents().withMemoryRebound(to: Float16.self, capacity: Sizes.N) { dst in
            for i in 0..<Sizes.N { unsafe dst[i] = xFp16[i] }
        }

        guard let cmd = ctx.queue.makeCommandBuffer() else {
            Issue.record("Failed to make command buffer"); return
        }
        kernel.encode(commandBuffer: cmd,
                      weights: weightsBuf, weightsOffset: 0,
                      scales: weightsBuf, scalesOffset: scalesOffset,
                      x: xBuf, y: yBuf,
                      m: UInt32(Sizes.M), n: UInt32(Sizes.N))
        cmd.commit()
        cmd.waitUntilCompleted()

        let yKernel: [Float] = (0..<Sizes.M).map { i in
            unsafe Float(yBuf.contents().load(fromByteOffset: i * MemoryLayout<Float16>.size,
                                       as: Float16.self))
        }
        let yRef = Self.cpuReferenceGemv(weightsFp32: weights, x: xFp32)

        var maxAbsDiff: Float = 0
        var refNorm: Float = 0
        for i in 0..<Sizes.M {
            maxAbsDiff = max(maxAbsDiff, abs(yKernel[i] - yRef[i]))
            refNorm    = max(refNorm, abs(yRef[i]))
        }
        let relErr = maxAbsDiff / max(refNorm, 1e-6)

        #expect(relErr < 5e-3,
                "kernel vs CPU ref relErr=\(relErr) maxAbsDiff=\(maxAbsDiff) refNorm=\(refNorm)")
    }
}
