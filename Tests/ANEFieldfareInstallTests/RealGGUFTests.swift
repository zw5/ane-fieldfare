import ANEFieldfareModelFormat
import Darwin
import Foundation
import Testing
@testable import ANEFieldfareInstallCore

/// The pinned GGUF, or a prefix of it: `ANE_FIELDFARE_TEST_GGUF` names a file
/// holding the GGUF's first bytes (the whole file by default, at the
/// repository's download location).
private let realGGUF: (path: String, readable: UInt64)? = {
    let path = ProcessInfo.processInfo.environment["ANE_FIELDFARE_TEST_GGUF"]
        ?? URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("../../checkpoints/source/gemma-4-26B_q4_0-it.gguf").standardizedFileURL.path
    var st = stat()
    guard unsafe stat(path, &st) == 0, st.st_size > 64 * 1024 * 1024 else { return nil }
    return (path, UInt64(st.st_size))
}()

@Suite(.enabled(if: realGGUF != nil, "Google's QAT GGUF (or a prefix of it) required"))
struct RealGGUFTests {
    private func read(_ range: Range<UInt64>) throws -> [UInt8] {
        let file = try #require(realGGUF)
        let fd = unsafe open(file.path, O_RDONLY | O_CLOEXEC)
        try #require(fd >= 0)
        defer { close(fd) }
        return try POSIXFile.read(fd: fd, range: range, path: file.path)
    }

    private func header() throws -> GGUFHeader {
        try GGUFReader.parse(Data(try read(0..<(32 * 1024 * 1024))), fileSize: SupportedModel.gguf.size)
    }

    /// The plan the installer makes for the pinned file: the architecture
    /// the runtime runs, and the expert layout its kernels expect.
    @Test func plansTheRuntimeLayout() throws {
        let plan = try Gemma4Plan.plan(try header())
        #expect(plan.arch.hiddenSize == 2816 && plan.arch.numLayers == 30 && plan.arch.numExperts == 128)
        #expect(plan.arch.fullAttentionLayers == [5, 11, 17, 23, 29])
        #expect(plan.arch.numKVHeads == 8 && plan.arch.numFullKVHeads == 2)
        #expect(plan.arch.headDim == 256 && plan.arch.fullHeadDim == 512 && plan.arch.vocabSize == 262_144)
        #expect(plan.experts.stride == 3_358_720)
        let offsets: [UInt64] = plan.experts.tensors.flatMap { t -> [UInt64] in [t.weights.offset, t.scales?.offset ?? 0] }
        #expect(offsets == [0, 991_232, 1_115_136, 2_106_368, 2_230_272, 3_221_504])
        #expect(plan.files.count == 31)
        #expect(plan.files.dropFirst().allSatisfy { $0.size == 128 * 3_358_720 })
        let resident = plan.resident.tensors
        #expect(resident.count == 2 + 30 * 20 - 5)
        #expect(resident.first { $0.name == "embed_tokens.weight" }?.encoding == .q4_0)
        #expect(resident.first { $0.name == "layers.3.router.proj.weight" }?.encoding == .affine8)
    }

    /// Layer 0 converted from the real bytes: attention keeps every Q4_0
    /// value, and expert 0's gate is the first half of its gate_up rows.
    @Test func convertsLayerZero() throws {
        let header = try header()
        let plan = try Gemma4Plan.plan(header)
        let q = try #require(header.tensor("blk.0.attn_q.weight"))
        let gateUp = try #require(header.tensor("blk.0.ffn_gate_up_exps.weight"))
        let end = gateUp.offset + gateUp.rowBytes * 704
        try #require(end <= (realGGUF?.readable ?? 0), "the prefix does not reach layer 0's experts")

        let start = q.offset
        let segments = plan.segments.filter { $0.source >= start && $0.sourceEnd <= end }
        var outputs: [Int: [UInt64: [UInt8]]] = [:]
        var converter = StreamingConverter(
            segments: segments, resumingAt: ConversionCheckpoint(offset: start, hash: ResumableSHA256()))
        try converter.consume(try read(start..<end)[...], at: start) { file, offset, bytes in
            outputs[file, default: [:]][offset] = bytes
        }

        let qTensor = try #require(plan.resident.tensors.first { $0.name == "layers.0.self_attn.q_proj.weight" })
        let qScales = try #require(qTensor.scales)
        let source = try read(q.offset..<(q.offset + q.size))
        var weights: [UInt8] = [], scales: [UInt8] = []
        RowCodec.repackQ4_0(source[...], cols: 2816, weights: &weights, scales: &scales)
        #expect(outputs[0]?[qTensor.weights.offset] == weights)
        #expect(outputs[0]?[qScales.offset] == scales)

        let gate = try #require(plan.experts.tensors.first)
        let expert0 = try read(gateUp.offset..<(gateUp.offset + gateUp.rowBytes * 704))
        weights = []; scales = []
        RowCodec.repackQ4_0(expert0[...], cols: 2816, weights: &weights, scales: &scales)
        #expect(outputs[1]?[gate.weights.offset] == weights, "layer 0 file, expert 0 gate")
    }
}
