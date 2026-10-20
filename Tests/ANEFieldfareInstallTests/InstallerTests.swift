import ANEFieldfareModelFormat
import Foundation
import Testing
@testable import ANEFieldfareInstallCore

/// A synthetic GGUF and tokenizer in a temporary directory, with pins that
/// match them.
private struct Fixture {
    let root: URL
    let ggufPath: String
    let bytes: [UInt8]
    let pin: PinnedFile
    let tokenizerDirectory: URL
    let tokenizerPins: [PinnedFile]
    var destination: URL { root.appendingPathComponent("model.fieldfare") }

    let mmprojPath: String
    let mmprojBytes: [UInt8]
    let mmprojPin: PinnedFile

    init(seed: UInt64 = 1) throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("fieldfare-install-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        bytes = GGUFFixture.gguf(GGUFFixture.tensors(seed: seed))
        ggufPath = root.appendingPathComponent("model.gguf").path
        try Data(bytes).write(to: URL(fileURLWithPath: ggufPath))
        pin = PinnedFile(repo: "fixture/gguf", revision: "r", path: "model.gguf",
                         size: UInt64(bytes.count), sha256: Installer.digest(bytes[...]))
        mmprojBytes = GGUFFixture.mmproj(GGUFFixture.visionTensors())
        mmprojPath = root.appendingPathComponent("mmproj.gguf").path
        try Data(mmprojBytes).write(to: URL(fileURLWithPath: mmprojPath))
        mmprojPin = PinnedFile(repo: "fixture/gguf", revision: "r", path: "mmproj.gguf",
                               size: UInt64(mmprojBytes.count), sha256: Installer.digest(mmprojBytes[...]))
        tokenizerDirectory = root.appendingPathComponent("tok", isDirectory: true)
        try FileManager.default.createDirectory(at: tokenizerDirectory, withIntermediateDirectories: true)
        var pins: [PinnedFile] = []
        for name in ["tokenizer.json", "tokenizer_config.json"] {
            let content = Array("{\"name\":\"\(name)\"}".utf8)
            try Data(content).write(to: tokenizerDirectory.appendingPathComponent(name))
            pins.append(PinnedFile(repo: "fixture/tok", revision: "r", path: name,
                                   size: UInt64(content.count), sha256: Installer.digest(content[...])))
        }
        tokenizerPins = pins
    }

    func options(source: (any ByteSource)? = nil, vision: Bool = false) throws -> InstallOptions {
        var options = InstallOptions(
            destination: destination,
            source: try source ?? LocalFileSource(path: ggufPath),
            sourceFile: pin,
            tokenizer: .directory(tokenizerDirectory),
            tokenizerFiles: tokenizerPins,
            modelID: "fixture/model")
        options.chunkBytes = 8 * 1024
        options.checkpointBytes = 16 * 1024
        options.headerPrefixBytes = 16 * 1024
        if vision { options.vision = (try LocalFileSource(path: mmprojPath), mmprojPin) }
        return options
    }

    func remove() { try? FileManager.default.removeItem(at: root) }
}

/// Serves the first `limit` bytes of a file, then fails every read.
private struct FailingSource: ByteSource {
    struct Interrupted: Error {}
    let inner: LocalFileSource
    let limit: UInt64
    var size: UInt64 { inner.size }
    var concurrency: Int { 1 }

    func read(_ range: Range<UInt64>) async throws -> [UInt8] {
        guard range.upperBound <= limit else { throw Interrupted() }
        return try await inner.read(range)
    }
}

private enum Decode {
    static func half(_ lo: UInt8, _ hi: UInt8) -> Float {
        Float(Float16(bitPattern: UInt16(lo) | UInt16(hi) << 8))
    }

    /// GGUF Q4_0 rows → values.
    static func ggufQ4_0(_ data: ArraySlice<UInt8>) -> [Float] {
        var out: [Float] = []
        for block in stride(from: data.startIndex, to: data.endIndex, by: 18) {
            let d = half(data[block], data[block + 1])
            var values = [Float](repeating: 0, count: 32)
            for j in 0..<16 {
                values[j] = d * Float(Int(data[block + 2 + j] & 0x0F) - 8)
                values[j + 16] = d * Float(Int(data[block + 2 + j] >> 4) - 8)
            }
            out += values
        }
        return out
    }

    /// `.fieldfare` q4_0 → values.
    static func q4_0(weights: ArraySlice<UInt8>, scales: ArraySlice<UInt8>) -> [Float] {
        var out: [Float] = []
        for (k, byte) in weights.enumerated() {
            let g = k / 16
            let d = half(scales[scales.startIndex + 2 * g], scales[scales.startIndex + 2 * g + 1])
            out.append(d * Float(Int(byte & 0x0F) - 8))
            out.append(d * Float(Int(byte >> 4) - 8))
        }
        return out
    }

    static func f32(_ data: ArraySlice<UInt8>) -> [Float] {
        stride(from: data.startIndex, to: data.endIndex, by: 4).map {
            Float(bitPattern: UInt32(data[$0]) | UInt32(data[$0 + 1]) << 8
                  | UInt32(data[$0 + 2]) << 16 | UInt32(data[$0 + 3]) << 24)
        }
    }

    static func bf16(_ data: ArraySlice<UInt8>) -> [Float] {
        stride(from: data.startIndex, to: data.endIndex, by: 2).map {
            Float(bitPattern: (UInt32(data[$0]) | UInt32(data[$0 + 1]) << 8) << 16)
        }
    }
}

@Suite struct InstallerTests {
    private func read(_ url: URL) throws -> [UInt8] { [UInt8](try Data(contentsOf: url)) }

    private func slice(_ bytes: [UInt8], _ region: FieldfareRegion, base: UInt64 = 0) -> ArraySlice<UInt8> {
        bytes[Int(base + region.offset)..<Int(base + region.offset + region.size)]
    }

    @Test func parsesTheFixtureHeader() throws {
        let tensors = GGUFFixture.tensors()
        let bytes = GGUFFixture.gguf(tensors)
        let header = try GGUFReader.parse(Data(bytes), fileSize: UInt64(bytes.count))
        #expect(header.tensors.count == tensors.count)
        #expect(header.metadata["tokenizer.ggml.merges"] == .longArray(count: 5000))
        for t in tensors {
            let parsed = try #require(header.tensor(t.name))
            #expect(parsed.dims == t.dims && parsed.type == t.type && parsed.size == UInt64(t.data.count))
            #expect(Array(bytes[Int(parsed.offset)..<Int(parsed.offset + parsed.size)]) == t.data)
        }
        #expect(throws: GGUFError.self) { try GGUFReader.parse(Data(bytes.prefix(200)), fileSize: UInt64(bytes.count)) }
    }

    @Test func installsAndConvertsEveryTensor() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let manifest = try await Installer.install(try fixture.options())
        let header = try GGUFReader.parse(Data(fixture.bytes), fileSize: UInt64(fixture.bytes.count))
        func source(_ name: String) throws -> ArraySlice<UInt8> {
            let t = try #require(header.tensor(name))
            return fixture.bytes[Int(t.offset)..<Int(t.offset + t.size)]
        }

        #expect(manifest.arch.fullAttentionLayers == [1])
        #expect(manifest.experts.stride % 16_384 == 0)
        #expect(Set(manifest.files.keys).isSuperset(of: ["resident.bin", "experts/layer_00.bin",
                                                          "tokenizer/tokenizer.json"]))
        #expect(!FileManager.default.fileExists(atPath: fixture.destination.path + ".partial"))
        let resident = try read(fixture.destination.appendingPathComponent("resident.bin"))
        let byName = Dictionary(uniqueKeysWithValues: manifest.resident.tensors.map { ($0.name, $0) })

        // Q4_0 tensors keep every value and scale.
        let q = try #require(byName["layers.0.self_attn.q_proj.weight"])
        #expect(q.encoding == .q4_0 && q.shape == [128, 256])
        let qScales = try #require(q.scales)
        #expect(Decode.q4_0(weights: slice(resident, q.weights), scales: slice(resident, qScales))
                == Decode.ggufQ4_0(try source("blk.0.attn_q.weight")))
        #expect(byName["layers.1.self_attn.v_proj.weight"] == nil)

        // Norms are BF16-exact in the source and copied exactly.
        let norm = try #require(byName["layers.1.post_feedforward_layernorm.weight"])
        #expect(Decode.bf16(slice(resident, norm.weights)) == Decode.f32(try source("blk.1.post_ffw_norm.weight")))

        // The Q6_K embedding is requantized to Q4_0 within one step of its scale.
        let embedding = try #require(byName["embed_tokens.weight"])
        var exact = [Float](repeating: 0, count: GGUFFixture.hidden * GGUFFixture.vocab)
        RowCodec.dequantizeQ6_K(try source("token_embd.weight"), into: &exact)
        let embedScales = try #require(embedding.scales)
        let requantized = Decode.q4_0(weights: slice(resident, embedding.weights),
                                      scales: slice(resident, embedScales))
        #expect(embedScales.size == UInt64(exact.count / 32 * 2))
        for g in 0..<(exact.count / 32) {
            let d = Decode.half(resident[Int(embedScales.offset) + 2 * g], resident[Int(embedScales.offset) + 2 * g + 1])
            for j in (32 * g)..<(32 * g + 32) {
                #expect(abs(requantized[j] - exact[j]) <= abs(d) * 1.01 + 1e-9)
            }
        }

        // The F32 router becomes affine8 within half a step of its scale.
        let router = try #require(byName["layers.0.router.proj.weight"])
        #expect(router.encoding == .affine8 && router.shape == [4, 256])
        let routerValues = Decode.f32(try source("blk.0.ffn_gate_inp.weight"))
        let rs = Decode.bf16(slice(resident, try #require(router.scales)))
        let rb = Decode.bf16(slice(resident, try #require(router.biases)))
        for (i, q) in slice(resident, router.weights).enumerated() {
            let value = Float(q) * rs[i / 64] + rb[i / 64]
            #expect(abs(value - routerValues[i]) <= rs[i / 64] * 0.5 + abs(routerValues[i]) * 0.01 + 1e-6)
        }

        // Experts: gate is the first half of gate_up's rows, up the second.
        let experts = Dictionary(uniqueKeysWithValues: manifest.experts.tensors.map { ($0.name, $0) })
        let gate = try #require(experts["gate"]), up = try #require(experts["up"]), down = try #require(experts["down"])
        let gateUp = Decode.ggufQ4_0(try source("blk.1.ffn_gate_up_exps.weight"))
        let downs = Decode.ggufQ4_0(try source("blk.1.ffn_down_exps.weight"))
        let layer = try read(fixture.destination.appendingPathComponent("experts/layer_01.bin"))
        let half = GGUFFixture.moe * GGUFFixture.hidden
        for e in 0..<GGUFFixture.experts {
            let base = UInt64(e) * manifest.experts.stride
            func values(_ t: FieldfareTensor) throws -> [Float] {
                Decode.q4_0(weights: slice(layer, t.weights, base: base),
                            scales: slice(layer, try #require(t.scales), base: base))
            }
            #expect(try values(gate) == Array(gateUp[(2 * e * half)..<((2 * e + 1) * half)]))
            #expect(try values(up) == Array(gateUp[((2 * e + 1) * half)..<((2 * e + 2) * half)]))
            #expect(try values(down) == Array(downs[(e * half)..<((e + 1) * half)]))
        }

        let verified = try await Installer.verify(fixture.destination)
        #expect(verified == manifest)
    }

    @Test func installsTheVisionTower() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let manifest = try await Installer.install(try fixture.options(vision: true))
        let vision = try #require(manifest.vision)
        #expect(vision.file == "vision.bin" && manifest.files["vision.bin"] != nil)
        #expect(manifest.source.files.map(\.path).contains("mmproj.gguf"))
        let byName = Dictionary(uniqueKeysWithValues: vision.tensors.map { ($0.name, $0) })
        #expect(vision.tensors.count == 5 + 13)
        #expect(byName.keys.filter { $0.hasPrefix("vision_tower.encoder.layers.0.") }.count == 13)
        let bytes = try read(fixture.destination.appendingPathComponent("vision.bin"))
        let header = try GGUFReader.parse(Data(fixture.mmprojBytes), fileSize: UInt64(fixture.mmprojBytes.count))
        func source(_ name: String) throws -> ArraySlice<UInt8> {
            let t = try #require(header.tensor(name))
            return fixture.mmprojBytes[Int(t.offset)..<Int(t.offset + t.size)]
        }

        // BF16 weights are copied; F32 norms are BF16-exact.
        let q = try #require(byName["vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight"])
        #expect(q.encoding == .bf16 && q.shape == [64, 64])
        #expect(slice(bytes, q.weights) == (try source("v.blk.0.attn_q.weight")))
        let norm = try #require(byName["vision_tower.encoder.layers.0.post_feedforward_layernorm.weight"])
        #expect(Decode.bf16(slice(bytes, norm.weights)) == Decode.f32(try source("v.blk.0.ffn_post_norm.weight")))

        // The conv kernel [out][c][y][x] becomes the linear row [out][y][x][c].
        let patch = try #require(byName["vision_tower.patch_embedder.input_proj.weight"])
        #expect(patch.shape == [64, 48])
        let kernel = Decode.f32(try source("v.patch_embd.weight"))
        let linear = Decode.bf16(slice(bytes, patch.weights))
        let size = GGUFFixture.patch
        for out in [0, 63] {
            for c in 0..<3 {
                for y in 0..<size {
                    for x in 0..<size {
                        #expect(linear[out * 48 + (y * size + x) * 3 + c] == kernel[((out * 3 + c) * size + y) * size + x])
                    }
                }
            }
        }

        // The projector is Q4_0 of its BF16 rows.
        let projector = try #require(byName["embed_vision.embedding_projection.weight"])
        #expect(projector.encoding == .q4_0 && projector.shape == [UInt64(GGUFFixture.hidden), 64])
        let exact = Decode.bf16(try source("mm.input_projection.weight"))
        let projectorScales = try #require(projector.scales)
        let values = Decode.q4_0(weights: slice(bytes, projector.weights), scales: slice(bytes, projectorScales))
        // At most one step of the group's scale (amax / 8), and an absolute
        // floor of 8 FP16 subnormal steps for groups whose scale underflows,
        // as in ggml.
        for g in 0..<(exact.count / 32) {
            let step = (exact[(32 * g)..<(32 * g + 32)].map(abs).max() ?? 0) / 8
            for j in (32 * g)..<(32 * g + 32) { #expect(abs(values[j] - exact[j]) <= step * 1.01 + 5e-7, "g=\(g) j=\(j) got=\(values[j]) want=\(exact[j]) step=\(step)") }
        }
        #expect(try await Installer.verify(fixture.destination) == manifest)
    }

    @Test func resumesAfterAnInterruption() async throws {
        let clean = try Fixture()
        defer { clean.remove() }
        let expected = try await Installer.install(try clean.options())

        let fixture = try Fixture()
        defer { fixture.remove() }
        let interrupted = FailingSource(inner: try LocalFileSource(path: fixture.ggufPath),
                                        limit: UInt64(fixture.bytes.count) / 2)
        await #expect(throws: FailingSource.Interrupted.self) {
            try await Installer.install(try fixture.options(source: interrupted))
        }
        let partial = URL(fileURLWithPath: fixture.destination.path + ".partial")
        #expect(FileManager.default.fileExists(atPath: Installer.checkpointURL("text", partial: partial).path))
        let resumed = try await Installer.install(try fixture.options())
        #expect(resumed == expected)
        for path in expected.files.keys {
            #expect(try read(fixture.destination.appendingPathComponent(path))
                    == read(clean.destination.appendingPathComponent(path)))
        }
    }

    @Test func rejectsASourceThatDoesNotMatchItsDigest() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        var options = try fixture.options()
        options.sourceFile = PinnedFile(repo: "r", revision: "r", path: "model.gguf",
                                        size: fixture.pin.size, sha256: String(repeating: "0", count: 64))
        await #expect(throws: SourceError.self) { try await Installer.install(options) }
        #expect(!FileManager.default.fileExists(atPath: fixture.destination.path))
        #expect(!FileManager.default.fileExists(atPath: fixture.destination.path + ".partial"))
    }

    @Test func verifyCatchesAChangedFile() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try await Installer.install(try fixture.options())
        let url = fixture.destination.appendingPathComponent("experts/layer_00.bin")
        var bytes = try read(url)
        bytes[100] ^= 1
        try Data(bytes).write(to: url)
        await #expect(throws: InstallError.self) { try await Installer.verify(fixture.destination) }
    }

    @Test func refusesToReplaceAnInstallWithoutOverwrite() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        try await Installer.install(try fixture.options())
        await #expect(throws: InstallError.alreadyInstalled(fixture.destination.standardizedFileURL.path)) {
            try await Installer.install(try fixture.options())
        }
        var options = try fixture.options()
        options.overwrite = true
        try await Installer.install(options)
    }
}

@Suite struct RowCodecTests {
    /// ggml's reference: `d = m / -8` for the first largest-magnitude value
    /// `m`, `q = min(15, trunc(x / d + 8.5))`.
    @Test func quantizesQ4_0LikeGGML() {
        var values = [Float](repeating: 0.25, count: 32)
        values[3] = -2       // the largest magnitude: d = 0.25, maps to 0
        values[7] = 2        // the opposite extreme clamps to 15
        values[8] = 0
        var weights: [UInt8] = [], scales: [UInt8] = []
        RowCodec.quantizeQ4_0(values, weights: &weights, scales: &scales)
        #expect(scales == [0x00, 0x34]) // FP16 0.25
        let q = weights.flatMap { [$0 & 0x0F, $0 >> 4] }
        #expect(q[3] == 0 && q[7] == 15 && q[8] == 8 && q[0] == 9)
    }

    @Test func repacksQ4_0Nibbles() {
        // Block values v[j] = j % 16 for the low half and 15 - j % 16 for the high half.
        var block: [UInt8] = [0x00, 0x3C]
        for j in 0..<16 { block.append(UInt8(j) | UInt8(15 - j) << 4) }
        var weights: [UInt8] = [], scales: [UInt8] = []
        RowCodec.repackQ4_0(block[...], cols: 32, weights: &weights, scales: &scales)
        #expect(scales == [0x00, 0x3C])
        let values = weights.flatMap { [Int($0 & 0x0F), Int($0 >> 4)] }
        #expect(values == Array(0..<16) + (0..<16).map { 15 - $0 })
    }
}
