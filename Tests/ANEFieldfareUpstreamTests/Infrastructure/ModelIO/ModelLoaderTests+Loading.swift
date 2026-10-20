import Darwin
import Foundation
import Metal
import Testing

@testable import ANEFieldfareHotpath
import ANEFieldfareModelFormat

extension ModelLoaderTests {
  private func load(_ dir: URL, expecting: ArchConfig = .gemma4Toy()) throws -> Model {
    let device = try #require(MTLCreateSystemDefaultDevice())
    return try Model.load(directoryURL: dir, device: device, expecting: expecting)
  }

  @Test func loadsValidDirectoryAndExposesResidentBytes() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    let model = try load(dir)
    let embed = model.embedding
    #expect(embed.length == UInt64(1024 * 64 / 2))
    #expect(embed.scaleLength == UInt64(1024 * 64 / 32 * 2))
    #expect(embed.biasLength == 0)
    #expect(embed.shape.0 == 1024 && embed.shape.1 == 64)
    // Tied lm_head returns the same view as embedding.
    #expect(model.lmHead.offset == model.embedding.offset)
    #expect(model.lmHead.length == model.embedding.length)
    let norm = model.finalNorm
    #expect(norm.length == UInt64(64 * 2))
    let contents = unsafe norm.buffer.contents()
    for i in 0..<Int(norm.length) {
      let got = unsafe contents.load(fromByteOffset: Int(norm.offset) + i, as: UInt8.self)
      #expect(got == UInt8(0xC0 | (i & 0x3F)), "norm byte \(i)")
    }
    let router = try model.router(layer: 1)
    #expect(router.biasLength == UInt64(8 * 64 / 64 * 2))
  }

  @Test func missingManifestFailsPartialInstall() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try FileManager.default.removeItem(at: dir.appendingPathComponent("manifest.json"))
    #expect {
      _ = try load(dir)
    } throws: { error in
      if case ModelError.partialInstall = error { return true }
      return false
    }
  }

  @Test func rejectsManifestFIFOWithoutBlocking() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    let manifest = dir.appendingPathComponent("manifest.json")
    try FileManager.default.removeItem(at: manifest)
    unsafe #expect(mkfifo(manifest.path, 0o600) == 0)
    #expect(throws: ModelError.self) { _ = try load(dir) }
  }

  @Test func rejectsArchitectureMismatch() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    #expect {
      _ = try load(dir, expecting: .gemma4_26B_A4B)
    } throws: { error in
      if case ModelError.archMismatch(let field, _, _) = error { return field == "hiddenSize" }
      return false
    }
  }

  @Test func rejectsAModelMissingATensorTheRuntimeBinds() throws {
    let dir = try Self.writeToySynthetic { manifest in
      let tensors = manifest.resident.tensors.filter { $0.name != "layers.1.self_attn.q_proj.weight" }
      manifest = FieldfareManifest(modelID: manifest.modelID, source: manifest.source, arch: manifest.arch,
                                   files: manifest.files,
                                   resident: FieldfareTensorFile(file: manifest.resident.file, tensors: tensors),
                                   experts: manifest.experts, vision: nil)
    }
    defer { try? FileManager.default.removeItem(at: dir) }
    #expect {
      _ = try load(dir)
    } throws: { error in
      if case ModelError.invalidManifest(let detail) = error { return detail.contains("q_proj") }
      return false
    }
  }

  @Test func rejectsATruncatedResidentFile() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    let url = dir.appendingPathComponent("resident.bin")
    let data = try Data(contentsOf: url)
    try data.prefix(data.count - 64).write(to: url)
    #expect(throws: ModelError.self) { _ = try load(dir) }
  }

  @Test func rejectsUnsupportedFormatVersion() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    let url = dir.appendingPathComponent("manifest.json")
    let text = try String(contentsOf: url, encoding: .utf8)
      .replacingOccurrences(of: "\"version\":1", with: "\"version\":2")
    try text.write(to: url, atomically: true, encoding: .utf8)
    #expect {
      _ = try load(dir)
    } throws: { error in
      if case ModelError.invalidManifest(let detail) = error { return detail.contains("version") }
      return false
    }
  }
}
