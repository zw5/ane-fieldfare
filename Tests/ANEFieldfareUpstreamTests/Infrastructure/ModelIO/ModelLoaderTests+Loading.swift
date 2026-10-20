import Foundation
import Metal
import Testing

@testable import ANEFieldfareHotpath
@testable import ANEFieldfareImportedFormat

extension ModelLoaderTests {
  @Test func loadsValidDirectoryAndExposesResidentBytes() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    let device = try #require(MTLCreateSystemDefaultDevice())
    let model = try Model.load(
      directoryURL: dir, device: device,
      expecting: .gemma4Toy())
    let embed = model.embedding
    #expect(embed.length == UInt64(1024 * 64 / 2))
    #expect(embed.shape.0 == 1024 && embed.shape.1 == 64)
    // Tied lm_head returns the same view as embedding.
    #expect(model.lmHead.offset == model.embedding.offset)
    #expect(model.lmHead.length == model.embedding.length)
    let norm = model.finalNorm
    #expect(norm.length == UInt64(64 * 2))
    let contents = unsafe norm.buffer.contents()
    // Norm region was patterned 0xC0 | (i & 0x3F).
    for i in 0..<Int(norm.length) {
      let got = unsafe contents.load(fromByteOffset: Int(norm.offset) + i, as: UInt8.self)
      #expect(got == UInt8(0xC0 | (i & 0x3F)), "norm byte \(i)")
    }
  }

  @Test func missingManifestFailsPartialInstall() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try FileManager.default.removeItem(at: dir.appendingPathComponent("manifest.json"))
    let device = try #require(MTLCreateSystemDefaultDevice())
    #expect {
      _ = try Model.load(
        directoryURL: dir, device: device,
        expecting: .gemma4Toy())
    } throws: { error in
      if case ModelError.partialInstall = error { return true }
      return false
    }
  }

  @Test func malformedZeroPayloadResidentEntryThrowsWithoutTrap() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    let weightsURL = dir.appendingPathComponent("model_weights.bin")
    var weights = try Data(contentsOf: weightsURL)
    let firstEntry = GTurboFormatV1.residentHeaderBytes
    weights.replaceSubrange((firstEntry + 8)..<(firstEntry + 24),
                            with: repeatElement(UInt8(0), count: 16))
    try weights.write(to: weightsURL)

    let device = try #require(MTLCreateSystemDefaultDevice())
    #expect {
      _ = try Model.load(directoryURL: dir, device: device, expecting: .gemma4Toy())
    } throws: { error in
      if case ModelError.indexCorrupt(let detail) = error {
        return detail.contains("primary payload is empty")
      }
      return false
    }
  }
}
