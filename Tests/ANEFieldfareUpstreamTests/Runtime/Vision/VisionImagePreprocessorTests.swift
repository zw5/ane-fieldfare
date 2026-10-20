import Foundation
import Metal
import Testing
@testable import ANEFieldfareHotpath

@Suite struct VisionImagePreprocessorTests {
    private struct Corpus: Decodable {
        struct Case: Decodable {
            let file: String
            let processedWidth: Int
            let processedHeight: Int
            let gridWidth: Int
            let gridHeight: Int

            enum CodingKeys: String, CodingKey {
                case file
                case processedWidth = "processed_width"
                case processedHeight = "processed_height"
                case gridWidth = "grid_width"
                case gridHeight = "grid_height"
            }
        }

        let cases: [Case]
    }

    @Test func geometryMatchesReferenceProcessor() throws {
        let square = try Gemma4ImageGeometry(sourceWidth: 641, sourceHeight: 641)
        #expect(square.processedWidth == 768)
        #expect(square.processedHeight == 768)
        #expect(square.patchGridWidth == 48)
        #expect(square.patchGridHeight == 48)
        #expect(square.softTokenCount == 256)

        let fortyEightMegapixels = try Gemma4ImageGeometry(
            sourceWidth: 8_000, sourceHeight: 6_000)
        #expect(fortyEightMegapixels.processedWidth == 912)
        #expect(fortyEightMegapixels.processedHeight == 672)
        #expect(fortyEightMegapixels.patchGridWidth == 57)
        #expect(fortyEightMegapixels.patchGridHeight == 42)
        #expect(fortyEightMegapixels.patchCount == 2_394)
        #expect(fortyEightMegapixels.softTokenCount == 266)

        let veryWide = try Gemma4ImageGeometry(sourceWidth: 32_768, sourceHeight: 1)
        #expect(veryWide.processedWidth == 13_440)
        #expect(veryWide.processedHeight == 48)
        #expect(veryWide.patchCount == 2_520)
    }

    @Test func frozenCorpusProducesExactGeometryAndPositions() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let root = try #require(Self.corpusRoot)
        let corpus = try JSONDecoder().decode(
            Corpus.self,
            from: Data(contentsOf: root.appendingPathComponent("corpus.json")))
        let preprocessor = Gemma4ImagePreprocessor(device: device)

        for fixture in corpus.cases {
            let result = try preprocessor.preprocess(
                preprocessor.plan(fileURL: root.appendingPathComponent(fixture.file)))
            #expect(result.geometry.processedWidth == fixture.processedWidth)
            #expect(result.geometry.processedHeight == fixture.processedHeight)
            #expect(result.geometry.patchGridWidth == fixture.gridWidth)
            #expect(result.geometry.patchGridHeight == fixture.gridHeight)
            #expect(result.patchesBF16.length
                    == result.geometry.patchCount * VisionConfig.patchDimension * 2)
            #expect(result.positionsInt32x2.length
                    == result.geometry.patchCount * 2 * MemoryLayout<Int32>.stride)

            let positions = unsafe result.positionsInt32x2.contents().bindMemory(
                to: Int32.self, capacity: result.geometry.patchCount * 2)
            for row in 0..<result.geometry.patchCount {
                unsafe #expect(positions[row * 2] == Int32(row % fixture.gridWidth))
                unsafe #expect(positions[row * 2 + 1] == Int32(row / fixture.gridWidth))
            }
            let patches = unsafe result.patchesBF16.contents().bindMemory(
                to: UInt16.self,
                capacity: result.geometry.patchCount * VisionConfig.patchDimension)
            let count = result.geometry.patchCount * VisionConfig.patchDimension
            for index in stride(from: 0, to: count, by: 251) {
                let value = unsafe Quantization.bf16ToFloat(patches[index])
                #expect(value.isFinite && value >= 0 && value <= 1)
            }
        }
    }

    @Test func rejectsOversizedSourceBeforeDecode() throws {
        let image = try #require(Self.corpusRoot).appendingPathComponent("natural-square.png")
        let preprocessor = Gemma4ImagePreprocessor(
            device: try #require(MTLCreateSystemDefaultDevice()),
            limits: VisionImageLimits(maximumEncodedBytes: 1))
        #expect(throws: VisionImageError.self) {
            try preprocessor.plan(fileURL: image)
        }
    }

    @Test func rejectsSymlinkSource() throws {
        let target = try #require(Self.corpusRoot).appendingPathComponent("natural-square.png")
        let link = FileManager.default.temporaryDirectory
            .appendingPathComponent("vision-link-\(UUID().uuidString).png")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)
        defer { try? FileManager.default.removeItem(at: link) }
        let preprocessor = Gemma4ImagePreprocessor(
            device: try #require(MTLCreateSystemDefaultDevice()))
        #expect(throws: VisionImageError.self) {
            try preprocessor.plan(fileURL: link)
        }
    }

    /// The frozen corpus, shipped as a test resource.
    private static var corpusRoot: URL? {
        Bundle.module.url(forResource: "images", withExtension: nil)
    }
}
