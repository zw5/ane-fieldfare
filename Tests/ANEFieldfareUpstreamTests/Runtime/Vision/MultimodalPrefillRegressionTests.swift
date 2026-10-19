import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import Testing
@testable import ANEFieldfareHotpath

/// Regressions from the PR review of this branch. Every one of them is a
/// second call, a resumed turn, or a shape the fixtures never had — which is
/// exactly what the suite did not cover.
@Suite struct MultimodalPrefillRegressionTests {
    /// Truncation is ImageIO's question to answer, not a byte scan's.
    ///
    /// The scan could not get this right in either direction: a truncated photo
    /// still carries the EOI of its EXIF thumbnail near the front, so it passed
    /// and was decoded with gray filler, while a Motion Photo with more than
    /// 4 MB of appended video pushed its real EOI past the search window and was
    /// rejected. Both cases are pinned here.
    @Test func truncatedImagesAreRejectedAndTrailingDataIsAccepted() throws {
        let manager = FileManager.default
        let directory = manager.temporaryDirectory
            .appendingPathComponent("jpeg-\(UUID().uuidString)", isDirectory: true)
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? manager.removeItem(at: directory) }

        // A real JPEG, encoded by ImageIO so the entropy-coded stream is valid.
        let colorSpace = try #require(CGColorSpace(name: CGColorSpace.sRGB))
        let bitmap = unsafe try #require(CGContext(
            data: nil, width: 640, height: 480, bitsPerComponent: 8,
            bytesPerRow: 640 * 4, space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        bitmap.setFillColor(CGColor(red: 0.2, green: 0.6, blue: 0.9, alpha: 1))
        bitmap.fill(CGRect(x: 0, y: 0, width: 640, height: 480))
        let image = try #require(bitmap.makeImage())
        let whole = directory.appendingPathComponent("whole.jpg")
        let destination = try #require(CGImageDestinationCreateWithURL(
            whole as CFURL, UTType.jpeg.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
        let encoded = try Data(contentsOf: whole)

        let reader = ImageMetadataReader(limits: VisionImageLimits())
        func read(_ url: URL) throws -> VisionImageMetadata {
            let opened = try VisionImageSource(fileURL: url)
                .open(maximumEncodedBytes: VisionImageLimits().maximumEncodedBytes)
            return try reader.read(opened: opened)
        }

        // Intact: accepted.
        #expect(try read(whole).orientedWidth == 640)

        // Trailing data past EOI, more than the old 4 MiB window: still accepted.
        let motion = directory.appendingPathComponent("motion.jpg")
        var withVideo = encoded
        withVideo.append(Data(repeating: 0x11, count: 8 * 1_024 * 1_024))
        try withVideo.write(to: motion)
        #expect(try read(motion).orientedWidth == 640,
                "a file with data after EOI was rejected")

        // Truncated mid-scan: rejected, where the byte scan let it through
        // because the thumbnail's own EOI sits near the front.
        let truncated = directory.appendingPathComponent("truncated.jpg")
        try encoded.prefix(encoded.count / 2).write(to: truncated)
        #expect(throws: VisionImageError.self) { _ = try read(truncated) }
    }

    /// PNG had no completeness check at all, so a file cut short was accepted,
    /// decoded with the undecoded remainder black, and described as "a solid,
    /// uniform black square" - an answer about an image that was never
    /// delivered. The JPEG guard next door had covered this class since it was
    /// written; PNG simply was not asked.
    @Test func aTruncatedPNGIsRejectedRatherThanDecodedAsBlack() throws {
        let manager = FileManager.default
        let directory = manager.temporaryDirectory
            .appendingPathComponent("png-\(UUID().uuidString)", isDirectory: true)
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? manager.removeItem(at: directory) }

        let colorSpace = try #require(CGColorSpace(name: CGColorSpace.sRGB))
        let bitmap = unsafe try #require(CGContext(
            data: nil, width: 320, height: 240, bitsPerComponent: 8,
            bytesPerRow: 320 * 4, space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        // Noise rather than a flat fill, so the file does not compress to a
        // size where "half of it" still carries every row.
        for x in stride(from: 0, to: 320, by: 2) {
            for y in stride(from: 0, to: 240, by: 2) {
                bitmap.setFillColor(CGColor(
                    red: Double((x &* 7) % 255) / 255, green: Double((y &* 13) % 255) / 255,
                    blue: Double((x &+ y) % 255) / 255, alpha: 1))
                bitmap.fill(CGRect(x: x, y: y, width: 2, height: 2))
            }
        }
        let image = try #require(bitmap.makeImage())
        let whole = directory.appendingPathComponent("whole.png")
        let destination = try #require(CGImageDestinationCreateWithURL(
            whole as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
        let encoded = try Data(contentsOf: whole)

        let reader = ImageMetadataReader(limits: VisionImageLimits())
        func read(_ url: URL) throws -> VisionImageMetadata {
            let opened = try VisionImageSource(fileURL: url)
                .open(maximumEncodedBytes: VisionImageLimits().maximumEncodedBytes)
            return try reader.read(opened: opened)
        }

        #expect(try read(whole).orientedWidth == 320, "an intact PNG was rejected")

        let truncated = directory.appendingPathComponent("truncated.png")
        try encoded.prefix(encoded.count / 2).write(to: truncated)
        #expect(throws: VisionImageError.self) { _ = try read(truncated) }

        // One byte short of IEND is the boundary the check actually draws.
        let clipped = directory.appendingPathComponent("clipped.png")
        try encoded.prefix(encoded.count - 1).write(to: clipped)
        #expect(throws: VisionImageError.self) { _ = try read(clipped) }
    }

    /// A 0xFF fill byte before a marker is legal (ITU T.81 B.1.1.2) and real
    /// scanners emit it. Reading it as the marker itself made the next two
    /// bytes a length, which jumped into the entropy data and reported an
    /// intact photo as truncated.
    @Test func fillBytesBeforeMarkersDoNotLookLikeTruncation() throws {
        let manager = FileManager.default
        let directory = manager.temporaryDirectory
            .appendingPathComponent("jpeg-fill-\(UUID().uuidString)", isDirectory: true)
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? manager.removeItem(at: directory) }

        let colorSpace = try #require(CGColorSpace(name: CGColorSpace.sRGB))
        let bitmap = unsafe try #require(CGContext(
            data: nil, width: 320, height: 240, bitsPerComponent: 8,
            bytesPerRow: 320 * 4, space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        bitmap.setFillColor(CGColor(red: 0.9, green: 0.4, blue: 0.1, alpha: 1))
        bitmap.fill(CGRect(x: 0, y: 0, width: 320, height: 240))
        let image = try #require(bitmap.makeImage())
        let source = directory.appendingPathComponent("plain.jpg")
        let destination = try #require(CGImageDestinationCreateWithURL(
            source as CFURL, UTType.jpeg.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
        let encoded = [UInt8](try Data(contentsOf: source))

        // Pad every marker in the header with extra 0xFF fill bytes, stopping
        // at the start of scan so the entropy data is untouched.
        var padded: [UInt8] = [encoded[0], encoded[1]]
        var index = 2
        while index + 3 < encoded.count {
            guard encoded[index] == 0xFF else { break }
            let marker = encoded[index + 1]
            let length = Int(encoded[index + 2]) << 8 | Int(encoded[index + 3])
            padded.append(contentsOf: [0xFF, 0xFF])
            if marker == 0xDA {
                padded.append(contentsOf: encoded[index...])
                break
            }
            padded.append(contentsOf: encoded[index..<(index + 2 + length)])
            index += 2 + length
        }

        let url = directory.appendingPathComponent("filled.jpg")
        try Data(padded).write(to: url)
        let opened = try VisionImageSource(fileURL: url)
            .open(maximumEncodedBytes: VisionImageLimits().maximumEncodedBytes)
        #expect(opened.hasCompleteJPEGStream(),
                "a legal fill byte was read as truncation")
        // And the whole path accepts it, since that is where users meet it.
        let metadata = try ImageMetadataReader(limits: VisionImageLimits())
            .read(opened: opened)
        #expect(metadata.orientedWidth == 320)
    }

    /// The whole reduction belongs to our filter, which is the pinned
    /// reference's. Decoding at the target handed it to ImageIO instead, and on
    /// a UI screenshot that changed the model's transcription — measured
    /// 0.0984 patch NRMSE against 0.0096 for a photo, and whole lines of text
    /// lost. The cost that once justified the target-size decode was a
    /// debug-build artifact; on the GPU the reference reduction is a few
    /// milliseconds.
    @Test func adownscaleDecodesAtTheSourceSize() throws {
        let metadata = VisionImageMetadata(
            encodedBytes: 1, encodedWidth: 4_032, encodedHeight: 3_024,
            orientedWidth: 4_032, orientedHeight: 3_024, orientation: 1,
            bitsPerComponent: 8, colorModel: "RGB", typeIdentifier: "public.jpeg")
        let geometry = try Gemma4ImageGeometry(
            sourceWidth: 4_032, sourceHeight: 3_024, config: VisionConfig())
        #expect(geometry.processedWidth < 4_032, "this fixture is not a downscale")
        #expect(Gemma4ImagePreprocessor.decodeMaxPixelSize(
            metadata: metadata, geometry: geometry) == 4_032,
                "the reduction was handed back to ImageIO's scaler")

        // An upscale still decodes at its own size; there is nothing to reduce.
        let small = VisionImageMetadata(
            encodedBytes: 1, encodedWidth: 64, encodedHeight: 64,
            orientedWidth: 64, orientedHeight: 64, orientation: 1,
            bitsPerComponent: 8, colorModel: "RGB", typeIdentifier: "public.png")
        let smallGeometry = try Gemma4ImageGeometry(
            sourceWidth: 64, sourceHeight: 64, config: VisionConfig())
        #expect(Gemma4ImagePreprocessor.decodeMaxPixelSize(
            metadata: small, geometry: smallGeometry) == 64)
    }
}
