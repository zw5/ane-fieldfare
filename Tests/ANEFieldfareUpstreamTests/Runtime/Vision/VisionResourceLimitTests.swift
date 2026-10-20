import CoreGraphics
import Foundation
import ImageIO
import Testing
@testable import ANEFieldfareHotpath

/// Resource-limit guards. Each test fails against the behaviour each comment
/// describes.
@Suite struct VisionResourceLimitTests {
    private static func temporaryDirectory(_ prefix: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(prefix)-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private static func read(_ url: URL, limits: VisionImageLimits) throws {
        let opened = try VisionImageSource(fileURL: url)
            .open(maximumEncodedBytes: limits.maximumEncodedBytes)
        _ = try ImageMetadataReader(limits: limits).read(opened: opened)
    }

    /// The per-side, pixel and decoded-bytes guards all threw the pixel-limit
    /// error, so two of the three sent the user to resize against a constraint
    /// their image already met.
    @Test func eachRejectionGuardNamesItsOwnLimit() throws {
        let directory = try Self.temporaryDirectory("limit-errors")
        defer { try? FileManager.default.removeItem(at: directory) }
        let wide = directory.appendingPathComponent("wide.png")
        try Self.writeSolidPNG(width: 300, height: 50, to: wide)
        let square = directory.appendingPathComponent("square.png")
        try Self.writeSolidPNG(width: 320, height: 320, to: square)

        // 300x50 is far under the default pixel cap; only a side exceeds.
        #expect {
            try Self.read(wide, limits: VisionImageLimits(maximumSourceDimension: 100))
        } throws: { error in
            guard case VisionImageError.sideTooLarge(
                let width, let height, let limit) = error else { return false }
            return width == 300 && height == 50 && limit == 100
        }

        // Both sides fit; only the pixel product exceeds.
        #expect {
            try Self.read(wide, limits: VisionImageLimits(maximumSourcePixels: 10_000))
        } throws: { error in
            guard case VisionImageError.dimensionsTooLarge(
                let width, let height, let limit) = error else { return false }
            return width == 300 && height == 50 && limit == 10_000
        }

        // Sides and pixels fit; only the decoded-surface ceiling exceeds. The
        // ceiling is inclusive: exactly at it is admitted, one byte under it
        // is refused, which catches a `<=` quietly changing to `<`.
        let decodedBytes = 320 * 320 * 4
        try Self.read(square, limits: VisionImageLimits(maximumDecodedBytes: decodedBytes))
        #expect {
            try Self.read(square, limits: VisionImageLimits(maximumDecodedBytes: decodedBytes - 1))
        } throws: { error in
            guard case VisionImageError.decodedBytesTooLarge(
                let width, let height, let limit) = error else { return false }
            return width == 320 && height == 320 && limit == decodedBytes - 1
        }

        // The shipped ceiling must admit the largest supported 8-bit camera
        // input. This fails if the earlier 96 MiB low-end policy returns.
        let shipped = VisionImageLimits()
        let pixels = 8_000 * 6_000
        #expect(pixels <= shipped.maximumSourcePixels)
        #expect(8_000 <= shipped.maximumSourceDimension)
        #expect(pixels * 4 <= shipped.maximumDecodedBytes)
    }

    private static func writeSolidPNG(width: Int, height: Int, to url: URL) throws {
        let colorSpace = try #require(CGColorSpace(name: CGColorSpace.sRGB))
        let bitmap = unsafe try #require(CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: width * 4, space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        bitmap.setFillColor(CGColor(gray: 0.5, alpha: 1))
        bitmap.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let image = try #require(bitmap.makeImage())
        let destination = try #require(CGImageDestinationCreateWithURL(
            url as CFURL, "public.png" as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
    }
}
