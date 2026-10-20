import Foundation
import ImageIO
import UniformTypeIdentifiers

public struct VisionImageLimits: Sendable, Equatable {
    public let maximumEncodedBytes: Int
    public let maximumSourcePixels: Int
    public let maximumSourceDimension: Int
    /// Decoders the pack is validated against. ImageIO sniffs the real bytes, so
    /// without this a request labelled `image/png` can reach the camera-RAW or
    /// EXR decoder — a far larger surface than the formats actually supported.
    public let allowedTypeIdentifiers: Set<String>
    /// Ceiling on one decoded surface, in bytes rather than pixels, since a
    /// 16-bit image decodes to about four times an 8-bit one. 200,000,000
    /// bytes is an 8-bit RGBA surface at the 50-million-pixel source limit.
    public let maximumDecodedBytes: Int

    public init(
        maximumEncodedBytes: Int = 64 * 1_024 * 1_024,
        maximumSourcePixels: Int = 50_000_000,
        maximumSourceDimension: Int = 32_768,
        allowedTypeIdentifiers: Set<String> = [
            "public.jpeg", "public.png", "public.heic", "public.heif",
            "public.tiff", "com.compuserve.gif", "public.heics",
        ],
        maximumDecodedBytes: Int = 200_000_000
    ) {
        precondition(
            maximumEncodedBytes > 0 && maximumSourcePixels > 0
                && maximumSourceDimension > 0)
        self.maximumEncodedBytes = maximumEncodedBytes
        self.maximumSourcePixels = maximumSourcePixels
        self.maximumSourceDimension = maximumSourceDimension
        self.allowedTypeIdentifiers = allowedTypeIdentifiers
        self.maximumDecodedBytes = maximumDecodedBytes
    }
}

public struct VisionImageMetadata: Sendable, Equatable {
    public let orientedWidth: Int
    public let orientedHeight: Int
}

public struct ImageMetadataReader: Sendable {
    public let limits: VisionImageLimits

    public init(limits: VisionImageLimits) {
        self.limits = limits
    }

    package func read(opened: OpenedVisionImage) throws -> VisionImageMetadata {
        let source = opened.source
        let frameCount = CGImageSourceGetCount(source)
        guard frameCount == 1 else {
            throw VisionImageError.unsupportedFrameCount(frameCount)
        }
        guard let typeIdentifier = CGImageSourceGetType(source) as String?,
              let sourceType = UTType(typeIdentifier),
              sourceType.conforms(to: .image) else {
            throw VisionImageError.invalidMetadata("unsupported image type")
        }
        // Checked against the sniffed type, not the caller's label: a data URL
        // announcing image/png can carry anything ImageIO recognises.
        guard limits.allowedTypeIdentifiers.contains(typeIdentifier) else {
            throw VisionImageError.invalidMetadata(
                "unsupported image type \(typeIdentifier)")
        }
        // A truncated JPEG decodes to gray filler rather than failing, so the
        // stream has to be checked structurally; see `hasCompleteJPEGStream`.
        if sourceType.conforms(to: .jpeg),
           !opened.hasCompleteJPEGStream() {
            throw VisionImageError.invalidMetadata(
                "JPEG scan is not terminated: the file is truncated")
        }
        // A truncated PNG decodes its missing rows as black.
        if sourceType.conforms(to: .png),
           !opened.hasCompletePNGStream() {
            throw VisionImageError.invalidMetadata(
                "PNG has no IEND chunk: the file is truncated")
        }
        guard let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) else {
            throw VisionImageError.invalidMetadata("missing frame properties")
        }
        let dictionary = properties as NSDictionary
        guard let width = (dictionary[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue,
              let height = (dictionary[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue,
              width > 0, height > 0 else {
            throw VisionImageError.invalidMetadata("missing positive pixel dimensions")
        }
        guard width <= limits.maximumSourceDimension,
              height <= limits.maximumSourceDimension else {
            throw VisionImageError.sideTooLarge(
                width: width, height: height, sideLimit: limits.maximumSourceDimension)
        }
        let (pixels, overflow) = width.multipliedReportingOverflow(by: height)
        guard !overflow, pixels <= limits.maximumSourcePixels else {
            throw VisionImageError.dimensionsTooLarge(
                width: width, height: height, pixelLimit: limits.maximumSourcePixels)
        }
        let depth = (dictionary[kCGImagePropertyDepth] as? NSNumber)?.intValue ?? 8
        // Four channels is the decode target; source depth drives how much
        // ImageIO materialises before the thumbnail conversion.
        let (depthBytes, depthOverflow) = pixels.multipliedReportingOverflow(
            by: max(1, depth / 8) * 4)
        // Every format decodes at native size (the reduction is our filter's),
        // so one ceiling covers them all.
        guard !depthOverflow, depthBytes <= limits.maximumDecodedBytes else {
            throw VisionImageError.decodedBytesTooLarge(
                width: width, height: height, byteLimit: limits.maximumDecodedBytes)
        }
        let orientation = (dictionary[kCGImagePropertyOrientation] as? NSNumber)?.intValue ?? 1
        guard (1...8).contains(orientation) else {
            throw VisionImageError.invalidMetadata("unsupported EXIF orientation \(orientation)")
        }
        let swapsAxes = (5...8).contains(orientation)
        return VisionImageMetadata(
            orientedWidth: swapsAxes ? height : width,
            orientedHeight: swapsAxes ? width : height)
    }
}
