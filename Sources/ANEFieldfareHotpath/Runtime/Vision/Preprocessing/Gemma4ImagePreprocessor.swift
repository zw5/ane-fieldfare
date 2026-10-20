import CoreGraphics
public import Foundation
import ImageIO
public import Metal

public final class Gemma4ImagePreprocessor {
    private static let unitBF16 = (0...255).map {
        Quantization.bf16Bits(Float($0) / 255)
    }

    private let device: any MTLDevice
    private let config: VisionConfig
    private let metadataReader: ImageMetadataReader
    /// Set when the reduction runs on the GPU. Building it needs a full
    /// `MetalContext`, so callers that only have a device keep the CPU filter.
    private let gpuResize: VisionResize?

    public init(
        device: any MTLDevice,
        config: VisionConfig = VisionConfig(),
        limits: VisionImageLimits = VisionImageLimits(),
        gpuResize: VisionResize? = nil
    ) {
        self.device = device
        self.config = config
        self.metadataReader = ImageMetadataReader(limits: limits)
        self.gpuResize = gpuResize
    }

    public func plan(fileURL: URL) throws -> VisionImagePlan {
        let opened = try VisionImageSource(fileURL: fileURL).open(
            maximumEncodedBytes: metadataReader.limits.maximumEncodedBytes)
        let metadata = try metadataReader.read(opened: opened)
        let geometry = try Gemma4ImageGeometry(
            sourceWidth: metadata.orientedWidth,
            sourceHeight: metadata.orientedHeight,
            config: config)
        return VisionImagePlan(metadata: metadata, geometry: geometry, opened: opened)
    }

    /// The size ImageIO is asked to decode at: the full oriented source. The
    /// whole reduction belongs to our bicubic filter, which matches the pinned
    /// reference; letting ImageIO's scaler do it changed the model's reading
    /// of a UI screenshot, dropping whole lines of text.
    static func decodeMaxPixelSize(metadata: VisionImageMetadata) -> Int {
        max(metadata.orientedWidth, metadata.orientedHeight)
    }

    public func preprocess(_ plan: VisionImagePlan) throws -> VisionPixelBuffer {
        let source = plan.opened.source
        let metadata = plan.metadata
        let geometry = plan.geometry
        let thumbnailOptions: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize:
                Self.decodeMaxPixelSize(metadata: metadata),
            kCGImageSourceShouldCacheImmediately: true,
            kCGImageSourceShouldAllowFloat: false,
        ]
        guard let decoded = CGImageSourceCreateThumbnailAtIndex(
            source, 0, thumbnailOptions as CFDictionary) else {
            throw VisionImageError.decodeFailed
        }

        let sourceRowBytes = try checkedMultiply(decoded.width, 4)
        let sourceBytes = try checkedMultiply(sourceRowBytes, decoded.height)
        // Decoded straight into GPU-visible memory when the GPU filter will
        // read it. ImageIO's full-resolution CGImage and this surface are live
        // at once during `draw` (~400 MB for a 48 MP source), which is what the
        // decoded-byte ceiling bounds.
        let sourceBuffer = gpuResize == nil
            ? nil
            : device.makeBuffer(length: sourceBytes, options: .storageModeShared)
        let sourceRGBA = unsafe sourceBuffer?.contents()
            ?? UnsafeMutableRawPointer.allocate(byteCount: sourceBytes, alignment: 64)
        unsafe sourceRGBA.initializeMemory(as: UInt8.self, repeating: 255, count: sourceBytes)
        defer { if sourceBuffer == nil { unsafe sourceRGBA.deallocate() } }
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let sourceDrawing = unsafe CGContext(
                data: sourceRGBA,
                width: decoded.width,
                height: decoded.height,
                bitsPerComponent: 8,
                bytesPerRow: sourceRowBytes,
                space: colorSpace,
                bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue
                    | CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw VisionImageError.allocationFailed
        }
        sourceDrawing.setFillColor(CGColor(gray: 1, alpha: 1))
        sourceDrawing.fill(CGRect(x: 0, y: 0, width: decoded.width, height: decoded.height))
        sourceDrawing.draw(
            decoded, in: CGRect(x: 0, y: 0, width: decoded.width, height: decoded.height))
        withExtendedLifetime(sourceDrawing) {}

        let rowBytes = try checkedMultiply(geometry.processedWidth, 4)
        let rgbaBytes = try checkedMultiply(rowBytes, geometry.processedHeight)
        let rgba = UnsafeMutableRawPointer.allocate(byteCount: rgbaBytes, alignment: 64)
        defer { unsafe rgba.deallocate() }
        // Same filter either way: the GPU path applies the weights the CPU
        // path builds, so this chooses only where the arithmetic runs.
        if let gpuResize, let sourceBuffer {
            try unsafe gpuResize.resize(
                sourceBuffer: sourceBuffer,
                sourceWidth: decoded.width,
                sourceHeight: decoded.height,
                sourceRowBytes: sourceRowBytes,
                destination: rgba,
                destinationWidth: geometry.processedWidth,
                destinationHeight: geometry.processedHeight,
                destinationRowBytes: rowBytes)
        } else {
            unsafe TorchBicubicResize.resize(
                source: sourceRGBA.assumingMemoryBound(to: UInt8.self),
                sourceWidth: decoded.width,
                sourceHeight: decoded.height,
                sourceRowBytes: sourceRowBytes,
                destination: rgba.assumingMemoryBound(to: UInt8.self),
                destinationWidth: geometry.processedWidth,
                destinationHeight: geometry.processedHeight,
                destinationRowBytes: rowBytes)
        }

        let patchElements = try checkedMultiply(geometry.patchCount, config.patchDimension)
        let patchBytes = try checkedMultiply(patchElements, MemoryLayout<UInt16>.stride)
        let positionElements = try checkedMultiply(geometry.patchCount, 2)
        let positionBytes = try checkedMultiply(
            positionElements, MemoryLayout<Int32>.stride)
        guard let patches = device.makeBuffer(length: patchBytes, options: .storageModeShared),
              let positions = device.makeBuffer(
                length: positionBytes, options: .storageModeShared) else {
            throw VisionImageError.allocationFailed
        }
        unsafe patchify(
            rgba: rgba.assumingMemoryBound(to: UInt8.self),
            rowBytes: rowBytes,
            geometry: geometry,
            patches: patches,
            positions: positions)
        return VisionPixelBuffer(
            patchesBF16: patches, positionsInt32x2: positions, geometry: geometry)
    }

    private func patchify(
        rgba: UnsafePointer<UInt8>,
        rowBytes: Int,
        geometry: Gemma4ImageGeometry,
        patches: any MTLBuffer,
        positions: any MTLBuffer
    ) {
        let patchPointer = unsafe patches.contents().bindMemory(
            to: UInt16.self, capacity: geometry.patchCount * config.patchDimension)
        let positionPointer = unsafe positions.contents().bindMemory(
            to: Int32.self, capacity: geometry.patchCount * 2)
        let size = config.patchSize
        for patchY in 0..<geometry.patchGridHeight {
            for patchX in 0..<geometry.patchGridWidth {
                let row = patchY * geometry.patchGridWidth + patchX
                unsafe positionPointer[row * 2] = Int32(patchX)
                unsafe positionPointer[row * 2 + 1] = Int32(patchY)
                var output = row * config.patchDimension
                for pixelY in 0..<size {
                    let inputRow = (patchY * size + pixelY) * rowBytes
                    for pixelX in 0..<size {
                        let input = inputRow + (patchX * size + pixelX) * 4
                        unsafe patchPointer[output] = unsafe Self.unitBF16[Int(rgba[input])]
                        unsafe patchPointer[output + 1] = unsafe Self.unitBF16[Int(rgba[input + 1])]
                        unsafe patchPointer[output + 2] = unsafe Self.unitBF16[Int(rgba[input + 2])]
                        output += 3
                    }
                }
            }
        }
    }

    private func checkedMultiply(_ lhs: Int, _ rhs: Int) throws -> Int {
        let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else {
            throw VisionImageError.invalidMetadata("buffer size overflow")
        }
        return value
    }
}
