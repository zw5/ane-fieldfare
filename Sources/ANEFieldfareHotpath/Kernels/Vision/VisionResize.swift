import Foundation
public import Metal

/// GPU separable bicubic resize, byte-identical to `TorchBicubicResize`: the
/// weights come from `TorchBicubicResize.weightTable`, so the two paths
/// cannot drift.
public final class VisionResize {
    private struct Params {
        var sourceWidth: UInt32
        var sourceHeight: UInt32
        var destinationWidth: UInt32
        var destinationHeight: UInt32
        var sourceRowBytes: UInt32
        var destinationRowBytes: UInt32
        var tapStride: UInt32
        var precision: Int32
    }

    private let context: MetalContext
    private let horizontal: any MTLComputePipelineState
    private let vertical: any MTLComputePipelineState

    public init(context: MetalContext) throws {
        self.context = context
        let library = try MetalContext.privateLibrary(
            device: context.device, module: "vision_resize")
        guard let horizontalFunction = library.makeFunction(name: "vision_resize_horizontal"),
              let verticalFunction = library.makeFunction(name: "vision_resize_vertical") else {
            throw MetalError.missingFunction("vision resize")
        }
        horizontal = try context.device.makeComputePipelineState(function: horizontalFunction)
        vertical = try context.device.makeComputePipelineState(function: verticalFunction)
    }

    /// Resizes RGBA8 pixels the caller has put in GPU-visible memory (so the
    /// full-resolution decode is not copied again). `destination` must hold
    /// `destinationHeight * destinationRowBytes` bytes.
    public func resize(
        sourceBuffer: any MTLBuffer,
        sourceWidth: Int,
        sourceHeight: Int,
        sourceRowBytes: Int,
        destination: UnsafeMutableRawPointer,
        destinationWidth: Int,
        destinationHeight: Int,
        destinationRowBytes: Int
    ) throws {
        let device = context.device
        let source = unsafe UnsafeRawPointer(sourceBuffer.contents())

        // Horizontal first, into an intermediate that keeps every source row:
        // the vertical pass then reads columns of the already-narrowed image,
        // which is the cheaper order whenever the width shrinks.
        let intermediateRowBytes = destinationWidth * 4
        let intermediate = try device.newBuffer(length: max(1, sourceHeight * intermediateRowBytes),
                                                options: .storageModeShared)
        let output = try device.newBuffer(length: max(1, destinationHeight * destinationRowBytes),
                                          options: .storageModeShared)
        let commandBuffer = try context.makeCommandBuffer()

        if sourceWidth != destinationWidth {
            try encode(commandBuffer: commandBuffer,
                       pipeline: horizontal,
                       table: TorchBicubicResize.weightTable(
                        input: sourceWidth, output: destinationWidth),
                       source: sourceBuffer,
                       destination: intermediate,
                       params: Params(
                        sourceWidth: UInt32(sourceWidth),
                        sourceHeight: UInt32(sourceHeight),
                        destinationWidth: UInt32(destinationWidth),
                        destinationHeight: UInt32(sourceHeight),
                        sourceRowBytes: UInt32(sourceRowBytes),
                        destinationRowBytes: UInt32(intermediateRowBytes),
                        tapStride: 0, precision: 0),
                       gridWidth: destinationWidth,
                       gridHeight: sourceHeight)
        } else {
            for row in 0..<sourceHeight {
                unsafe memcpy(intermediate.contents().advanced(by: row * intermediateRowBytes),
                       source.advanced(by: row * sourceRowBytes),
                       sourceWidth * 4)
            }
        }

        if sourceHeight != destinationHeight {
            try encode(commandBuffer: commandBuffer,
                       pipeline: vertical,
                       table: TorchBicubicResize.weightTable(
                        input: sourceHeight, output: destinationHeight),
                       source: intermediate,
                       destination: output,
                       params: Params(
                        sourceWidth: UInt32(destinationWidth),
                        sourceHeight: UInt32(sourceHeight),
                        destinationWidth: UInt32(destinationWidth),
                        destinationHeight: UInt32(destinationHeight),
                        sourceRowBytes: UInt32(intermediateRowBytes),
                        destinationRowBytes: UInt32(destinationRowBytes),
                        tapStride: 0, precision: 0),
                       gridWidth: destinationWidth * 4,
                       gridHeight: destinationHeight)
        }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        try checkCommandBufferError(commandBuffer)

        let finished = sourceHeight == destinationHeight ? intermediate : output
        let rowBytes = sourceHeight == destinationHeight
            ? intermediateRowBytes : destinationRowBytes
        for row in 0..<destinationHeight {
            unsafe memcpy(destination.advanced(by: row * destinationRowBytes),
                   finished.contents().advanced(by: row * rowBytes),
                   destinationWidth * 4)
        }
    }

    private func encode(
        commandBuffer: any MTLCommandBuffer,
        pipeline: any MTLComputePipelineState,
        table: TorchBicubicResize.WeightTable,
        source: any MTLBuffer,
        destination: any MTLBuffer,
        params: Params,
        gridWidth: Int,
        gridHeight: Int
    ) throws {
        let encoder = try commandBuffer.makeTracedComputeEncoder()
        var params = params
        params.tapStride = UInt32(table.tapStride)
        params.precision = Int32(table.precision)

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(source, offset: 0, index: 0)
        encoder.setBuffer(destination, offset: 0, index: 1)
        // Buffers, not `setBytes`: that path is capped at 4 KB and a 4x
        // reduction of a 4,000-pixel axis needs about 32 KB of taps.
        guard let starts = unsafe context.device.makeBuffer(
            bytes: table.starts,
            length: table.starts.count * MemoryLayout<Int32>.stride,
            options: .storageModeShared),
              let weights = unsafe context.device.makeBuffer(
                bytes: table.weights,
                length: table.weights.count * MemoryLayout<Int16>.stride,
                options: .storageModeShared) else {
            throw MetalError.allocationFailed("\(table.weights.count * MemoryLayout<Int16>.stride)-byte buffer")
        }
        encoder.setBuffer(starts, offset: 0, index: 2)
        encoder.setBuffer(weights, offset: 0, index: 3)
        encoder.setValue(params, index: 4)

        let width = min(pipeline.threadExecutionWidth, gridWidth)
        let height = max(1, min(pipeline.maxTotalThreadsPerThreadgroup / width, gridHeight))
        encoder.dispatchThreads(
            MTLSize(width: gridWidth, height: gridHeight, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: height, depth: 1))
        encoder.endEncoding()
    }
}
