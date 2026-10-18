import Foundation
import Metal

public enum PrefillOracleElement: String, Sendable {
    case fp16
    case uint32
}

public struct PrefillOracleSnapshot: @unchecked Sendable {
    public let boundary: String
    public let layer: Int
    public let position: Int
    public let element: PrefillOracleElement
    public let count: Int
    public let bytes: Data
}

/// Explicit diagnostic sink for the 64-vs-128 causal-prefill admission gate.
/// A nil sink is the production path and encodes no extra blits or waits.
public final class PrefillBoundaryOracleCollector: @unchecked Sendable {
    private struct Pending {
        let boundary: String
        let layer: Int
        let position: Int
        let element: PrefillOracleElement
        let count: Int
        let buffer: MTLBuffer
    }

    private var pending: [Pending] = []

    public init() {}

    func encode(commandBuffer: MTLCommandBuffer,
                boundary: String,
                layer: Int,
                position: Int,
                element: PrefillOracleElement,
                source: MTLBuffer,
                sourceOffset: Int,
                count: Int) throws {
        let stride = element == .fp16 ? MemoryLayout<UInt16>.stride : MemoryLayout<UInt32>.stride
        guard count >= 0, sourceOffset >= 0, sourceOffset + count * stride <= source.length,
              let destination = source.device.makeBuffer(length: max(1, count * stride),
                                                         options: .storageModeShared),
              let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw ModelError.residentBufferWrapFailed
        }
        blit.copy(from: source, sourceOffset: sourceOffset,
                  to: destination, destinationOffset: 0, size: count * stride)
        blit.endEncoding()
        pending.append(Pending(boundary: boundary, layer: layer, position: position,
                               element: element, count: count, buffer: destination))
    }

    public func snapshots() -> [PrefillOracleSnapshot] {
        pending.map { item in
            let stride = item.element == .fp16
                ? MemoryLayout<UInt16>.stride : MemoryLayout<UInt32>.stride
            return PrefillOracleSnapshot(
                boundary: item.boundary, layer: item.layer, position: item.position,
                element: item.element, count: item.count,
                bytes: Data(bytes: item.buffer.contents(), count: item.count * stride))
        }
    }
}
