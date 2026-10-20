import Darwin
import Foundation
import Metal
import Synchronization
import ANEFieldfareModelFormat

/// Per-layer routed-expert mappings, opened on first touch.
///
/// Opening a layer checks its file size against the manifest, so it runs
/// inside the table's lock: two callers racing on one layer open it once, and a release
/// cannot interleave with an open.
final class RoutedExpertLayers: Sendable {
    private let table: Mutex<[MappedExpertLayer?]>
    private let device: any MTLDevice
    private let experts: FieldfareExperts
    private let files: [String: FieldfareFile]
    private let modelDirectory: ModelDirectory

    init(device: any MTLDevice,
         experts: FieldfareExperts,
         files: [String: FieldfareFile],
         modelDirectory: ModelDirectory) {
        self.table = Mutex(Array(repeating: nil, count: experts.files.count))
        self.device = device
        self.experts = experts
        self.files = files
        self.modelDirectory = modelDirectory
    }

    /// The layer's mapping, opening its file on first touch.
    func layer(_ layer: Int) throws -> MappedExpertLayer {
        try table.withLock { layers in
            if let mapped = layers[layer] {
                return mapped
            }
            let mapped = try open(layer: layer)
            layers[layer] = mapped
            return mapped
        }
    }

    var openLayerCount: Int {
        table.withLock { $0.count { $0 != nil } }
    }

    /// Unmaps every open layer once its buffers are released and returns how
    /// many were open.
    func release() -> Int {
        table.withLock { layers in
            let released = layers.count { $0 != nil }
            layers = Array(repeating: nil, count: layers.count)
            return released
        }
    }

    private func open(layer: Int) throws -> MappedExpertLayer {
        let path = experts.files[layer]
        let layerFD = try modelDirectory.openFile(path)
        defer { close(layerFD) }
        // A short file would fault when a mapped expert past its end is read.
        guard let entry = files[path] else {
            throw ModelError.missingFile(name: path)
        }
        let actualSize = try modelDirectory.fileSize(fileDescriptor: layerFD, relativePath: path)
        guard actualSize == entry.size else {
            throw ModelError.tensorSizeMismatch(name: path, expected: entry.size, actual: actualSize)
        }
        return try MappedExpertLayer(
            fileDescriptor: layerFD,
            path: path,
            device: device,
            expertCount: experts.perLayer,
            expertBytes: Int(experts.stride),
            expertOffsets: (0..<UInt64(experts.perLayer)).map { $0 * experts.stride })
    }
}
