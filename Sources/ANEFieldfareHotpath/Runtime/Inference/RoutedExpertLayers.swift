import Darwin
import Foundation
import Metal
import Synchronization
import ANEFieldfareImportedFormat

/// Per-layer routed-expert mappings, opened on first touch.
///
/// Opening a layer checks its file size against the manifest, so it runs
/// inside the table's lock: two callers racing on one layer open it once, and a release
/// cannot interleave with an open.
final class RoutedExpertLayers: Sendable {
    private let table: Mutex<[MappedExpertLayer?]>
    private let device: any MTLDevice
    private let layout: PackedExpertsLayout
    private let manifest: Manifest
    private let directoryURL: URL
    private let modelDirectory: GTurboModelDirectory

    init(device: any MTLDevice,
         layout: PackedExpertsLayout,
         manifest: Manifest,
         directoryURL: URL,
         modelDirectory: GTurboModelDirectory) {
        self.table = Mutex(Array(repeating: nil, count: layout.numLayers))
        self.device = device
        self.layout = layout
        self.manifest = manifest
        self.directoryURL = directoryURL
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
        let basename = layout.layers[layer].file
        let url = directoryURL
            .appendingPathComponent("packed_experts")
            .appendingPathComponent(basename)
        let manifestRel = "packed_experts/\(basename)"
        let layerFD = try modelDirectory.openFile(manifestRel)
        defer { close(layerFD) }
        // A short file would fault when a mapped expert past its end is read.
        guard let entry = manifest.files[manifestRel] else {
            throw ModelError.missingFile(name: manifestRel)
        }
        let actualSize = try modelDirectory.fileSize(
            fileDescriptor: layerFD, relativePath: manifestRel)
        guard actualSize == entry.size else {
            throw ModelError.tensorSizeMismatch(
                name: manifestRel, expected: entry.size, actual: actualSize)
        }
        return try MappedExpertLayer(
            fileDescriptor: layerFD,
            path: url.path,
            device: device,
            expertCount: layout.expertsPerLayer,
            expertBytes: Int(layout.expertStride),
            expertOffsets: layout.layers[layer].experts.map(\.offset))
    }
}
