import Darwin
import Foundation
import Metal
import Synchronization
import ANEFieldfareImportedFormat

/// Per-layer routed-expert mappings, opened on first touch.
///
/// Opening a layer checks its file against the manifest, so it runs inside the
/// table's lock: two callers racing on one layer open it once, and a release
/// cannot interleave with an open.
final class RoutedExpertLayers: Sendable {
    private struct Table: Sendable {
        var layers: [MappedExpertLayer?]
        /// Survives `release`, so reopening a layer skips its SHA-256 pass.
        var verified: [Bool]
    }

    private let table: Mutex<Table>
    private let device: any MTLDevice
    private let layout: PackedExpertsLayout
    private let manifest: Manifest
    private let directoryURL: URL
    private let modelDirectory: GTurboModelDirectory
    private let integrityPolicy: ModelIntegrityPolicy

    init(device: any MTLDevice,
         layout: PackedExpertsLayout,
         manifest: Manifest,
         directoryURL: URL,
         modelDirectory: GTurboModelDirectory,
         integrityPolicy: ModelIntegrityPolicy) {
        self.table = Mutex(Table(
            layers: Array(repeating: nil, count: layout.numLayers),
            verified: Array(repeating: false, count: layout.numLayers)))
        self.device = device
        self.layout = layout
        self.manifest = manifest
        self.directoryURL = directoryURL
        self.modelDirectory = modelDirectory
        self.integrityPolicy = integrityPolicy
    }

    /// The layer's mapping, opening and verifying its file on first touch.
    func layer(_ layer: Int) throws -> MappedExpertLayer {
        try table.withLock { table in
            if let mapped = table.layers[layer] {
                return mapped
            }
            let mapped = try open(layer: layer, verified: table.verified[layer])
            table.layers[layer] = mapped
            table.verified[layer] = true
            return mapped
        }
    }

    var openLayerCount: Int {
        table.withLock { $0.layers.count { $0 != nil } }
    }

    /// Unmaps every open layer once its buffers are released and returns how
    /// many were open. Verification state is kept.
    func release() -> Int {
        table.withLock { table in
            let released = table.layers.count { $0 != nil }
            table.layers = Array(repeating: nil, count: table.layers.count)
            return released
        }
    }

    private func open(layer: Int, verified: Bool) throws -> MappedExpertLayer {
        let basename = layout.layers[layer].file
        let url = directoryURL
            .appendingPathComponent("packed_experts")
            .appendingPathComponent(basename)
        let manifestRel = "packed_experts/\(basename)"
        let layerFD = try modelDirectory.openFile(manifestRel)
        defer { close(layerFD) }
        if !verified {
            guard let entry = manifest.files[manifestRel] else {
                throw ModelError.missingFile(name: manifestRel)
            }
            let actualSize = try modelDirectory.fileSize(
                fileDescriptor: layerFD, relativePath: manifestRel)
            guard actualSize == entry.size else {
                throw ModelError.tensorSizeMismatch(
                    name: manifestRel, expected: entry.size, actual: actualSize)
            }
            switch integrityPolicy {
            case .fullSha256:
                try Sha256Verifier.verifyFile(fileDescriptor: layerFD,
                                              named: manifestRel,
                                              expectedHex: entry.sha256)
            case .sizeCheckTrustedReceipt:
                break
            }
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
