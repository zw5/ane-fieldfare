package import Foundation

/// A byte range inside one of the model's files.
package struct FieldfareRegion: Codable, Equatable, Sendable {
    package let offset: UInt64
    package let size: UInt64

    package init(offset: UInt64, size: UInt64) {
        self.offset = offset
        self.size = size
    }
}

package struct FieldfareTensor: Codable, Equatable, Sendable {
    package let name: String
    package let encoding: FieldfareEncoding
    /// Logical shape; quantized tensors are `[rows, cols]`.
    package let shape: [UInt64]
    package let weights: FieldfareRegion
    package let scales: FieldfareRegion?
    package let biases: FieldfareRegion?

    package init(name: String, encoding: FieldfareEncoding, shape: [UInt64],
                 weights: FieldfareRegion, scales: FieldfareRegion? = nil,
                 biases: FieldfareRegion? = nil) {
        self.name = name
        self.encoding = encoding
        self.shape = shape
        self.weights = weights
        self.scales = scales
        self.biases = biases
    }

    /// The exact region sizes the encoding requires for this shape.
    package static func regionSizes(encoding: FieldfareEncoding, shape: [UInt64],
                                    field: String) throws
        -> (weights: UInt64, scales: UInt64, biases: UInt64) {
        guard !shape.isEmpty, shape.allSatisfy({ $0 > 0 }) else {
            throw FieldfareFormatError.invalid(field: field, reason: "empty shape")
        }
        switch encoding {
        case .bf16:
            var count: UInt64 = 1
            for dim in shape { count = try fieldfareMultiply(count, dim, field: field) }
            return (try fieldfareMultiply(count, 2, field: field), 0, 0)
        case .q4_0, .affine8:
            guard shape.count == 2, let group = encoding.groupSize,
                  shape[1] % group == 0 else {
                throw FieldfareFormatError.invalid(
                    field: field, reason: "\(encoding) needs [rows, cols] with cols % \(encoding.groupSize ?? 0) == 0")
            }
            let count = try fieldfareMultiply(shape[0], shape[1], field: field)
            let groups = try fieldfareMultiply(shape[0], shape[1] / group, field: field)
            let groupBytes = try fieldfareMultiply(groups, 2, field: field)
            return encoding == .q4_0
                ? (count / 2, groupBytes, 0)
                : (count, groupBytes, groupBytes)
        }
    }

    /// Every region present, with the sizes its encoding requires.
    package func validatedRegions(field: String) throws -> [FieldfareRegion] {
        let sizes = try Self.regionSizes(encoding: encoding, shape: shape, field: field)
        func check(_ region: FieldfareRegion?, _ size: UInt64, _ part: String) throws -> FieldfareRegion? {
            if size == 0 {
                guard region == nil else {
                    throw FieldfareFormatError.invalid(field: "\(field).\(part)", reason: "not used by \(encoding)")
                }
                return nil
            }
            guard let region, region.size == size else {
                throw FieldfareFormatError.invalid(
                    field: "\(field).\(part)", reason: "\(encoding) \(shape) needs \(size) bytes")
            }
            return region
        }
        return [try check(weights, sizes.weights, "weights"),
                try check(scales, sizes.scales, "scales"),
                try check(biases, sizes.biases, "biases")].compactMap { $0 }
    }
}

/// Where the weights came from: a pinned repository revision and the exact
/// files, each verified against its SHA-256 before the install committed.
package struct FieldfareSource: Codable, Equatable, Sendable {
    package struct File: Codable, Equatable, Sendable {
        package let repo: String
        package let revision: String
        package let path: String
        package let size: UInt64
        package let sha256: String

        package init(repo: String, revision: String, path: String, size: UInt64, sha256: String) {
            self.repo = repo
            self.revision = revision
            self.path = path
            self.size = size
            self.sha256 = sha256
        }
    }

    package let files: [File]

    package init(files: [File]) {
        self.files = files
    }
}

package struct FieldfareFile: Codable, Equatable, Sendable {
    package let size: UInt64
    package let sha256: String

    package init(size: UInt64, sha256: String) {
        self.size = size
        self.sha256 = sha256
    }
}

package struct FieldfareArch: Codable, Equatable, Sendable {
    package let hiddenSize: Int
    package let intermediateSize: Int
    package let moeIntermediateSize: Int
    package let numHeads: Int
    package let numKVHeads: Int
    package let numFullKVHeads: Int
    package let headDim: Int
    package let fullHeadDim: Int
    package let vocabSize: Int
    package let slidingWindow: Int
    package let finalLogitSoftcap: Double
    package let ropeTheta: Double
    package let fullRopeTheta: Double
    package let partialRotaryFactor: Double
    package let numLayers: Int
    package let numExperts: Int
    package let topKExperts: Int
    package let tieWordEmbeddings: Bool
    package let attentionKEqV: Bool
    package let hiddenActivation: String
    package let fullAttentionLayers: [Int]

    package init(hiddenSize: Int, intermediateSize: Int, moeIntermediateSize: Int,
                 numHeads: Int, numKVHeads: Int, numFullKVHeads: Int,
                 headDim: Int, fullHeadDim: Int, vocabSize: Int,
                 slidingWindow: Int, finalLogitSoftcap: Double,
                 ropeTheta: Double, fullRopeTheta: Double,
                 partialRotaryFactor: Double, numLayers: Int, numExperts: Int,
                 topKExperts: Int, tieWordEmbeddings: Bool, attentionKEqV: Bool,
                 hiddenActivation: String, fullAttentionLayers: [Int]) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.moeIntermediateSize = moeIntermediateSize
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.numFullKVHeads = numFullKVHeads
        self.headDim = headDim
        self.fullHeadDim = fullHeadDim
        self.vocabSize = vocabSize
        self.slidingWindow = slidingWindow
        self.finalLogitSoftcap = finalLogitSoftcap
        self.ropeTheta = ropeTheta
        self.fullRopeTheta = fullRopeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.numLayers = numLayers
        self.numExperts = numExperts
        self.topKExperts = topKExperts
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionKEqV = attentionKEqV
        self.hiddenActivation = hiddenActivation
        self.fullAttentionLayers = fullAttentionLayers
    }
}

/// Tensors packed into one file.
package struct FieldfareTensorFile: Codable, Equatable, Sendable {
    package let file: String
    package let tensors: [FieldfareTensor]

    package init(file: String, tensors: [FieldfareTensor]) {
        self.file = file
        self.tensors = tensors
    }
}

/// Routed experts: `files[layer]` holds `perLayer` blobs of `stride` bytes,
/// expert `e` at `e * stride`. Every blob has the same tensor table, with
/// offsets relative to the blob.
package struct FieldfareExperts: Codable, Equatable, Sendable {
    package let stride: UInt64
    package let perLayer: Int
    package let files: [String]
    package let tensors: [FieldfareTensor]

    package init(stride: UInt64, perLayer: Int, files: [String], tensors: [FieldfareTensor]) {
        self.stride = stride
        self.perLayer = perLayer
        self.files = files
        self.tensors = tensors
    }
}

package struct FieldfareManifest: Codable, Equatable, Sendable {
    package let format: String
    package let version: Int
    package let modelID: String
    package let source: FieldfareSource
    package let arch: FieldfareArch
    /// Every file in the model directory except this manifest.
    package let files: [String: FieldfareFile]
    package let resident: FieldfareTensorFile
    package let experts: FieldfareExperts
    package let vision: FieldfareTensorFile?

    package init(modelID: String, source: FieldfareSource, arch: FieldfareArch,
                 files: [String: FieldfareFile], resident: FieldfareTensorFile,
                 experts: FieldfareExperts, vision: FieldfareTensorFile?) {
        self.format = FieldfareFormat.name
        self.version = FieldfareFormat.version
        self.modelID = modelID
        self.source = source
        self.arch = arch
        self.files = files
        self.resident = resident
        self.experts = experts
        self.vision = vision
    }
}

package enum FieldfareManifestCodec {
    package static func decode(_ data: Data) throws -> FieldfareManifest {
        let manifest: FieldfareManifest
        do {
            manifest = try JSONDecoder().decode(FieldfareManifest.self, from: data)
        } catch {
            throw FieldfareFormatError.invalid(field: "manifest", reason: "\(error)")
        }
        try validate(manifest)
        return manifest
    }

    package static func encode(_ manifest: FieldfareManifest) throws -> Data {
        try validate(manifest)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(manifest)
    }

    package static func validate(_ manifest: FieldfareManifest) throws {
        guard manifest.format == FieldfareFormat.name else {
            throw FieldfareFormatError.invalid(field: "format", reason: "expected \(FieldfareFormat.name)")
        }
        guard manifest.version == FieldfareFormat.version else {
            throw FieldfareFormatError.invalid(field: "version", reason: "unsupported \(manifest.version)")
        }
        for path in manifest.files.keys {
            try FieldfarePath.validate(path, field: "files.\(path)")
            guard path != FieldfareFormat.manifestPath else {
                throw FieldfareFormatError.invalid(field: "files", reason: "lists the manifest")
            }
        }
        func size(of path: String, field: String) throws -> UInt64 {
            guard let file = manifest.files[path] else {
                throw FieldfareFormatError.invalid(field: field, reason: "\(path) is not in files")
            }
            return file.size
        }

        try validateTensorFile(manifest.resident, fileSize: size(of: manifest.resident.file, field: "resident"),
                               field: "resident")
        if let vision = manifest.vision {
            try validateTensorFile(vision, fileSize: size(of: vision.file, field: "vision"), field: "vision")
        }

        let experts = manifest.experts
        guard experts.stride > 0, experts.stride % FieldfareFormat.expertAlignment == 0 else {
            throw FieldfareFormatError.invalid(field: "experts.stride", reason: "not a multiple of 16 KiB")
        }
        guard experts.perLayer == manifest.arch.numExperts,
              experts.files.count == manifest.arch.numLayers else {
            throw FieldfareFormatError.invalid(field: "experts", reason: "does not match arch")
        }
        let layerBytes = try fieldfareMultiply(UInt64(experts.perLayer), experts.stride, field: "experts")
        for (layer, path) in experts.files.enumerated() {
            guard try size(of: path, field: "experts.files[\(layer)]") == layerBytes else {
                throw FieldfareFormatError.invalid(field: "experts.files[\(layer)]",
                                                   reason: "size is not perLayer * stride")
            }
        }
        guard Set(experts.files).count == experts.files.count else {
            throw FieldfareFormatError.invalid(field: "experts.files", reason: "duplicate file")
        }
        try validateTensors(experts.tensors, limit: experts.stride, field: "experts")
    }

    private static func validateTensorFile(_ tensorFile: FieldfareTensorFile, fileSize: UInt64,
                                           field: String) throws {
        try validateTensors(tensorFile.tensors, limit: fileSize, field: field)
    }

    /// Unique names; every region the encoding needs, aligned, inside
    /// `[0, limit)`, and no two regions overlapping.
    private static func validateTensors(_ tensors: [FieldfareTensor], limit: UInt64,
                                        field: String) throws {
        guard Set(tensors.map(\.name)).count == tensors.count else {
            throw FieldfareFormatError.invalid(field: field, reason: "duplicate tensor name")
        }
        var regions: [(FieldfareRegion, String)] = []
        for tensor in tensors {
            let tensorField = "\(field).\(tensor.name)"
            for region in try tensor.validatedRegions(field: tensorField) {
                guard region.offset % FieldfareFormat.regionAlignment == 0 else {
                    throw FieldfareFormatError.invalid(field: tensorField, reason: "region not 64-byte aligned")
                }
                let end = try fieldfareAdd(region.offset, region.size, field: tensorField)
                guard end <= limit else {
                    throw FieldfareFormatError.invalid(field: tensorField, reason: "region past end of file")
                }
                regions.append((region, tensorField))
            }
        }
        regions.sort { $0.0.offset < $1.0.offset }
        for (previous, next) in zip(regions, regions.dropFirst())
        where previous.0.offset + previous.0.size > next.0.offset {
            throw FieldfareFormatError.invalid(field: next.1, reason: "overlaps \(previous.1)")
        }
    }
}
