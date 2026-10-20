package import Foundation

package struct GTurboManifestFileV1: Codable, Equatable, Sendable {
    package let size: UInt64

    package init(size: UInt64) {
        self.size = size
    }
}

package struct GTurboManifestArchV1: Codable, Equatable, Sendable {
    package let hiddenSize: Int
    package let ffnIntermediate: Int
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
    package let fullAttentionLayerMask: [Int]

    package init(hiddenSize: Int, ffnIntermediate: Int, moeIntermediateSize: Int,
                 numHeads: Int, numKVHeads: Int, numFullKVHeads: Int,
                 headDim: Int, fullHeadDim: Int, vocabSize: Int,
                 slidingWindow: Int, finalLogitSoftcap: Double,
                 ropeTheta: Double, fullRopeTheta: Double,
                 partialRotaryFactor: Double, numLayers: Int, numExperts: Int,
                 topKExperts: Int, tieWordEmbeddings: Bool, attentionKEqV: Bool,
                 hiddenActivation: String, fullAttentionLayerMask: [Int]) {
        self.hiddenSize = hiddenSize
        self.ffnIntermediate = ffnIntermediate
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
        self.fullAttentionLayerMask = fullAttentionLayerMask
    }
}

package struct GTurboManifestQuantSlotV1: Codable, Equatable, Sendable {
    package let weightBits: Int
    package let scheme: String
    package let scaleType: String
    package let biasType: String
    package let groupSize: Int

    package init(weightBits: Int, scheme: String, scaleType: String,
                 biasType: String, groupSize: Int) {
        self.weightBits = weightBits
        self.scheme = scheme
        self.scaleType = scaleType
        self.biasType = biasType
        self.groupSize = groupSize
    }
}

package struct GTurboManifestQuantV1: Codable, Equatable, Sendable {
    package let embedding: GTurboManifestQuantSlotV1
    package let attention: GTurboManifestQuantSlotV1
    package let router: GTurboManifestQuantSlotV1
    package let sharedExpert: GTurboManifestQuantSlotV1
    package let routedExpert: GTurboManifestQuantSlotV1

    package init(embedding: GTurboManifestQuantSlotV1,
                 attention: GTurboManifestQuantSlotV1,
                 router: GTurboManifestQuantSlotV1,
                 sharedExpert: GTurboManifestQuantSlotV1,
                 routedExpert: GTurboManifestQuantSlotV1) {
        self.embedding = embedding
        self.attention = attention
        self.router = router
        self.sharedExpert = sharedExpert
        self.routedExpert = routedExpert
    }
}

package struct GTurboManifestV1: Codable, Equatable, Sendable {
    package let magic: String
    package let versionMajor: Int
    package let versionMinor: Int
    package let flags: [String: Bool]
    package let modelID: String
    package let sourceSnapshotHash: String?
    package let arch: GTurboManifestArchV1
    package let quant: GTurboManifestQuantV1?
    package let files: [String: GTurboManifestFileV1]
    package let expertsPerLayer: Int
    package let numLayers: Int
    package let expertStride: UInt64

    package init(magic: String = GTurboFormatV1.magic,
                 versionMajor: Int = GTurboFormatV1.versionMajor,
                 versionMinor: Int = GTurboFormatV1.versionMinor,
                 flags: [String: Bool], modelID: String,
                 sourceSnapshotHash: String?, arch: GTurboManifestArchV1,
                 quant: GTurboManifestQuantV1?,
                 files: [String: GTurboManifestFileV1],
                 expertsPerLayer: Int, numLayers: Int, expertStride: UInt64) {
        self.magic = magic
        self.versionMajor = versionMajor
        self.versionMinor = versionMinor
        self.flags = flags
        self.modelID = modelID
        self.sourceSnapshotHash = sourceSnapshotHash
        self.arch = arch
        self.quant = quant
        self.files = files
        self.expertsPerLayer = expertsPerLayer
        self.numLayers = numLayers
        self.expertStride = expertStride
    }
}

package enum GTurboManifestCodec {
    package static func decode(_ data: Data) throws -> GTurboManifestV1 {
        do { return try JSONDecoder().decode(GTurboManifestV1.self, from: data) }
        catch { throw GTurboFormatError.invalid(field: "manifest.json", reason: "\(error)") }
    }

    /// The structure a reader relies on: the streaming metadata agrees with
    /// the architecture, the expert stride is set, and every file path is a
    /// safe relative path, unique on an Apple filesystem. Magic, version,
    /// flags and stride alignment are `ManifestReader`'s, which reports them
    /// with their own errors and compares the architecture and quantization
    /// with the expected model exactly.
    package static func validate(_ manifest: GTurboManifestV1) throws {
        guard manifest.expertStride > 0,
              manifest.arch.numLayers == manifest.numLayers,
              manifest.arch.numExperts == manifest.expertsPerLayer else {
            throw GTurboFormatError.invalid(
                field: "manifest", reason: "streaming metadata disagrees with the architecture")
        }
        let reservedFiles: Set<String> = ["manifest.json"]
        var canonicalPaths: [String: String] = [:]
        for path in manifest.files.keys.sorted() {
            try GTurboPathValidator.validateRelativePath(path, field: "manifest.files.\(path)")
            let key = GTurboPathValidator.appleFilesystemKey(path)
            guard canonicalPaths.updateValue(path, forKey: key) == nil else {
                throw GTurboFormatError.invalid(
                    field: "manifest.files.\(path)", reason: "filesystem-equivalent duplicate path")
            }
            guard key != "tokenizer",
                  !reservedFiles.contains(key),
                  !reservedFiles.contains(where: { key.hasPrefix("\($0)/") }) else {
                throw GTurboFormatError.invalid(
                    field: "manifest.files.\(path)", reason: "reserved artifact filename")
            }
        }
        for (key, path) in canonicalPaths {
            var components = key.split(separator: "/").map(String.init)
            while components.count > 1 {
                _ = components.removeLast()
                let ancestor = components.joined(separator: "/")
                if canonicalPaths[ancestor] != nil {
                    throw GTurboFormatError.invalid(
                        field: "manifest.files.\(path)",
                        reason: "file path collides with a directory prefix")
                }
            }
        }
    }
}
