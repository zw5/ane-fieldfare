package import Foundation

package enum GTurboVisionFormatV1 {
    package static let magic = "GTURBO-VISION"
    package static let artifactKind = "gemma4_vision_companion"
    package static let versionMajor = 1
    package static let versionMinor = 0
    package static let alignmentBytes: UInt64 = 16_384
    package static let metadataMaxBytes: UInt64 = 4 * 1024 * 1024
    package static let weightsFile = "vision_weights.bin"
    package static let processorFile = "processor_config.json"
    package static let manifestFile = "manifest.json"
}

package enum GTurboVisionDTypeV1: String, Codable, Equatable, Sendable {
    case u32
    case bf16

    fileprivate var elementBytes: UInt64 {
        switch self {
        case .u32: 4
        case .bf16: 2
        }
    }
}

package struct GTurboVisionQuantizationV1: Codable, Equatable, Sendable {
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

package struct GTurboVisionTensorRegionV1: Codable, Equatable, Sendable {
    package let name: String
    package let executionPosition: Int
    package let file: String
    package let offset: UInt64
    package let size: UInt64
    package let shape: [UInt64]
    package let dtype: GTurboVisionDTypeV1
    package let quantization: GTurboVisionQuantizationV1?

    package init(name: String, executionPosition: Int,
                 file: String = GTurboVisionFormatV1.weightsFile,
                 offset: UInt64, size: UInt64, shape: [UInt64],
                 dtype: GTurboVisionDTypeV1,
                 quantization: GTurboVisionQuantizationV1? = nil) {
        self.name = name
        self.executionPosition = executionPosition
        self.file = file
        self.offset = offset
        self.size = size
        self.shape = shape
        self.dtype = dtype
        self.quantization = quantization
    }
}

package struct GTurboVisionManifestV1: Codable, Equatable, Sendable {
    package let magic: String
    package let artifactKind: String
    package let versionMajor: Int
    package let versionMinor: Int
    package let modelID: String
    package let sourceRevision: String
    package let compatibleTextSourceSnapshotHash: String
    package let files: [String: GTurboManifestFileV1]
    package let tensors: [GTurboVisionTensorRegionV1]

    package init(magic: String = GTurboVisionFormatV1.magic,
                 artifactKind: String = GTurboVisionFormatV1.artifactKind,
                 versionMajor: Int = GTurboVisionFormatV1.versionMajor,
                 versionMinor: Int = GTurboVisionFormatV1.versionMinor,
                 modelID: String, sourceRevision: String,
                 compatibleTextSourceSnapshotHash: String,
                 files: [String: GTurboManifestFileV1],
                 tensors: [GTurboVisionTensorRegionV1]) {
        self.magic = magic
        self.artifactKind = artifactKind
        self.versionMajor = versionMajor
        self.versionMinor = versionMinor
        self.modelID = modelID
        self.sourceRevision = sourceRevision
        self.compatibleTextSourceSnapshotHash = compatibleTextSourceSnapshotHash
        self.files = files
        self.tensors = tensors
    }
}

package enum GTurboVisionManifestCodec {
    package static func decode(_ data: Data) throws -> GTurboVisionManifestV1 {
        guard UInt64(data.count) <= GTurboVisionFormatV1.metadataMaxBytes else {
            throw GTurboFormatError.invalid(
                field: GTurboVisionFormatV1.manifestFile, reason: "metadata exceeds 4 MiB")
        }
        let manifest: GTurboVisionManifestV1
        do { manifest = try JSONDecoder().decode(GTurboVisionManifestV1.self, from: data) }
        catch {
            throw GTurboFormatError.invalid(
                field: GTurboVisionFormatV1.manifestFile, reason: "\(error)")
        }
        try GTurboVisionStructuralValidator.validate(manifest)
        return manifest
    }

    package static func encode(_ manifest: GTurboVisionManifestV1) throws -> Data {
        try GTurboVisionStructuralValidator.validate(manifest)
        let encoder = JSONEncoder()
        do {
            let object = try JSONSerialization.jsonObject(with: encoder.encode(manifest))
            let data = try JSONSerialization.data(
                withJSONObject: object,
                options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
            guard UInt64(data.count) <= GTurboVisionFormatV1.metadataMaxBytes else {
                throw GTurboFormatError.invalid(
                    field: GTurboVisionFormatV1.manifestFile, reason: "metadata exceeds 4 MiB")
            }
            return data
        } catch let error as GTurboFormatError {
            throw error
        } catch {
            throw GTurboFormatError.invalid(
                field: GTurboVisionFormatV1.manifestFile, reason: "\(error)")
        }
    }
}

package enum GTurboVisionStructuralValidator {
    package static func validate(_ manifest: GTurboVisionManifestV1) throws {
        guard manifest.magic == GTurboVisionFormatV1.magic,
              manifest.artifactKind == GTurboVisionFormatV1.artifactKind else {
            throw GTurboFormatError.invalid(field: "vision.manifest.identity",
                                             reason: "unsupported artifact")
        }
        guard manifest.versionMajor == GTurboVisionFormatV1.versionMajor,
              manifest.versionMinor == GTurboVisionFormatV1.versionMinor else {
            throw GTurboFormatError.invalid(field: "vision.manifest.version",
                                             reason: "unsupported version")
        }
        guard !manifest.modelID.isEmpty, !manifest.sourceRevision.isEmpty,
              !manifest.compatibleTextSourceSnapshotHash.isEmpty else {
            throw GTurboFormatError.invalid(field: "vision.manifest",
                                             reason: "missing source identity")
        }

        let expectedFiles = Set([
            GTurboVisionFormatV1.weightsFile,
            GTurboVisionFormatV1.processorFile,
        ])
        guard Set(manifest.files.keys) == expectedFiles else {
            throw GTurboFormatError.invalid(field: "vision.manifest.files",
                                             reason: "expected exactly two payload files")
        }
        var filesystemKeys = Set<String>()
        for path in manifest.files.keys {
            try GTurboPathValidator.validateBasename(path, field: "vision.manifest.files.\(path)")
            guard filesystemKeys.insert(GTurboPathValidator.appleFilesystemKey(path)).inserted else {
                throw GTurboFormatError.invalid(field: "vision.manifest.files.\(path)",
                                                 reason: "filesystem-equivalent duplicate path")
            }
        }

        guard !manifest.tensors.isEmpty else {
            throw GTurboFormatError.invalid(field: "vision.manifest.tensors",
                                             reason: "expected at least one tensor")
        }
        var names = Set<String>()
        var previousEnd: UInt64 = 0
        guard let weightsSize = manifest.files[GTurboVisionFormatV1.weightsFile]?.size else {
            throw GTurboFormatError.invalid(field: "vision.manifest.files",
                                             reason: "missing \(GTurboVisionFormatV1.weightsFile)")
        }
        for (index, tensor) in manifest.tensors.enumerated() {
            guard !tensor.name.isEmpty, !tensor.name.contains("\0"),
                  names.insert(tensor.name).inserted else {
                throw GTurboFormatError.invalid(field: "vision.tensor.name",
                                                 reason: "empty or duplicate tensor name")
            }
            guard tensor.executionPosition == index else {
                throw GTurboFormatError.invalid(field: "vision.tensor.executionPosition",
                                                 reason: "regions are reordered or duplicated")
            }
            guard tensor.file == GTurboVisionFormatV1.weightsFile else {
                throw GTurboFormatError.invalid(field: "vision.tensor.file",
                                                 reason: "tensor must use vision_weights.bin")
            }
            guard tensor.offset % GTurboVisionFormatV1.alignmentBytes == 0,
                  tensor.offset >= previousEnd else {
                throw GTurboFormatError.invalid(field: "vision.tensor.offset",
                                                 reason: "misaligned or overlapping region")
            }
            guard !tensor.shape.isEmpty, tensor.shape.allSatisfy({ $0 > 0 }) else {
                throw GTurboFormatError.invalid(field: "vision.tensor.shape",
                                                 reason: "invalid shape")
            }
            var elements: UInt64 = 1
            for dimension in tensor.shape {
                elements = try gturboCheckedMultiply(
                    elements, dimension, field: "vision.tensor.shape")
            }
            let expectedSize = try gturboCheckedMultiply(
                elements, tensor.dtype.elementBytes, field: "vision.tensor.size")
            guard tensor.size == expectedSize else {
                throw GTurboFormatError.invalid(field: "vision.tensor.size",
                                                 reason: "size does not match storage shape")
            }
            switch (tensor.dtype, tensor.quantization) {
            case (.bf16, nil):
                break
            case let (.u32, quantization?):
                guard quantization.weightBits == 4,
                      quantization.scheme == "affine",
                      quantization.scaleType == "bf16",
                      quantization.biasType == "bf16",
                      quantization.groupSize == 64 else {
                    throw GTurboFormatError.invalid(field: "vision.tensor.quantization",
                                                     reason: "unsupported quantization")
                }
            default:
                throw GTurboFormatError.invalid(field: "vision.tensor.quantization",
                                                 reason: "dtype and quantization disagree")
            }
            let end = try gturboCheckedAdd(tensor.offset, tensor.size,
                                           field: "vision.tensor.range")
            guard end <= weightsSize else {
                throw GTurboFormatError.invalid(field: "vision.tensor.range",
                                                 reason: "region exceeds vision_weights.bin")
            }
            previousEnd = end
        }
    }
}
