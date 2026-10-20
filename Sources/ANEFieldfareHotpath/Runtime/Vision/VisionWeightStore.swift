import Darwin
import Foundation
import Metal
import ANEFieldfareModelFormat

public enum VisionPackError: Error, Equatable, CustomStringConvertible, Sendable {
    case notInstalled(String)
    case invalidMetadata(String)
    case fileSizeMismatch(file: String, expected: UInt64, actual: UInt64)
    case regionExceedsDevice(name: String, bytes: UInt64, maximum: UInt64)

    public var description: String {
        switch self {
        case let .notInstalled(path):
            "\(path) was installed without the vision weights"
        case let .invalidMetadata(detail):
            "vision weights are invalid: \(detail)"
        case let .fileSizeMismatch(file, expected, actual):
            "\(file) size \(actual) does not match manifest size \(expected)"
        case let .regionExceedsDevice(name, bytes, maximum):
            "vision tensor \(name) region \(bytes) exceeds Metal maximum \(maximum)"
        }
    }
}

/// The vision tower's tensors: the `vision` section of the model's
/// manifest, in its own file inside the model directory.
final class VisionWeightStore {
    let fileURL: URL
    let tensors: [String: FieldfareTensor]
    private let names: [String]
    private let weightsFD: Int32

    static func open(directory: ModelDirectory, directoryURL: URL,
                     manifest: FieldfareManifest) throws -> VisionWeightStore {
        guard let vision = manifest.vision else {
            throw VisionPackError.notInstalled(directoryURL.standardizedFileURL.path)
        }
        let weightsFD: Int32
        let actual: UInt64
        do {
            weightsFD = try directory.openFile(vision.file)
            actual = try directory.fileSize(fileDescriptor: weightsFD, relativePath: vision.file)
        } catch {
            throw VisionPackError.invalidMetadata("\(error)")
        }
        let expected = manifest.files[vision.file]?.size ?? 0
        guard actual == expected else {
            close(weightsFD)
            throw VisionPackError.fileSizeMismatch(file: vision.file, expected: expected, actual: actual)
        }
        return VisionWeightStore(fileURL: directoryURL.standardizedFileURL.appendingPathComponent(vision.file),
                                 tensors: vision.tensors, weightsFD: weightsFD)
    }

    private init(fileURL: URL, tensors: [FieldfareTensor], weightsFD: Int32) {
        self.fileURL = fileURL
        self.tensors = Dictionary(uniqueKeysWithValues: tensors.map { ($0.name, $0) })
        self.names = tensors.map(\.name)
        self.weightsFD = weightsFD
    }

    deinit { close(weightsFD) }

    /// Maps the smallest span of the file holding every region of the named
    /// tensors.
    func mapRegion(tensorNames: [String],
                   device: any MTLDevice) throws -> VisionMappedWeightRegion {
        let selected = try tensorNames.map { name -> FieldfareTensor in
            guard let tensor = tensors[name] else {
                throw VisionPackError.invalidMetadata("missing tensor \(name)")
            }
            return tensor
        }
        let regions = selected.flatMap { [$0.weights, $0.scales, $0.biases].compactMap { $0 } }
        guard let start = regions.map(\.offset).min(),
              let end = regions.map({ $0.offset + $0.size }).max(),
              end > start else {
            throw VisionPackError.invalidMetadata("empty mapped tensor group")
        }
        let length = end - start
        guard length <= UInt64(device.maxBufferLength) else {
            throw VisionPackError.regionExceedsDevice(
                name: tensorNames.first ?? "group", bytes: length,
                maximum: UInt64(device.maxBufferLength))
        }
        let resident = try ResidentBuffer(fileURL: fileURL, fileOffset: start, residentSize: length,
                                          device: device, fileDescriptor: weightsFD)
        return VisionMappedWeightRegion(
            resident: resident,
            tensors: Dictionary(uniqueKeysWithValues: selected.map { ($0.name, $0) }),
            base: start)
    }

    func tensorNames(withPrefix prefix: String) -> [String] {
        names.filter { $0.hasPrefix(prefix) }
    }
}

struct VisionMappedWeightRegion {
    let resident: ResidentBuffer
    let tensors: [String: FieldfareTensor]
    /// File offset of byte 0 of `buffer`.
    let base: UInt64

    var buffer: any MTLBuffer { resident.buffer }

    func offset(of name: String) throws -> Int {
        guard let tensor = tensors[name] else {
            throw VisionPackError.invalidMetadata("tensor \(name) is outside mapped region")
        }
        return Int(tensor.weights.offset - base)
    }

    func scalesOffset(of name: String) throws -> Int {
        guard let scales = tensors[name]?.scales else {
            throw VisionPackError.invalidMetadata("tensor \(name) has no scales")
        }
        return Int(scales.offset - base)
    }
}
