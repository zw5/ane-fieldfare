package import Foundation
package import ANEFieldfareImportedFormat

// Writers for synthetic gturbo fixtures. The runtime only reads the format.

extension GTurboManifestCodec {
    package static func encode(_ manifest: GTurboManifestV1) throws -> Data {
        try validate(manifest)
        return try prettyJSON(manifest, field: "manifest.json")
    }
}

extension GTurboPackedExpertsLayoutCodec {
    package static func encode(_ layout: GTurboPackedExpertsLayoutV1) throws -> Data {
        try GTurboV1StructuralValidator.validate(layout)
        return try prettyJSON(layout, field: "packed_experts/layout.json")
    }
}

extension GTurboVisionManifestCodec {
    package static func encode(_ manifest: GTurboVisionManifestV1) throws -> Data {
        try GTurboVisionStructuralValidator.validate(manifest)
        return try prettyJSON(manifest, field: GTurboVisionFormatV1.manifestFile)
    }
}

private func prettyJSON(_ value: some Encodable, field: String) throws -> Data {
    do {
        let object = try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
        return try JSONSerialization.data(
            withJSONObject: object, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    } catch {
        throw GTurboFormatError.invalid(field: field, reason: "\(error)")
    }
}

extension GTurboResidentIndexCodec {
    package static func writeHeader(into buffer: UnsafeMutableRawPointer,
                                    header: GTurboResidentIndexHeaderV1) {
        unsafe store(header.indexSize, buffer, 0)
        unsafe store(header.residentSize, buffer, 8)
        unsafe store(header.entryCount, buffer, 16)
    }

    package static func writeEntry(into buffer: UnsafeMutableRawPointer,
                                   entry: GTurboResidentIndexEntryV1,
                                   nameOffset: UInt32) {
        precondition(entry.shape.count == 4)
        unsafe store(nameOffset, buffer, 0)
        unsafe store(UInt16(entry.name.utf8.count), buffer, 4)
        unsafe store(entry.dtype, buffer, 6)
        unsafe store(UInt8(0), buffer, 7)
        unsafe store(entry.fileOffset, buffer, 8)
        unsafe store(entry.sizeBytes, buffer, 16)
        for i in 0..<4 { unsafe store(entry.shape[i], buffer, 24 + i * 4) }
        unsafe store(entry.scaleOffset, buffer, 40)
        unsafe store(entry.scaleSize, buffer, 48)
        unsafe store(entry.biasOffset, buffer, 56)
        unsafe store(entry.biasSize, buffer, 64)
    }

    private static func store<T: FixedWidthInteger>(_ value: T, _ buffer: UnsafeMutableRawPointer, _ offset: Int) {
        unsafe buffer.advanced(by: offset).storeBytes(of: value.littleEndian, as: T.self)
    }
}
