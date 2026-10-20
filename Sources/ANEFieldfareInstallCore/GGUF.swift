package import Foundation

/// The parts of a GGUF v3 header the installer reads: scalar and short-array
/// metadata, and the tensor table. Long arrays (the tokenizer vocabulary and
/// merges) are skipped, keeping only their length.
package struct GGUFHeader: Sendable {
    package enum Value: Sendable, Equatable {
        case int(Int64)
        case uint(UInt64)
        case float(Double)
        case bool(Bool)
        case string(String)
        case array([Value])
        /// An array longer than `GGUFReader.maxKeptArray`, kept as its length.
        case longArray(count: UInt64)
    }

    package enum TensorType: UInt32, Sendable {
        case f32 = 0
        case f16 = 1
        case q4_0 = 2
        case q8_0 = 8
        case q6_K = 14
        case bf16 = 30

        /// (values per block, bytes per block)
        package var block: (values: UInt64, bytes: UInt64) {
            switch self {
            case .f32: (1, 4)
            case .f16, .bf16: (1, 2)
            case .q4_0: (32, 18)
            case .q8_0: (32, 34)
            case .q6_K: (256, 210)
            }
        }
    }

    package struct Tensor: Sendable, Equatable {
        package let name: String
        /// GGUF order: `dims[0]` is the contiguous (column) dimension.
        package let dims: [UInt64]
        package let type: TensorType
        /// Absolute file offset.
        package let offset: UInt64
        package let size: UInt64

        package var count: UInt64 { dims.reduce(1, *) }
        /// Values per row.
        package var cols: UInt64 { dims[0] }
        /// Rows across every dimension but the first.
        package var rows: UInt64 { dims.dropFirst().reduce(1, *) }
        package var rowBytes: UInt64 { cols / type.block.values * type.block.bytes }
    }

    package let version: UInt32
    package let metadata: [String: Value]
    /// In file order.
    package let tensors: [Tensor]
    package let dataStart: UInt64

    package func tensor(_ name: String) -> Tensor? { tensors.first { $0.name == name } }
}

package enum GGUFError: Error, Equatable, CustomStringConvertible, Sendable {
    /// The header continues past the bytes given; retry with at least this many.
    case needMoreBytes(UInt64)
    case invalid(String)

    package var description: String {
        switch self {
        case let .needMoreBytes(n): "GGUF header needs at least \(n) bytes"
        case let .invalid(reason): "GGUF: \(reason)"
        }
    }
}

package enum GGUFReader {
    package static let maxKeptArray: UInt64 = 4096
    static let maxStringBytes: UInt64 = 16 * 1024 * 1024

    /// Parses the header from a file prefix. Throws `needMoreBytes` when the
    /// prefix ends inside the header, so a caller can fetch more and retry.
    package static func parse(_ data: Data, fileSize: UInt64) throws -> GGUFHeader {
        var cursor = Cursor(data: data)
        guard try cursor.bytes(4) == Data("GGUF".utf8) else { throw GGUFError.invalid("bad magic") }
        let version = try cursor.u32()
        guard version == 3 else { throw GGUFError.invalid("unsupported version \(version)") }
        let tensorCount = try cursor.u64()
        let kvCount = try cursor.u64()
        guard tensorCount < 1 << 20, kvCount < 1 << 20 else { throw GGUFError.invalid("implausible counts") }

        var metadata: [String: GGUFHeader.Value] = [:]
        for _ in 0..<kvCount {
            let key = try cursor.string()
            metadata[key] = try cursor.value(type: cursor.u32())
        }
        var alignment: UInt64 = 32
        if case let .uint(value)? = metadata["general.alignment"] { alignment = value }
        guard alignment > 0, alignment.nonzeroBitCount == 1 else { throw GGUFError.invalid("bad alignment") }

        var raw: [(String, [UInt64], UInt32, UInt64)] = []
        for _ in 0..<tensorCount {
            let name = try cursor.string()
            let rank = try cursor.u32()
            guard (1...4).contains(rank) else { throw GGUFError.invalid("\(name): rank \(rank)") }
            var dims: [UInt64] = []
            for _ in 0..<rank { dims.append(try cursor.u64()) }
            raw.append((name, dims, try cursor.u32(), try cursor.u64()))
        }
        let dataStart = (UInt64(cursor.position) + alignment - 1) / alignment * alignment

        var tensors: [GGUFHeader.Tensor] = []
        for (name, dims, typeCode, relative) in raw {
            guard let type = GGUFHeader.TensorType(rawValue: typeCode) else {
                throw GGUFError.invalid("\(name): unsupported tensor type \(typeCode)")
            }
            guard dims.allSatisfy({ $0 > 0 }), dims[0] % type.block.values == 0 else {
                throw GGUFError.invalid("\(name): dims \(dims) do not fit \(type) blocks")
            }
            var count: UInt64 = 1
            for dim in dims {
                let (product, overflow) = count.multipliedReportingOverflow(by: dim)
                guard !overflow else { throw GGUFError.invalid("\(name): size overflow") }
                count = product
            }
            let size = count / type.block.values * type.block.bytes
            let (offset, overflow) = dataStart.addingReportingOverflow(relative)
            guard !overflow, relative % alignment == 0, offset <= fileSize, size <= fileSize - offset else {
                throw GGUFError.invalid("\(name): data outside the file")
            }
            tensors.append(.init(name: name, dims: dims, type: type, offset: offset, size: size))
        }
        tensors.sort { $0.offset < $1.offset }
        for (a, b) in zip(tensors, tensors.dropFirst()) where a.offset + a.size > b.offset {
            throw GGUFError.invalid("\(b.name) overlaps \(a.name)")
        }
        guard Set(tensors.map(\.name)).count == tensors.count else {
            throw GGUFError.invalid("duplicate tensor name")
        }
        return GGUFHeader(version: version, metadata: metadata, tensors: tensors, dataStart: dataStart)
    }

    private struct Cursor {
        let data: Data
        var position = 0

        init(data: Data) { self.data = data }

        mutating func bytes(_ n: UInt64) throws -> Data {
            let end = UInt64(position) + n
            guard end <= UInt64(data.count) else { throw GGUFError.needMoreBytes(max(end, UInt64(data.count) * 2)) }
            let start = data.startIndex + position
            position = Int(end)
            return data.subdata(in: start..<(start + Int(n)))
        }

        mutating func skip(_ n: UInt64) throws {
            let end = UInt64(position) + n
            guard end <= UInt64(data.count) else { throw GGUFError.needMoreBytes(max(end, UInt64(data.count) * 2)) }
            position = Int(end)
        }

        mutating func scalar<T: FixedWidthInteger>(_: T.Type) throws -> T {
            // Little-endian; built from the bytes so no unaligned load happens.
            var value = T.zero
            for (i, byte) in try bytes(UInt64(MemoryLayout<T>.size)).enumerated() {
                value |= T(truncatingIfNeeded: byte) << (8 * i)
            }
            return value
        }

        mutating func u32() throws -> UInt32 { try scalar(UInt32.self) }
        mutating func u64() throws -> UInt64 { try scalar(UInt64.self) }

        mutating func string() throws -> String {
            let length = try u64()
            guard length <= GGUFReader.maxStringBytes else { throw GGUFError.invalid("string too long") }
            guard let s = String(data: try bytes(length), encoding: .utf8) else {
                throw GGUFError.invalid("string is not UTF-8")
            }
            return s
        }

        /// Fixed byte width of a scalar value type, nil for strings and arrays.
        static func width(_ type: UInt32) -> UInt64? {
            switch type {
            case 0, 1, 7: 1
            case 2, 3: 2
            case 4, 5, 6: 4
            case 10, 11, 12: 8
            default: nil
            }
        }

        mutating func value(type: UInt32) throws -> GGUFHeader.Value {
            switch type {
            case 0: .uint(UInt64(try scalar(UInt8.self)))
            case 1: .int(Int64(try scalar(Int8.self)))
            case 2: .uint(UInt64(try scalar(UInt16.self)))
            case 3: .int(Int64(try scalar(Int16.self)))
            case 4: .uint(UInt64(try u32()))
            case 5: .int(Int64(try scalar(Int32.self)))
            case 6: .float(Double(Float(bitPattern: try u32())))
            case 7: .bool(try scalar(UInt8.self) != 0)
            case 8: .string(try string())
            case 9: try array()
            case 10: .uint(try u64())
            case 11: .int(try scalar(Int64.self))
            case 12: .float(Double(bitPattern: try u64()))
            default: throw GGUFError.invalid("unknown metadata type \(type)")
            }
        }

        mutating func array() throws -> GGUFHeader.Value {
            let elementType = try u32()
            let count = try u64()
            if count <= GGUFReader.maxKeptArray {
                var values: [GGUFHeader.Value] = []
                for _ in 0..<count { values.append(try value(type: elementType)) }
                return .array(values)
            }
            if let width = Self.width(elementType) {
                let (bytes, overflow) = width.multipliedReportingOverflow(by: count)
                guard !overflow else { throw GGUFError.invalid("array size overflow") }
                try skip(bytes)
            } else if elementType == 8 {
                for _ in 0..<count {
                    let length = try u64()
                    guard length <= GGUFReader.maxStringBytes else { throw GGUFError.invalid("string too long") }
                    try skip(length)
                }
            } else {
                throw GGUFError.invalid("long array of type \(elementType)")
            }
            return .longArray(count: count)
        }
    }
}

extension GGUFHeader.Value {
    package var intValue: Int? {
        switch self {
        case let .int(v): Int(exactly: v)
        case let .uint(v): Int(exactly: v)
        default: nil
        }
    }

    package var doubleValue: Double? {
        switch self {
        case let .float(v): v
        case let .int(v): Double(v)
        case let .uint(v): Double(v)
        default: nil
        }
    }
}
