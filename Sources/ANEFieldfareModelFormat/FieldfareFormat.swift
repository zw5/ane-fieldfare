import Foundation

/// The `.fieldfare` model directory: one `manifest.json`, one resident file
/// holding every tensor read on every token, one fixed-stride file of routed
/// experts per layer, an optional vision file, and the tokenizer sidecar.
package enum FieldfareFormat {
    package static let name = "ane-fieldfare-model"
    package static let version = 1
    package static let manifestPath = "manifest.json"
    package static let manifestMaxBytes: UInt64 = 4 * 1024 * 1024
    package static let tokenizerDirectory = "tokenizer"
    /// Expert blobs start on 16 KiB pages so each one is a zero-copy buffer.
    package static let expertAlignment: UInt64 = 16_384
    /// Every tensor region starts on this boundary inside its file.
    package static let regionAlignment: UInt64 = 64
}

/// How a tensor's bytes encode its values.
package enum FieldfareEncoding: String, Codable, Sendable, CaseIterable {
    /// Row-major BF16, any rank.
    case bf16
    /// 4-bit, 32 values per group, one FP16 scale `d` per group, no bias:
    /// `w = d * (q - 8)`. Byte `k` of a row holds value `2k` in its low
    /// nibble and `2k + 1` in its high nibble. Scales are `[rows][cols / 32]`.
    case q4_0
    /// Unsigned 8-bit, 64 values per group, BF16 scale and bias per group:
    /// `w = q * scale + bias`. Scales and biases are `[rows][cols / 64]`.
    case affine8

    package var groupSize: UInt64? {
        switch self {
        case .bf16: nil
        case .q4_0: 32
        case .affine8: 64
        }
    }
}

package enum FieldfareFormatError: Error, Equatable, CustomStringConvertible, Sendable {
    case invalid(field: String, reason: String)
    case overflow(field: String)

    package var description: String {
        switch self {
        case let .invalid(field, reason): "\(field): \(reason)"
        case let .overflow(field): "\(field): arithmetic overflow"
        }
    }
}

@inline(__always)
package func fieldfareAdd(_ lhs: UInt64, _ rhs: UInt64, field: String) throws -> UInt64 {
    let (value, overflow) = lhs.addingReportingOverflow(rhs)
    guard !overflow else { throw FieldfareFormatError.overflow(field: field) }
    return value
}

@inline(__always)
package func fieldfareMultiply(_ lhs: UInt64, _ rhs: UInt64, field: String) throws -> UInt64 {
    let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
    guard !overflow else { throw FieldfareFormatError.overflow(field: field) }
    return value
}

package enum FieldfarePath {
    /// A relative path with no empty, `.` or `..` component, already in
    /// normal form, so a directory-relative open cannot leave the model root.
    package static func validate(_ path: String, field: String) throws {
        guard !path.isEmpty, !path.hasPrefix("/"), !path.contains("\0") else {
            throw FieldfareFormatError.invalid(field: field, reason: "unsafe relative path")
        }
        let components = path.components(separatedBy: "/")
        guard components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
            throw FieldfareFormatError.invalid(field: field, reason: "non-canonical path")
        }
        guard NSString.path(withComponents: components) == path else {
            throw FieldfareFormatError.invalid(field: field, reason: "non-normalized path")
        }
    }
}
