/// Defaults the CLI's flags and the server's request defaults share.
public enum OptionDefaults {
    public static let temperature: Float = 0.2
    public static let topK: Int? = 64
    public static let topP: Float? = 0.95
    public static let maxContext = 32_768
    /// The smallest `--max-context` either tool accepts.
    public static let minimumMaxContext = 16
}

/// Walks a command line flag by flag. The CLI and the server read the flags
/// they share through the same readers, so each flag has one rule.
public struct ArgumentCursor {
    private let argv: [String]
    private var index = 0

    public init(_ argv: [String]) { self.argv = argv }

    /// The next flag, or nil when the line is consumed.
    public mutating func nextFlag() -> String? {
        guard index < argv.count else { return nil }
        index += 1
        return argv[index - 1]
    }

    /// The value after `flag`.
    public mutating func value(for flag: String) throws -> String {
        guard index < argv.count else { throw ArgsError.missingValue(flag: flag) }
        index += 1
        return argv[index - 1]
    }

    /// The value after `flag`, read as `T`, or `invalidValue` when `accept` refuses it.
    public mutating func value<T: LosslessStringConvertible>(
        for flag: String, as _: T.Type, where accept: (T) -> Bool = { _ in true }) throws -> T {
        let text = try value(for: flag)
        guard let parsed = T(text), accept(parsed) else { throw ArgsError.invalidValue(flag: flag, value: text) }
        return parsed
    }

    /// The value after `flag`, one of `T`'s raw values.
    public mutating func choice<T: RawRepresentable>(for flag: String, as _: T.Type) throws -> T
        where T.RawValue == String {
        let text = try value(for: flag)
        guard let parsed = T(rawValue: text) else { throw ArgsError.invalidValue(flag: flag, value: text) }
        return parsed
    }

    public mutating func temperature(for flag: String) throws -> Float {
        try value(for: flag, as: Float.self) { $0.isFinite && $0 >= 0 }
    }

    /// 1...256, or 0 for off.
    public mutating func topK(for flag: String) throws -> Int? {
        let parsed = try value(for: flag, as: Int.self) { (0...256).contains($0) }
        return parsed == 0 ? nil : parsed
    }

    public mutating func topP(for flag: String) throws -> Float {
        try value(for: flag, as: Float.self) { $0 > 0 && $0 <= 1 }
    }

    public mutating func maxContext(for flag: String) throws -> Int {
        try value(for: flag, as: Int.self) { $0 >= OptionDefaults.minimumMaxContext }
    }
}

/// Nucleus sampling runs over the top-k candidates; there is no
/// full-vocabulary nucleus.
public func validateTruncation(temperature: Float, topK: Int?, topP: Float?) throws {
    if temperature > 0, topK == nil, let topP, topP < 1 {
        throw ArgsError.invalidValue(flag: "--top-p", value: "\(topP) requires --top-k between 1 and 256")
    }
}
