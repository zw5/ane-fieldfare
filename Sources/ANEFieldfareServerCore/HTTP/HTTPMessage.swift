public import Foundation

public enum HTTPVersion: String, Sendable, Equatable {
    case http10 = "HTTP/1.0"
    case http11 = "HTTP/1.1"
}

/// Header fields in arrival order; names compare case-insensitively.
public struct HTTPHeaders: Sendable, Equatable {
    public struct Field: Sendable, Equatable {
        public let name: String
        public let value: String
    }

    public private(set) var fields: [Field] = []

    public init() {}

    public mutating func add(_ name: String, _ value: String) {
        fields.append(Field(name: name, value: value))
    }

    /// Replaces every field named `name` with one field.
    public mutating func set(_ name: String, _ value: String) {
        fields.removeAll { $0.name.caseInsensitiveCompare(name) == .orderedSame }
        fields.append(Field(name: name, value: value))
    }

    /// The first value of `name`.
    public subscript(_ name: String) -> String? {
        fields.first { $0.name.caseInsensitiveCompare(name) == .orderedSame }?.value
    }

    /// Every value of `name`, in order.
    public func values(_ name: String) -> [String] {
        fields.filter { $0.name.caseInsensitiveCompare(name) == .orderedSame }.map(\.value)
    }

    /// The comma-separated tokens of every `name` field, lowercased.
    public func tokens(_ name: String) -> [String] {
        values(name).flatMap { $0.split(separator: ",") }
            .map { $0.trimmingCharacters(in: .whitespaces).lowercased() }
            .filter { !$0.isEmpty }
    }
}

public struct HTTPRequest: Sendable, Equatable {
    public let method: String
    /// The request target as sent, query included.
    public let target: String
    public let version: HTTPVersion
    public let headers: HTTPHeaders
    public let body: Data

    public init(method: String, target: String, version: HTTPVersion, headers: HTTPHeaders, body: Data) {
        self.method = method
        self.target = target
        self.version = version
        self.headers = headers
        self.body = body
    }

    /// The target's path, without the query.
    public var path: String {
        if let index = target.firstIndex(of: "?") { return String(target[..<index]) }
        return target
    }

    /// Whether the connection stays open after the response (RFC 9112 §9.3):
    /// HTTP/1.1 persists unless `Connection: close`; HTTP/1.0 only with
    /// `Connection: keep-alive`.
    public var keepAlive: Bool {
        let connection = headers.tokens("Connection")
        switch version {
        case .http11: return !connection.contains("close")
        case .http10: return connection.contains("keep-alive")
        }
    }
}

public struct HTTPResponse: Sendable {
    public var status: Int
    public var headers: HTTPHeaders
    public var body: Data

    public init(status: Int, headers: HTTPHeaders = HTTPHeaders(), body: Data = Data()) {
        self.status = status
        self.headers = headers
        self.body = body
    }

    /// The status line, headers and (unless `includeBody` is false, for
    /// `HEAD`) the body. `Content-Length` and `Connection` are set here, so a
    /// response cannot disagree with how the connection is handled.
    public func serialized(keepAlive: Bool, includeBody: Bool = true) -> Data {
        var head = "HTTP/1.1 \(status) \(HTTPResponse.reasonPhrase(status))\r\n"
        for field in headers.fields where !Self.managed(field.name) {
            head += "\(field.name): \(field.value)\r\n"
        }
        // 1xx, 204 and 304 carry no body and no length (RFC 9110 §8.6).
        if status >= 200, status != 204, status != 304 {
            head += "Content-Length: \(body.count)\r\n"
        }
        head += "Connection: \(keepAlive ? "keep-alive" : "close")\r\n\r\n"
        var data = Data(head.utf8)
        if includeBody, status >= 200, status != 204, status != 304 { data.append(body) }
        return data
    }

    /// The head of a streamed response. HTTP/1.1 frames the body with chunked
    /// transfer coding, so the connection can stay open after it; HTTP/1.0
    /// has no chunked coding, so the body runs to the connection's close.
    public static func streamingHead(status: Int, headers: HTTPHeaders, version: HTTPVersion,
                                     keepAlive: Bool) -> Data {
        var head = "HTTP/1.1 \(status) \(reasonPhrase(status))\r\n"
        for field in headers.fields where !managed(field.name) {
            head += "\(field.name): \(field.value)\r\n"
        }
        if version == .http11 {
            head += "Transfer-Encoding: chunked\r\n"
            head += "Connection: \(keepAlive ? "keep-alive" : "close")\r\n\r\n"
        } else {
            head += "Connection: close\r\n\r\n"
        }
        return Data(head.utf8)
    }

    private static func managed(_ name: String) -> Bool {
        ["content-length", "connection", "transfer-encoding"].contains(name.lowercased())
    }

    public static func reasonPhrase(_ status: Int) -> String {
        switch status {
        case 100: return "Continue"
        case 200: return "OK"
        case 204: return "No Content"
        case 400: return "Bad Request"
        case 401: return "Unauthorized"
        case 404: return "Not Found"
        case 405: return "Method Not Allowed"
        case 408: return "Request Timeout"
        case 413: return "Content Too Large"
        case 429: return "Too Many Requests"
        case 431: return "Request Header Fields Too Large"
        case 500: return "Internal Server Error"
        case 501: return "Not Implemented"
        case 503: return "Service Unavailable"
        case 505: return "HTTP Version Not Supported"
        default: return "Status \(status)"
        }
    }
}

/// Chunked transfer coding (RFC 9112 §7.1) for streamed response bodies.
public enum HTTPChunkedEncoding {
    public static func chunk(_ data: Data) -> Data {
        guard !data.isEmpty else { return Data() }
        var framed = Data(String(data.count, radix: 16).utf8)
        framed.append(contentsOf: [0x0D, 0x0A])
        framed.append(data)
        framed.append(contentsOf: [0x0D, 0x0A])
        return framed
    }

    /// The last chunk with an empty trailer section.
    public static let end = Data("0\r\n\r\n".utf8)
}
