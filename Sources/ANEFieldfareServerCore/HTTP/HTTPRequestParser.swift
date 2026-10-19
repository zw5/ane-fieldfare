public import Foundation

/// A request the parser refuses, with the status to answer it with. The
/// connection closes after the answer: once framing is in doubt, nothing
/// after it can be trusted to start a request.
public struct HTTPParseError: Error, Sendable, Equatable, CustomStringConvertible {
    public let status: Int
    public let message: String

    public init(status: Int, message: String) {
        self.status = status
        self.message = message
    }

    public var description: String { "\(status): \(message)" }
}

/// Incremental HTTP/1.x request parser (RFC 9112).
///
/// Bytes go in with `append` as they arrive; `next()` returns a complete
/// request, asks for more data, or asks the caller to send `100 Continue`.
/// Bytes past a complete request stay buffered for the next one, so pipelined
/// requests parse in order.
///
/// Bodies are framed by `Content-Length` or by `Transfer-Encoding: chunked`.
/// A request with both is refused rather than resolved, since the two
/// disagreeing is the request-smuggling case. Other transfer codings are
/// refused with 501.
public struct HTTPRequestParser {
    public struct Limits: Sendable {
        public var maxHeaderBytes: Int
        public var maxBodyBytes: Int

        public init(maxHeaderBytes: Int = 32 * 1024, maxBodyBytes: Int = 16 * 1024 * 1024) {
            self.maxHeaderBytes = maxHeaderBytes
            self.maxBodyBytes = maxBodyBytes
        }
    }

    public enum Event: Sendable, Equatable {
        case needMoreData
        /// The client sent `Expect: 100-continue` and waits for the interim
        /// response before sending the body. Returned once per request.
        case continueExpected
        case request(HTTPRequest)
    }

    private struct Head {
        let method: String
        let target: String
        let version: HTTPVersion
        let headers: HTTPHeaders
        let expectsContinue: Bool
    }

    private enum ChunkPhase {
        case size
        case data(remaining: Int)
        case dataEnd
        case trailers
    }

    private enum State {
        case head
        case fixedBody(Head, length: Int)
        case chunkedBody(Head, ChunkPhase, Data)
    }

    public let limits: Limits
    private var buffer: [UInt8] = []
    private var cursor = 0
    private var state: State = .head
    private var continueSent = false
    private var trailerBytes = 0

    public init(limits: Limits = Limits()) {
        self.limits = limits
    }

    /// Whether unconsumed bytes are buffered, i.e. a request has started.
    public var hasBufferedData: Bool {
        if case .head = state { return cursor < buffer.count }
        return true
    }

    public mutating func append(_ data: Data) {
        if cursor > 0, cursor >= buffer.count / 2 {
            buffer.removeFirst(cursor)
            cursor = 0
        }
        buffer.append(contentsOf: data)
    }

    public mutating func next() throws(HTTPParseError) -> Event {
        switch state {
        case .head:
            guard let head = try parseHead() else { return .needMoreData }
            return try startBody(head)
        case .fixedBody(let head, let length):
            return try fixedBody(head, length: length)
        case .chunkedBody(let head, let phase, let body):
            return try chunkedBody(head, phase: phase, body: body)
        }
    }

    // MARK: - Head

    private static let crlf: [UInt8] = [0x0D, 0x0A]

    /// The index of the next CRLF at or after `from`, if buffered.
    private func crlf(from: Int) -> Int? {
        var index = from
        while index + 1 < buffer.count {
            if buffer[index] == 0x0D, buffer[index + 1] == 0x0A { return index }
            index += 1
        }
        return nil
    }

    private mutating func parseHead() throws(HTTPParseError) -> Head? {
        // A server should ignore at least one empty line before the request
        // line (RFC 9112 §2.2); clients send one after a POST body.
        while cursor + 1 < buffer.count, buffer[cursor] == 0x0D, buffer[cursor + 1] == 0x0A {
            cursor += 2
        }
        var lines: [String] = []
        var index = cursor
        while true {
            guard let end = crlf(from: index) else {
                if buffer.count - cursor > limits.maxHeaderBytes {
                    throw HTTPParseError(status: 431, message: "request header section is too large")
                }
                return nil
            }
            if end - cursor > limits.maxHeaderBytes {
                throw HTTPParseError(status: 431, message: "request header section is too large")
            }
            if end == index {
                index = end + 2
                break
            }
            lines.append(String(decoding: buffer[index..<end], as: UTF8.self))
            index = end + 2
        }
        cursor = index
        guard let requestLine = lines.first else {
            throw HTTPParseError(status: 400, message: "missing request line")
        }
        let parts = requestLine.split(separator: " ", omittingEmptySubsequences: false)
        guard parts.count == 3, !parts[0].isEmpty, !parts[1].isEmpty,
              parts[0].allSatisfy({ $0.isLetter && $0.isASCII }) else {
            throw HTTPParseError(status: 400, message: "malformed request line")
        }
        let version: HTTPVersion
        switch parts[2] {
        case "HTTP/1.1": version = .http11
        case "HTTP/1.0": version = .http10
        default:
            guard parts[2].hasPrefix("HTTP/") else {
                throw HTTPParseError(status: 400, message: "malformed request line")
            }
            throw HTTPParseError(status: 505, message: "only HTTP/1.0 and HTTP/1.1 are supported")
        }
        var target = String(parts[1])
        // Absolute form (RFC 9112 §3.2.2): keep the path and query.
        for scheme in ["http://", "https://"] where target.lowercased().hasPrefix(scheme) {
            let rest = target.dropFirst(scheme.count)
            target = rest.firstIndex(of: "/").map { String(rest[$0...]) } ?? "/"
        }
        guard target.hasPrefix("/") || target == "*" else {
            throw HTTPParseError(status: 400, message: "malformed request target")
        }

        var headers = HTTPHeaders()
        for line in lines.dropFirst() {
            guard let first = line.first, first != " ", first != "\t" else {
                throw HTTPParseError(status: 400, message: "obsolete header line folding is not accepted")
            }
            guard let colon = line.firstIndex(of: ":") else {
                throw HTTPParseError(status: 400, message: "malformed header line")
            }
            let name = String(line[..<colon])
            // No whitespace between a field name and its colon (RFC 9112 §5.1).
            guard !name.isEmpty, !name.contains(where: { $0 == " " || $0 == "\t" }) else {
                throw HTTPParseError(status: 400, message: "malformed header name")
            }
            let value = line[line.index(after: colon)...].trimmingCharacters(in: CharacterSet(charactersIn: " \t"))
            headers.add(name, value)
        }
        let expectsContinue = headers.tokens("Expect").contains("100-continue")
        return Head(method: String(parts[0]), target: target, version: version, headers: headers,
                    expectsContinue: expectsContinue)
    }

    private mutating func startBody(_ head: Head) throws(HTTPParseError) -> Event {
        continueSent = false
        trailerBytes = 0
        let codings = head.headers.tokens("Transfer-Encoding")
        let lengths = head.headers.values("Content-Length")
        if !codings.isEmpty {
            guard lengths.isEmpty else {
                throw HTTPParseError(status: 400, message: "Content-Length and Transfer-Encoding together")
            }
            guard codings == ["chunked"] else {
                throw HTTPParseError(status: 501, message: "unsupported transfer coding: \(codings.joined(separator: ", "))")
            }
            guard head.version == .http11 else {
                throw HTTPParseError(status: 400, message: "chunked transfer coding requires HTTP/1.1")
            }
            return try chunkedBody(head, phase: .size, body: Data())
        }
        var length = 0
        if !lengths.isEmpty {
            // A list of identical values is one length (RFC 9112 §6.3).
            let values = Set(lengths.flatMap { $0.split(separator: ",") }
                .map { $0.trimmingCharacters(in: .whitespaces) })
            guard values.count == 1, let value = values.first, !value.isEmpty, value.count <= 18,
                  value.allSatisfy({ $0.isASCII && $0.isNumber }), let parsed = Int(value) else {
                throw HTTPParseError(status: 400, message: "invalid Content-Length")
            }
            length = parsed
        }
        guard length <= limits.maxBodyBytes else {
            throw HTTPParseError(status: 413, message: "request body exceeds \(limits.maxBodyBytes) bytes")
        }
        return try fixedBody(head, length: length)
    }

    // MARK: - Bodies

    private mutating func fixedBody(_ head: Head, length: Int) throws(HTTPParseError) -> Event {
        guard buffer.count - cursor >= length else {
            state = .fixedBody(head, length: length)
            return askForBody(head, received: buffer.count - cursor)
        }
        let body = Data(buffer[cursor..<(cursor + length)])
        cursor += length
        return finish(head, body: body)
    }

    private mutating func chunkedBody(_ head: Head, phase: ChunkPhase,
                                      body: Data) throws(HTTPParseError) -> Event {
        var phase = phase
        var body = body
        while true {
            switch phase {
            case .size:
                guard let end = crlf(from: cursor) else {
                    if buffer.count - cursor > 1024 {
                        throw HTTPParseError(status: 400, message: "malformed chunk size line")
                    }
                    state = .chunkedBody(head, phase, body)
                    return askForBody(head, received: body.count + buffer.count - cursor)
                }
                let line = String(decoding: buffer[cursor..<end], as: UTF8.self)
                let sizeText = (line.split(separator: ";", maxSplits: 1, omittingEmptySubsequences: false)
                    .first.map(String.init) ?? "").trimmingCharacters(in: CharacterSet(charactersIn: " \t"))
                guard !sizeText.isEmpty, sizeText.count <= 15, sizeText.allSatisfy(\.isHexDigit),
                      let size = Int(sizeText, radix: 16) else {
                    throw HTTPParseError(status: 400, message: "malformed chunk size")
                }
                guard body.count + size <= limits.maxBodyBytes else {
                    throw HTTPParseError(status: 413, message: "request body exceeds \(limits.maxBodyBytes) bytes")
                }
                cursor = end + 2
                phase = size == 0 ? .trailers : .data(remaining: size)
            case .data(let remaining):
                let available = min(remaining, buffer.count - cursor)
                if available > 0 {
                    body.append(contentsOf: buffer[cursor..<(cursor + available)])
                    cursor += available
                }
                if available < remaining {
                    state = .chunkedBody(head, .data(remaining: remaining - available), body)
                    return askForBody(head, received: body.count)
                }
                phase = .dataEnd
            case .dataEnd:
                guard buffer.count - cursor >= 2 else {
                    state = .chunkedBody(head, phase, body)
                    return .needMoreData
                }
                guard buffer[cursor] == 0x0D, buffer[cursor + 1] == 0x0A else {
                    throw HTTPParseError(status: 400, message: "chunk data is not followed by CRLF")
                }
                cursor += 2
                phase = .size
            case .trailers:
                guard let end = crlf(from: cursor) else {
                    if buffer.count - cursor + trailerBytes > limits.maxHeaderBytes {
                        throw HTTPParseError(status: 431, message: "trailer section is too large")
                    }
                    state = .chunkedBody(head, phase, body)
                    return .needMoreData
                }
                trailerBytes += end - cursor + 2
                if trailerBytes > limits.maxHeaderBytes {
                    throw HTTPParseError(status: 431, message: "trailer section is too large")
                }
                let empty = end == cursor
                cursor = end + 2
                if empty { return finish(head, body: body) }
            }
        }
    }

    private mutating func askForBody(_ head: Head, received: Int) -> Event {
        if head.expectsContinue, !continueSent, received == 0 {
            continueSent = true
            return .continueExpected
        }
        return .needMoreData
    }

    private mutating func finish(_ head: Head, body: Data) -> Event {
        state = .head
        return .request(HTTPRequest(method: head.method, target: head.target, version: head.version,
                                    headers: head.headers, body: body))
    }
}
