import Foundation
import Testing
import ANEFieldfareServerCore

/// The HTTP/1.1 request parser, fed bytes the way a socket delivers them.
/// Content-Length, chunked and 100-continue bodies over a live connection are
/// covered end to end in `ServerIntegrationTests`.
@Suite struct HTTPRequestParserTests {
    private func parse(_ text: String, limits: HTTPRequestParser.Limits = .init()) throws -> [HTTPRequest] {
        var parser = HTTPRequestParser(limits: limits)
        parser.append(Data(text.utf8))
        var requests: [HTTPRequest] = []
        while case .request(let request) = try parser.next() { requests.append(request) }
        return requests
    }

    private func status(_ text: String, limits: HTTPRequestParser.Limits = .init()) -> Int? {
        do {
            _ = try parse(text, limits: limits)
            return nil
        } catch let error as HTTPParseError {
            return error.status
        } catch {
            return -1
        }
    }

    @Test func bytesArrivingOneAtATimeParseTheSameRequest() throws {
        let text = "POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n4\r\nWiki\r\n6;ext=1\r\npedia \r\n0\r\nTrailer: t\r\n\r\n"
        var parser = HTTPRequestParser()
        var result: HTTPRequest?
        for byte in Data(text.utf8) {
            parser.append(Data([byte]))
            if case .request(let request) = try parser.next() { result = request }
        }
        let request = try #require(result)
        #expect(String(decoding: request.body, as: UTF8.self) == "Wikipedia ")
    }

    @Test func pipelinedRequestsParseInOrder() throws {
        let requests = try parse(
            "GET /health HTTP/1.1\r\n\r\n"
            + "POST /a?x=1 HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\nhi"
            + "POST http://localhost:8080/b HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n0\r\n\r\n")
        try #require(requests.map(\.path) == ["/health", "/a", "/b"])
        #expect(requests[1].target == "/a?x=1")
        #expect(requests[1].headers["content-type"] == "application/json")
        #expect(requests.map { String(decoding: $0.body, as: UTF8.self) } == ["", "hi", "abc"])
    }

    @Test func keepAliveFollowsVersionAndConnectionHeader() throws {
        let requests = try parse(
            "GET / HTTP/1.1\r\n\r\n"
            + "GET / HTTP/1.1\r\nConnection: Close\r\n\r\n"
            + "GET / HTTP/1.0\r\n\r\n"
            + "GET / HTTP/1.0\r\nConnection: keep-alive\r\n\r\n")
        #expect(requests.map(\.keepAlive) == [true, false, false, true])
    }

    @Test func malformedRequestsAreRefusedWithTheirStatus() {
        #expect(status("GARBAGE\r\n\r\n") == 400)
        #expect(status("GET /a HTTP/2.0\r\n\r\n") == 505)
        #expect(status("GET /a HTTP/1.1\r\nBad Header: x\r\n\r\n") == 400)
        #expect(status("GET /a HTTP/1.1\r\nA: b\r\n folded\r\n\r\n") == 400)
        #expect(status("POST /a HTTP/1.1\r\nContent-Length: -1\r\n\r\n") == 400)
        #expect(status("POST /a HTTP/1.1\r\nContent-Length: 1\r\nContent-Length: 2\r\n\r\nab") == 400)
        #expect(status("POST /a HTTP/1.1\r\nContent-Length: 2\r\nTransfer-Encoding: chunked\r\n\r\n") == 400)
        #expect(status("POST /a HTTP/1.1\r\nTransfer-Encoding: gzip\r\n\r\n") == 501)
        #expect(status("POST /a HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\nzz\r\n") == 400)
        #expect(status("POST /a HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabcX\r\n") == 400)
    }

    @Test func limitsAreEnforced() {
        let small = HTTPRequestParser.Limits(maxHeaderBytes: 64, maxBodyBytes: 8)
        #expect(status("GET /a HTTP/1.1\r\nX: " + String(repeating: "a", count: 100) + "\r\n\r\n", limits: small) == 431)
        #expect(status("POST /a HTTP/1.1\r\nContent-Length: 9\r\n\r\n", limits: small) == 413)
        #expect(status("POST /a HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nabcde\r\n5\r\n", limits: small) == 413)
        // Identical repeated lengths are one length.
        #expect(status("POST /a HTTP/1.1\r\nContent-Length: 2\r\nContent-Length: 2\r\n\r\nab", limits: small) == nil)
    }
}
