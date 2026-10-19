import Foundation
import Testing
import ANEFieldfareServerCore

/// The HTTP/1.1 request parser, fed bytes the way a socket delivers them.
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

    @Test func contentLengthBody() throws {
        let requests = try parse("POST /v1/chat/completions?x=1 HTTP/1.1\r\nHost: a\r\nContent-Type: application/json\r\nContent-Length: 7\r\n\r\n{\"a\":1}")
        let request = try #require(requests.first)
        #expect(requests.count == 1)
        #expect(request.method == "POST")
        #expect(request.path == "/v1/chat/completions")
        #expect(request.target == "/v1/chat/completions?x=1")
        #expect(request.version == .http11)
        #expect(request.headers["content-type"] == "application/json")
        #expect(String(decoding: request.body, as: UTF8.self) == "{\"a\":1}")
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
        #expect(!parser.hasBufferedData)
    }

    @Test func pipelinedRequestsParseInOrder() throws {
        let requests = try parse(
            "GET /health HTTP/1.1\r\n\r\n"
            + "POST /a HTTP/1.1\r\nContent-Length: 2\r\n\r\nhi"
            + "POST /b HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n0\r\n\r\n")
        #expect(requests.map(\.path) == ["/health", "/a", "/b"])
        #expect(requests.map { String(decoding: $0.body, as: UTF8.self) } == ["", "hi", "abc"])
    }

    @Test func incompleteBodyWaitsForMoreData() throws {
        var parser = HTTPRequestParser()
        parser.append(Data("POST /a HTTP/1.1\r\nContent-Length: 10\r\n\r\n12345".utf8))
        #expect(try parser.next() == .needMoreData)
        parser.append(Data("67890".utf8))
        guard case .request(let request) = try parser.next() else {
            Issue.record("expected a request")
            return
        }
        #expect(request.body.count == 10)
    }

    @Test func expectContinueIsAskedForOnceBeforeTheBody() throws {
        var parser = HTTPRequestParser()
        parser.append(Data("POST /a HTTP/1.1\r\nExpect: 100-continue\r\nContent-Length: 3\r\n\r\n".utf8))
        #expect(try parser.next() == .continueExpected)
        #expect(try parser.next() == .needMoreData)
        parser.append(Data("abc".utf8))
        guard case .request = try parser.next() else {
            Issue.record("expected a request")
            return
        }
    }

    @Test func keepAliveFollowsVersionAndConnectionHeader() throws {
        let requests = try parse(
            "GET / HTTP/1.1\r\n\r\n"
            + "GET / HTTP/1.1\r\nConnection: Close\r\n\r\n"
            + "GET / HTTP/1.0\r\n\r\n"
            + "GET / HTTP/1.0\r\nConnection: keep-alive\r\n\r\n")
        #expect(requests.map(\.keepAlive) == [true, false, false, true])
    }

    @Test func absoluteFormTargetKeepsThePath() throws {
        let request = try #require(try parse("GET http://localhost:8080/v1/models HTTP/1.1\r\n\r\n").first)
        #expect(request.path == "/v1/models")
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

    @Test func responsesSetLengthAndConnection() {
        var headers = HTTPHeaders()
        headers.add("Content-Type", "application/json")
        headers.add("Content-Length", "999")
        let response = HTTPResponse(status: 200, headers: headers, body: Data("{}".utf8))
        let text = String(decoding: response.serialized(keepAlive: false), as: UTF8.self)
        #expect(text == "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}")
        let head = String(decoding: response.serialized(keepAlive: true, includeBody: false), as: UTF8.self)
        #expect(head.hasSuffix("Content-Length: 2\r\nConnection: keep-alive\r\n\r\n"))
        let empty = String(decoding: HTTPResponse(status: 204).serialized(keepAlive: true), as: UTF8.self)
        #expect(empty == "HTTP/1.1 204 No Content\r\nConnection: keep-alive\r\n\r\n")
    }

    @Test func chunkedEncodingFramesAndTerminates() {
        #expect(String(decoding: HTTPChunkedEncoding.chunk(Data(String(repeating: "x", count: 26).utf8)), as: UTF8.self)
                == "1a\r\n" + String(repeating: "x", count: 26) + "\r\n")
        #expect(HTTPChunkedEncoding.chunk(Data()).isEmpty)
        #expect(String(decoding: HTTPChunkedEncoding.end, as: UTF8.self) == "0\r\n\r\n")
    }
}
