import Darwin
public import Foundation
import Synchronization

public struct RemoteRangeExpectation: Sendable, Equatable {
    public let filename: String
    public let offset: UInt64
    public let length: UInt64
    public let totalSize: UInt64
    public let xetHash: String?

    public init(filename: String,
                offset: UInt64,
                length: UInt64,
                totalSize: UInt64,
                xetHash: String?) {
        self.filename = filename
        self.offset = offset
        self.length = length
        self.totalSize = totalSize
        self.xetHash = xetHash
    }
}

public struct RemoteRangeTransferResult: Sendable, Equatable {
    public let path: String
    public let byteCount: UInt64
}

public enum RemoteRangeTransfer {
    public static func run(configuration: URLSessionConfiguration,
                           request: URLRequest,
                           targetPath: String,
                           expectation: RemoteRangeExpectation,
                           maximumRedirects: Int,
                           progress: @escaping @Sendable (UInt64) -> Void = { _ in })
        async throws -> RemoteRangeTransferResult {
        try Posix.mkdirP((targetPath as NSString).deletingLastPathComponent)
        let delegate = try RemoteRangeTransferDelegate(
            targetPath: targetPath,
            expectation: expectation,
            maximumRedirects: maximumRedirects,
            originalRequest: request,
            progress: progress)
        let queue = OperationQueue()
        queue.maxConcurrentOperationCount = 1
        queue.qualityOfService = .utility
        let session = URLSession(configuration: configuration,
                                 delegate: delegate,
                                 delegateQueue: queue)
        return try await withTaskCancellationHandler {
            defer { session.finishTasksAndInvalidate() }
            return try await delegate.start(session: session, request: request)
        } onCancel: {
            delegate.cancel()
        }
    }
}

private final class RemoteRangeTransferDelegate: NSObject, URLSessionDataDelegate, Sendable {
    /// Everything a callback, `start` or `cancel` may change. `start` and
    /// `cancel` run on the caller's threads while callbacks run on the
    /// session's delegate queue, so all of it lives behind one lock.
    private struct State: Sendable {
        var continuation: CheckedContinuation<RemoteRangeTransferResult, any Error>?
        var task: URLSessionDataTask?
        var terminal = false
        var rejection: (any Error)?
        var receivedBytes: UInt64 = 0
        var redirectPolicy: RemoteRedirectPolicy
        var responseAccepted = false
        var lastProgressNanoseconds: UInt64?
    }

    private let targetPath: String
    private let expectation: RemoteRangeExpectation
    private let fd: Int32
    private let progress: @Sendable (UInt64) -> Void
    private let state: Mutex<State>

    init(targetPath: String,
         expectation: RemoteRangeExpectation,
         maximumRedirects: Int,
         originalRequest: URLRequest,
        progress: @escaping @Sendable (UInt64) -> Void) throws {
        self.targetPath = targetPath
        self.expectation = expectation
        self.state = Mutex(State(redirectPolicy: RemoteRedirectPolicy(
            originalRequest: originalRequest,
            maximumRedirects: maximumRedirects)))
        self.progress = progress
        let fd = unsafe open(targetPath, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard fd >= 0 else {
            throw RepackError.fileOpenFailed(path: targetPath, errno: errno)
        }
        self.fd = fd
    }

    func start(session: URLSession, request: URLRequest) async throws -> RemoteRangeTransferResult {
        try await withCheckedThrowingContinuation { continuation in
            let task = session.dataTask(with: request)
            let shouldCancel = state.withLock { state in
                state.continuation = continuation
                state.task = task
                return state.terminal || state.rejection != nil
            }
            if shouldCancel {
                task.cancel()
            } else {
                task.resume()
            }
        }
    }

    func cancel() {
        let task = state.withLock { state in
            if state.rejection == nil {
                state.rejection = CancellationError()
            }
            return state.task
        }
        task?.cancel()
    }

    func urlSession(_ session: URLSession,
                    task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest,
                    completionHandler: @escaping (URLRequest?) -> Void) {
        do {
            completionHandler(try state.withLock { state in
                try state.redirectPolicy.request(response: response, proposedRequest: request)
            })
        } catch {
            reject(error)
            completionHandler(nil)
        }
    }

    func urlSession(_ session: URLSession,
                    dataTask: URLSessionDataTask,
                    didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        do {
            if let cancellation = rejectionSnapshot() {
                throw cancellation
            }
            guard let http = response as? HTTPURLResponse else {
                throw RepackError.remoteProtocolInvalid(
                    detail: "missing HTTP response for range \(expectation.filename)")
            }
            try validate(http)
            state.withLock { $0.responseAccepted = true }
            completionHandler(.allow)
        } catch {
            reject(error)
            completionHandler(.cancel)
        }
    }

    func urlSession(_ session: URLSession,
                    dataTask: URLSessionDataTask,
                    didReceive data: Data) {
        let offset: UInt64? = state.withLock { state in
            state.responseAccepted && state.rejection == nil ? state.receivedBytes : nil
        }
        guard let offset else { return }
        let attempted = offset + UInt64(data.count)
        guard attempted <= expectation.length else {
            reject(RepackError.remoteBodyExceeded(
                path: expectation.filename,
                limit: expectation.length,
                attempted: attempted))
            dataTask.cancel()
            return
        }
        do {
            try unsafe data.withUnsafeBytes { raw in
                guard let base = raw.baseAddress else { return }
                try unsafe Posix.pwriteAll(fd: fd,
                                    path: targetPath,
                                    buf: base,
                                    count: raw.count,
                                    offset: offset)
            }
            let now = DispatchTime.now().uptimeNanoseconds
            let report: UInt64? = state.withLock { state in
                state.receivedBytes = attempted
                let force = attempted == expectation.length
                guard force
                    || state.lastProgressNanoseconds == nil
                    || now &- (state.lastProgressNanoseconds ?? now) >= 1_000_000_000 else {
                    return nil
                }
                state.lastProgressNanoseconds = now
                return attempted
            }
            if let report { progress(report) }
        } catch {
            reject(error)
            dataTask.cancel()
        }
    }

    func urlSession(_ session: URLSession,
                    task: URLSessionTask,
                    didCompleteWithError error: (any Error)?) {
        let (rejection, responseAccepted, receivedBytes) = state.withLock {
            ($0.rejection, $0.responseAccepted, $0.receivedBytes)
        }
        let result: Result<RemoteRangeTransferResult, any Error>
        if let rejection {
            result = .failure(rejection)
        } else if let error {
            result = .failure(error)
        } else if !responseAccepted {
            result = .failure(RepackError.remoteProtocolInvalid(
                detail: "range \(expectation.filename) completed without an accepted response"))
        } else if receivedBytes != expectation.length {
            result = .failure(RepackError.remoteBodyTruncated(
                path: expectation.filename,
                expected: expectation.length,
                actual: receivedBytes))
        } else {
            result = .success(RemoteRangeTransferResult(
                path: targetPath,
                byteCount: receivedBytes))
        }
        finish(result)
    }

    private func validate(_ response: HTTPURLResponse) throws {
        guard response.statusCode == 206 else {
            throw RepackError.remoteHTTPResponse(
                url: redactRemoteURL(response.url),
                status: response.statusCode,
                retryAfter: remoteHeader(response, "Retry-After"))
        }
        guard (remoteHeader(response, "Content-Encoding") ?? "identity").lowercased() == "identity" else {
            throw RepackError.remoteProtocolInvalid(
                detail: "compressed range response for \(expectation.filename)")
        }
        guard let contentLength = remoteHeader(response, "Content-Length"),
              UInt64(contentLength) == expectation.length else {
            throw RepackError.remoteProtocolInvalid(
                detail: "wrong Content-Length for \(expectation.filename)")
        }
        guard let range = remoteHeader(response, "Content-Range") else {
            throw RepackError.remoteProtocolInvalid(
                detail: "missing Content-Range for \(expectation.filename)")
        }
        let prefix = "bytes "
        let end = expectation.offset + expectation.length - 1
        guard range.hasPrefix(prefix) else {
            throw RepackError.remoteProtocolInvalid(detail: "malformed Content-Range \(range)")
        }
        let parts = range.dropFirst(prefix.count).split(separator: "/", maxSplits: 1)
        let bounds = parts.first?.split(separator: "-", maxSplits: 1) ?? []
        guard parts.count == 2,
              bounds.count == 2,
              UInt64(bounds[0]) == expectation.offset,
              UInt64(bounds[1]) == end,
              UInt64(parts[1]) == expectation.totalSize else {
            throw RepackError.remoteProtocolInvalid(detail: "wrong Content-Range \(range)")
        }
        if let xetHash = expectation.xetHash {
            guard normalizedStrongETag(remoteHeader(response, "ETag")) == xetHash.lowercased() else {
                throw RepackError.remoteProtocolInvalid(
                    detail: "final ranged validator differs for \(expectation.filename)")
            }
        }
    }

    private func reject(_ error: any Error) {
        state.withLock { state in
            if state.rejection == nil {
                state.rejection = error
            }
        }
    }

    private func rejectionSnapshot() -> (any Error)? {
        state.withLock { $0.rejection }
    }

    private func finish(_ result: Result<RemoteRangeTransferResult, any Error>) {
        let claimed: CheckedContinuation<RemoteRangeTransferResult, any Error>?? = state.withLock { state in
            guard !state.terminal else { return nil }
            state.terminal = true
            defer {
                state.continuation = nil
                state.task = nil
            }
            return .some(state.continuation)
        }
        guard let continuation = claimed else { return }

        close(fd)
        if case .failure = result {
            try? FileManager.default.removeItem(atPath: targetPath)
        }
        continuation?.resume(with: result)
    }
}

struct RemoteRedirectPolicy: Sendable {
    private let originalRange: String?
    private let originalAuthorization: String?
    private let sourceHost: String?
    private let maximumRedirects: Int
    private var redirectCount = 0
    private var authorizationMayBeForwarded = true

    init(originalRequest: URLRequest, maximumRedirects: Int) {
        self.originalRange = originalRequest.value(forHTTPHeaderField: "Range")
        self.originalAuthorization = originalRequest.value(
            forHTTPHeaderField: "Authorization")
        self.sourceHost = originalRequest.url?.host?.lowercased()
        self.maximumRedirects = maximumRedirects
    }

    mutating func request(
        response: HTTPURLResponse,
        proposedRequest: URLRequest
    ) throws -> URLRequest {
        redirectCount += 1
        guard redirectCount <= maximumRedirects else {
            throw RepackError.remoteRedirectRejected(
                url: redactRemoteURL(response.url),
                detail: "more than \(maximumRedirects) redirects")
        }
        guard proposedRequest.url?.scheme?.lowercased() == "https" else {
            throw RepackError.remoteRedirectRejected(
                url: redactRemoteURL(proposedRequest.url),
                detail: "only HTTPS redirects are allowed")
        }
        let targetHost = proposedRequest.url?.host?.lowercased()
        if targetHost != sourceHost {
            authorizationMayBeForwarded = false
        }
        var redirected = proposedRequest
        redirected.httpMethod = "GET"
        redirected.setValue(originalRange, forHTTPHeaderField: "Range")
        redirected.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
        redirected.setValue(
            authorizationMayBeForwarded && targetHost == sourceHost
                ? originalAuthorization
                : nil,
            forHTTPHeaderField: "Authorization")
        return redirected
    }
}

func remoteHeader(_ response: HTTPURLResponse, _ name: String) -> String? {
    for (key, value) in response.allHeaderFields
    where String(describing: key).caseInsensitiveCompare(name) == .orderedSame {
        return String(describing: value)
    }
    return nil
}

func normalizedStrongETag(_ value: String?) -> String? {
    guard var value = value?.trimmingCharacters(in: .whitespacesAndNewlines),
          !value.isEmpty,
          !value.lowercased().hasPrefix("w/") else {
        return nil
    }
    if value.hasPrefix("\""), value.hasSuffix("\""), value.count >= 2 {
        value.removeFirst()
        value.removeLast()
    }
    return value.lowercased()
}

func redactRemoteURL(_ url: URL?) -> String {
    guard let url else { return "<unknown>" }
    var components = URLComponents(url: url, resolvingAgainstBaseURL: false)
    components?.query = nil
    components?.fragment = nil
    return components?.url?.absoluteString ?? "<redacted>"
}
