// Adapted from Turbo Fieldfare's installer (drumih/turbo-fieldfare 2e9ea50, Apache-2.0).
public import Foundation
import Synchronization

public struct RemoteDownloadSessionPolicy: Sendable, Equatable {
    public var requestTimeoutSeconds: TimeInterval
    public var resourceTimeoutSeconds: TimeInterval
    public var waitsForConnectivity: Bool
    public var maximumConnectionsPerHost: Int
    public var maximumRedirects: Int

    public init(requestTimeoutSeconds: TimeInterval = 300,
                resourceTimeoutSeconds: TimeInterval = 7 * 24 * 60 * 60,
                waitsForConnectivity: Bool = true,
                maximumConnectionsPerHost: Int = 1,
                maximumRedirects: Int = 5) {
        self.requestTimeoutSeconds = requestTimeoutSeconds
        self.resourceTimeoutSeconds = resourceTimeoutSeconds
        self.waitsForConnectivity = waitsForConnectivity
        self.maximumConnectionsPerHost = maximumConnectionsPerHost
        self.maximumRedirects = maximumRedirects
    }
}

public final class RemoteDownloadSession: Sendable {
    public let policy: RemoteDownloadSessionPolicy
    private let protocolClasses: [AnyClass]?

    public init(policy: RemoteDownloadSessionPolicy = RemoteDownloadSessionPolicy(),
                protocolClasses: [AnyClass]? = nil) {
        self.policy = policy
        self.protocolClasses = protocolClasses
    }

    /// A fresh configuration per session, so no session shares mutable
    /// configuration state.
    private func makeConfiguration() -> URLSessionConfiguration {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = policy.requestTimeoutSeconds
        configuration.timeoutIntervalForResource = policy.resourceTimeoutSeconds
        configuration.waitsForConnectivity = policy.waitsForConnectivity
        configuration.httpMaximumConnectionsPerHost = policy.maximumConnectionsPerHost
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.urlCache = nil
        if let protocolClasses {
            configuration.protocolClasses = protocolClasses
        }
        return configuration
    }

    public func response(for request: URLRequest) async throws -> (Data, URLResponse) {
        let configuration = makeConfiguration()
        let delegate = MetadataRedirectDelegate(
            originalRequest: request,
            maximumRedirects: policy.maximumRedirects)
        let session = URLSession(configuration: configuration,
                                 delegate: delegate,
                                 delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        return try await session.data(for: request)
    }

    public func transfer(request: URLRequest,
                         targetPath: String,
                         expectation: RemoteRangeExpectation,
                         progress: @escaping @Sendable (UInt64) -> Void = { _ in }) async throws
        -> RemoteRangeTransferResult {
        try await RemoteRangeTransfer.run(
            configuration: makeConfiguration(),
            request: request,
            targetPath: targetPath,
            expectation: expectation,
            maximumRedirects: policy.maximumRedirects,
            progress: progress)
    }

    var configurationSnapshot: RemoteDownloadSessionConfigurationSnapshot {
        let configuration = makeConfiguration()
        return RemoteDownloadSessionConfigurationSnapshot(
            requestTimeoutSeconds: configuration.timeoutIntervalForRequest,
            resourceTimeoutSeconds: configuration.timeoutIntervalForResource,
            waitsForConnectivity: configuration.waitsForConnectivity,
            maximumConnectionsPerHost: configuration.httpMaximumConnectionsPerHost,
            requestCachePolicy: configuration.requestCachePolicy,
            hasURLCache: configuration.urlCache != nil)
    }
}

struct RemoteDownloadSessionConfigurationSnapshot: Equatable {
    let requestTimeoutSeconds: TimeInterval
    let resourceTimeoutSeconds: TimeInterval
    let waitsForConnectivity: Bool
    let maximumConnectionsPerHost: Int
    let requestCachePolicy: URLRequest.CachePolicy
    let hasURLCache: Bool
}

private final class MetadataRedirectDelegate: NSObject, URLSessionTaskDelegate, Sendable {
    private let policy: Mutex<RemoteMetadataRedirectPolicy>

    init(originalRequest: URLRequest, maximumRedirects: Int) {
        self.policy = Mutex(RemoteMetadataRedirectPolicy(
            originalRequest: originalRequest,
            maximumRedirects: maximumRedirects))
    }

    func urlSession(_ session: URLSession,
                    task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest,
                    completionHandler: @escaping (URLRequest?) -> Void) {
        completionHandler(policy.withLock { $0.request(proposedRequest: request) })
    }
}

struct RemoteMetadataRedirectPolicy {
    private let sourceHost: String?
    private let originalMethod: String?
    private let originalAuthorization: String?
    private let maximumRedirects: Int
    private var redirectCount = 0

    init(originalRequest: URLRequest, maximumRedirects: Int) {
        self.sourceHost = originalRequest.url?.host?.lowercased()
        self.originalMethod = originalRequest.httpMethod
        self.originalAuthorization = originalRequest.value(
            forHTTPHeaderField: "Authorization")
        self.maximumRedirects = maximumRedirects
    }

    mutating func request(proposedRequest: URLRequest) -> URLRequest? {
        redirectCount += 1
        guard redirectCount <= maximumRedirects,
              proposedRequest.url?.scheme?.lowercased() == "https",
              proposedRequest.url?.host?.lowercased() == sourceHost else {
            return nil
        }
        var redirected = proposedRequest
        redirected.httpMethod = originalMethod
        redirected.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
        redirected.setValue(
            originalAuthorization,
            forHTTPHeaderField: "Authorization")
        return redirected
    }
}
