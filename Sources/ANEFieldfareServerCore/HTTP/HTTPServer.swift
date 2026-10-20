import Foundation
import Network
import Synchronization

/// What a request handler did with the connection.
public enum HTTPDisposition: Sendable {
    /// The response was sent and the connection may serve another request
    /// if the request allows it.
    case keepAlive
    /// The response was sent (or the client is gone); close the connection.
    case close
}

/// Answers one parsed request on `connection`. The handler sends its own
/// response, so it can stream.
public typealias HTTPHandler = @Sendable (HTTPRequest, HTTPConnection) async -> HTTPDisposition

/// HTTP/1.1 over Network.framework's `NWListener`.
///
/// Each accepted connection runs in its own task: read and parse a request,
/// hand it to the handler, repeat while the connection persists. A
/// connection idle for `idleTimeout` between requests is closed.
public final class HTTPServer: Sendable {
    public struct Configuration: Sendable {
        public var host: String
        public var port: UInt16
        public var idleTimeout: Duration

        public init(host: String, port: UInt16, idleTimeout: Duration = .seconds(60)) {
            self.host = host
            self.port = port
            self.idleTimeout = idleTimeout
        }
    }

    public struct StartError: Error, CustomStringConvertible {
        public let description: String
    }

    private let configuration: Configuration
    private let handler: HTTPHandler
    private let listener: NWListener
    private let connections = Mutex<[ObjectIdentifier: Task<Void, Never>]>([:])

    public init(configuration: Configuration, handler: @escaping HTTPHandler) throws {
        self.configuration = configuration
        self.handler = handler
        let parameters = NWParameters.tcp
        parameters.allowLocalEndpointReuse = true
        let port = NWEndpoint.Port(rawValue: configuration.port) ?? .any
        parameters.requiredLocalEndpoint = .hostPort(host: NWEndpoint.Host(configuration.host), port: port)
        self.listener = try NWListener(using: parameters)
    }

    /// Starts listening and returns the bound port once the listener is ready.
    public func start() async throws -> UInt16 {
        listener.newConnectionHandler = { [weak self] connection in
            guard let self else {
                connection.cancel()
                return
            }
            let wrapped = HTTPConnection(connection)
            wrapped.start()
            let key = ObjectIdentifier(wrapped)
            // Registered under the lock the task's own removal takes, so the
            // removal cannot run first and leave a finished task behind.
            self.connections.withLock { connections in
                connections[key] = Task {
                    await self.serve(wrapped)
                    _ = self.connections.withLock { $0.removeValue(forKey: key) }
                }
            }
        }
        let started = Mutex(false)
        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<UInt16, any Error>) in
            listener.stateUpdateHandler = { [listener] state in
                let outcome: Result<UInt16, any Error>?
                switch state {
                case .ready:
                    outcome = .success(listener.port?.rawValue ?? 0)
                case .failed(let error):
                    outcome = .failure(StartError(description: "listener failed: \(error)"))
                case .waiting(let error):
                    outcome = .failure(StartError(description: "listener cannot start: \(error)"))
                case .cancelled:
                    outcome = .failure(StartError(description: "listener cancelled"))
                default:
                    outcome = nil
                }
                guard let outcome else { return }
                let first = started.withLock { done -> Bool in
                    defer { done = true }
                    return !done
                }
                if first { continuation.resume(with: outcome) }
            }
            listener.start(queue: NetworkQueue.shared)
        }
    }

    /// Stops accepting and cancels every open connection's task.
    public func stop() {
        listener.cancel()
        let tasks = connections.withLock { connections -> [Task<Void, Never>] in
            defer { connections.removeAll() }
            return Array(connections.values)
        }
        for task in tasks { task.cancel() }
    }

    private func serve(_ connection: HTTPConnection) async {
        var parser = HTTPRequestParser()
        while !Task.isCancelled {
            let request: HTTPRequest
            do {
                guard let next = try await nextRequest(&parser, on: connection) else {
                    connection.cancel()
                    return
                }
                request = next
            } catch let error as HTTPParseError {
                let response = OpenAIErrorResponse.response(APIError(status: error.status, message: error.message))
                await connection.close(after: response.serialized(keepAlive: false))
                return
            } catch {
                connection.cancel()
                return
            }
            let disposition = await handler(request, connection)
            guard case .keepAlive = disposition, request.keepAlive else {
                await connection.close()
                return
            }
        }
        connection.cancel()
    }

    /// The next request, or nil when the connection ended or sat idle.
    private func nextRequest(_ parser: inout HTTPRequestParser,
                             on connection: HTTPConnection) async throws -> HTTPRequest? {
        while true {
            switch try parser.next() {
            case .request(let request):
                return request
            case .continueExpected:
                try await connection.send(Data("HTTP/1.1 100 Continue\r\n\r\n".utf8))
            case .needMoreData:
                let timeout = configuration.idleTimeout
                let data = try await withThrowingTaskGroup(of: Data?.self) { group in
                    group.addTask { try await connection.read() }
                    group.addTask {
                        try await Task.sleep(for: timeout)
                        return nil
                    }
                    defer { group.cancelAll() }
                    return try await group.next() ?? nil
                }
                guard let data, !data.isEmpty else {
                    // Idle timeout or the peer finished sending. A request
                    // cut off part way gets no answer: the client is gone.
                    return nil
                }
                parser.append(data)
            }
        }
    }
}
