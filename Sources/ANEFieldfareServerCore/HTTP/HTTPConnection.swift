public import Foundation
public import Network
import Synchronization

/// The queue Network.framework delivers listener and connection callbacks on.
enum NetworkQueue {
    // SAFETY: Network.framework requires a dispatch queue for its callbacks.
    // Every callback registered on it only locks a `Mutex`, resumes a checked
    // continuation or starts a `Task`; no state is owned by the queue itself.
    static let shared = DispatchQueue(label: "ane-fieldfare.server.network", qos: .userInitiated)
}

public struct ConnectionClosed: Error, Sendable {}

/// One accepted TCP connection, bridged to Swift concurrency.
///
/// Reads are pumped continuously into a bounded buffer: one receive is kept
/// outstanding while fewer than `highWater` bytes wait unread. That is what
/// lets `waitUntilClosed()` notice a client that hangs up while its request is
/// queued or generating, without anyone reading. All mutable state is in one
/// `Mutex`; waiters are checked continuations, removed and resumed on
/// cancellation, so every wait here is cancellable.
public final class HTTPConnection: Sendable {
    private struct Waiter {
        let id: UInt64
        let continuation: CheckedContinuation<Void, Never>
    }

    private struct State {
        var buffer = Data()
        /// The peer finished sending, or the connection failed.
        var closed = false
        var receiving = false
        var waiters: [Waiter] = []
        var nextWaiterID: UInt64 = 0
    }

    private let connection: NWConnection
    private let state = Mutex(State())
    private static let highWater = 1 << 20

    public init(_ connection: NWConnection) {
        self.connection = connection
    }

    func start() {
        connection.stateUpdateHandler = { [weak self] state in
            switch state {
            case .failed, .cancelled:
                self?.markClosed()
            case .ready:
                self?.pump()
            default:
                break
            }
        }
        connection.start(queue: NetworkQueue.shared)
    }

    // MARK: - Reading

    private func pump() {
        let start = state.withLock { state -> Bool in
            guard !state.receiving, !state.closed, state.buffer.count < Self.highWater else { return false }
            state.receiving = true
            return true
        }
        guard start else { return }
        connection.receive(minimumIncompleteLength: 1, maximumLength: 64 * 1024) {
            [weak self] content, _, isComplete, error in
            guard let self else { return }
            let waiters = self.state.withLock { state -> [Waiter] in
                state.receiving = false
                if let content, !content.isEmpty { state.buffer.append(content) }
                if isComplete || error != nil { state.closed = true }
                defer { state.waiters.removeAll() }
                return state.waiters
            }
            for waiter in waiters { waiter.continuation.resume() }
            self.pump()
        }
    }

    private func markClosed() {
        let waiters = state.withLock { state -> [Waiter] in
            state.closed = true
            defer { state.waiters.removeAll() }
            return state.waiters
        }
        for waiter in waiters { waiter.continuation.resume() }
    }

    /// Suspends until `ready` holds or the task is cancelled.
    private func wait(until ready: (State) -> Bool) async {
        let id = state.withLock { state -> UInt64 in
            state.nextWaiterID += 1
            return state.nextWaiterID
        }
        await withTaskCancellationHandler {
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                let resumeNow = state.withLock { state -> Bool in
                    if ready(state) || Task.isCancelled { return true }
                    state.waiters.append(Waiter(id: id, continuation: continuation))
                    return false
                }
                if resumeNow { continuation.resume() }
            }
        } onCancel: {
            let waiter = state.withLock { state -> Waiter? in
                guard let index = state.waiters.firstIndex(where: { $0.id == id }) else { return nil }
                return state.waiters.remove(at: index)
            }
            waiter?.continuation.resume()
        }
    }

    /// The bytes received since the last read; empty once the peer has
    /// finished sending and everything was read. Throws `CancellationError`.
    public func read() async throws -> Data {
        while true {
            try Task.checkCancellation()
            let taken = state.withLock { state -> Data? in
                if !state.buffer.isEmpty {
                    defer { state.buffer.removeAll(keepingCapacity: true) }
                    return state.buffer
                }
                return state.closed ? Data() : nil
            }
            if let taken {
                pump()
                return taken
            }
            pump()
            await wait { !$0.buffer.isEmpty || $0.closed }
        }
    }

    /// Returns once the peer has closed or reset the connection, or when the
    /// task is cancelled.
    public func waitUntilClosed() async {
        pump()
        await wait { $0.closed }
    }

    public var isClosed: Bool { state.withLock { $0.closed } }

    // MARK: - Writing

    /// Sends `data` and waits until Network.framework has taken it. Throws
    /// `ConnectionClosed` when the connection failed.
    public func send(_ data: Data) async throws {
        guard !data.isEmpty else { return }
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
            connection.send(content: data, completion: .contentProcessed { error in
                if error != nil {
                    continuation.resume(throwing: ConnectionClosed())
                } else {
                    continuation.resume()
                }
            })
        }
    }

    /// Sends whatever is left, then the end of the stream, and cancels the
    /// connection.
    public func close(after data: Data = Data()) async {
        if !data.isEmpty { try? await send(data) }
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            connection.send(content: nil, contentContext: .finalMessage, isComplete: true,
                            completion: .contentProcessed { _ in continuation.resume() })
        }
        connection.cancel()
    }

    /// Drops the connection without a graceful end.
    public func cancel() {
        connection.cancel()
    }
}
