import Synchronization

/// One generation at a time, first come first served, with a bounded line.
///
/// `acquire()` returns when the caller owns the gate; it throws `Busy` at
/// once when `maxWaiting` callers are already waiting, and
/// `CancellationError` when the waiting task is cancelled (its place in line
/// is given up). The owner calls `release()` exactly once.
///
/// All state is in one `Mutex`. A waiter's checked continuation is resumed by
/// exactly one of `release()` (it now owns the gate), the cancellation handler
/// (it left the line), or `acquire` itself (immediate grant or refusal).
public final class GenerationGate: Sendable {
    public struct Busy: Error, Sendable {}

    public struct Status: Sendable, Equatable {
        public let running: Bool
        public let waiting: Int
    }

    private struct Waiter {
        let id: UInt64
        let continuation: CheckedContinuation<Void, any Error>
    }

    private struct State {
        var running = false
        var waiters: [Waiter] = []
        var nextID: UInt64 = 0
    }

    private enum Admission {
        case granted
        case refused(any Error)
        case queued
    }

    public let maxWaiting: Int
    private let state = Mutex(State())

    public init(maxWaiting: Int) {
        self.maxWaiting = max(0, maxWaiting)
    }

    public var status: Status {
        state.withLock { Status(running: $0.running, waiting: $0.waiters.count) }
    }

    public func acquire() async throws {
        let id = state.withLock { state -> UInt64 in
            state.nextID += 1
            return state.nextID
        }
        let maxWaiting = maxWaiting
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
                let admission = state.withLock { state -> Admission in
                    // The handler may already have run and found nothing to
                    // remove; the flag is set before it runs.
                    if Task.isCancelled { return .refused(CancellationError()) }
                    if !state.running, state.waiters.isEmpty {
                        state.running = true
                        return .granted
                    }
                    guard state.waiters.count < maxWaiting else { return .refused(Busy()) }
                    state.waiters.append(Waiter(id: id, continuation: continuation))
                    return .queued
                }
                switch admission {
                case .granted: continuation.resume()
                case .refused(let error): continuation.resume(throwing: error)
                case .queued: break
                }
            }
        } onCancel: {
            let waiter = state.withLock { state -> Waiter? in
                guard let index = state.waiters.firstIndex(where: { $0.id == id }) else { return nil }
                return state.waiters.remove(at: index)
            }
            waiter?.continuation.resume(throwing: CancellationError())
        }
    }

    /// Hands the gate to the longest waiter, or opens it.
    public func release() {
        let next = state.withLock { state -> Waiter? in
            guard !state.waiters.isEmpty else {
                state.running = false
                return nil
            }
            return state.waiters.removeFirst()
        }
        next?.continuation.resume()
    }
}
