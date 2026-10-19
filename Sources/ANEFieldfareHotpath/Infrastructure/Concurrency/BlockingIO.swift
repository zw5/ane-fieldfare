import Dispatch

/// The one exit from Swift concurrency for blocking file I/O.
///
/// An expert read is several `pread`s of a few MB; a first-touch layer open
/// can hash a ~430 MB file. Either would park a cooperative-pool thread for
/// milliseconds to seconds and starve every other task, so this work runs on
/// a GCD queue instead and callers `await` its result.
enum BlockingIO {
    // SAFETY: a GCD queue is the point of this type. Work submitted here is
    // `@Sendable` and its result crosses back through a checked continuation.
    private static let queue = DispatchQueue(
        label: "ane-fieldfare.blocking-io", qos: .userInitiated, attributes: .concurrent)

    static func run<Value: Sendable>(
        _ body: @escaping @Sendable () throws -> Value
    ) async throws -> Value {
        try await withCheckedThrowingContinuation { continuation in
            queue.async { continuation.resume(with: Result(catching: body)) }
        }
    }

    /// Starts best-effort work whose result nobody awaits, such as a
    /// speculative prefetch read. The work must record its own outcome.
    static func detach(_ body: @escaping @Sendable () -> Void) {
        queue.async(execute: body)
    }
}
