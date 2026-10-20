import Darwin
import Dispatch
import Foundation

package enum InstallError: Error, Equatable, CustomStringConvertible, Sendable {
    case posix(call: String, errno: Int32)
    case locked(String)
    case alreadyInstalled(String)
    case invalidPartial(String)
    case verificationFailed(String)

    package var description: String {
        switch self {
        case let .posix(call, errno): "\(call): \(unsafe String(cString: strerror(errno)))"
        case let .locked(path): "another install holds \(path)"
        case let .alreadyInstalled(path): "\(path) already exists (pass --overwrite to replace it)"
        case let .invalidPartial(reason): "cannot resume the partial install: \(reason) (pass --restart)"
        case let .verificationFailed(reason): "verification failed: \(reason)"
        }
    }
}

/// The one exit from Swift concurrency for blocking file I/O, as in the
/// runtime: callers `await` work that runs on a GCD queue.
enum BlockingIO {
    // SAFETY: a GCD queue is the point of this type. Work submitted here is
    // `@Sendable` and its result crosses back through a checked continuation.
    private static var queue: DispatchQueue { .global(qos: .userInitiated) }

    static func run<Value: Sendable>(
        _ body: @escaping @Sendable () throws -> Value
    ) async throws -> Value {
        try await withCheckedThrowingContinuation { continuation in
            queue.async { continuation.resume(with: Result(catching: body)) }
        }
    }
}

enum POSIXFile {
    static func read(fd: Int32, range: Range<UInt64>, path: String) throws -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: Int(range.upperBound - range.lowerBound))
        var done = 0
        while done < bytes.count {
            let got = bytes.withUnsafeMutableBytes { raw in
                unsafe pread(fd, raw.baseAddress?.advanced(by: done), raw.count - done,
                             off_t(range.lowerBound) + off_t(done))
            }
            if got < 0, errno == EINTR { continue }
            guard got > 0 else { throw InstallError.posix(call: "pread(\(path))", errno: got < 0 ? errno : EIO) }
            done += got
        }
        return bytes
    }

    static func write(fd: Int32, bytes: [UInt8], at offset: UInt64, path: String) throws {
        var done = 0
        while done < bytes.count {
            let wrote = bytes.withUnsafeBytes { raw in
                unsafe pwrite(fd, raw.baseAddress?.advanced(by: done), raw.count - done,
                              off_t(offset) + off_t(done))
            }
            if wrote < 0, errno == EINTR { continue }
            guard wrote > 0 else { throw InstallError.posix(call: "pwrite(\(path))", errno: errno) }
            done += wrote
        }
    }

    static func sync(fd: Int32, path: String) throws {
        guard fcntl(fd, F_FULLFSYNC) == 0 || fsync(fd) == 0 else {
            throw InstallError.posix(call: "fsync(\(path))", errno: errno)
        }
    }

    static func size(fd: Int32, path: String) throws -> UInt64 {
        var st = stat()
        guard unsafe fstat(fd, &st) == 0 else { throw InstallError.posix(call: "fstat(\(path))", errno: errno) }
        return UInt64(st.st_size)
    }

    /// Writes `data` to `url` through a temporary file and a rename.
    static func atomicWrite(_ data: Data, to url: URL) throws {
        let tmp = url.appendingPathExtension("tmp")
        try data.write(to: tmp)
        let fd = unsafe open(tmp.path, O_RDONLY | O_CLOEXEC)
        guard fd >= 0 else { throw InstallError.posix(call: "open(\(tmp.path))", errno: errno) }
        defer { close(fd) }
        try sync(fd: fd, path: tmp.path)
        guard unsafe rename(tmp.path, url.path) == 0 else {
            throw InstallError.posix(call: "rename(\(tmp.path))", errno: errno)
        }
    }

    static func syncDirectory(_ url: URL) throws {
        let fd = unsafe open(url.path, O_RDONLY | O_DIRECTORY | O_CLOEXEC)
        guard fd >= 0 else { throw InstallError.posix(call: "open(\(url.path))", errno: errno) }
        defer { close(fd) }
        try sync(fd: fd, path: url.path)
    }

    /// SHA-256 of a whole file, read past the page cache.
    static func sha256(path: String) throws -> (size: UInt64, digest: String) {
        let fd = unsafe open(path, O_RDONLY | O_CLOEXEC)
        guard fd >= 0 else { throw InstallError.posix(call: "open(\(path))", errno: errno) }
        defer { close(fd) }
        _ = fcntl(fd, F_NOCACHE, 1)
        let size = try self.size(fd: fd, path: path)
        var hash = ResumableSHA256()
        var offset: UInt64 = 0
        let chunk: UInt64 = 8 * 1024 * 1024
        while offset < size {
            let end = min(size, offset + chunk)
            hash.update(try read(fd: fd, range: offset..<end, path: path)[...])
            offset = end
        }
        return (size, hash.finalize())
    }
}

/// An exclusive `flock` held for the life of the value.
final class InstallLock: Sendable {
    private let fd: Int32

    init(path: String) throws {
        let fd = unsafe open(path, O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard fd >= 0 else { throw InstallError.posix(call: "open(\(path))", errno: errno) }
        guard flock(fd, LOCK_EX | LOCK_NB) == 0 else {
            close(fd)
            throw InstallError.locked(path)
        }
        self.fd = fd
    }

    deinit {
        flock(fd, LOCK_UN)
        close(fd)
    }
}
