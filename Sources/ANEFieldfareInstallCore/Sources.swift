import Darwin
package import Foundation

/// A file pinned to an exact repository revision, size and SHA-256.
package struct PinnedFile: Sendable, Equatable {
    package let repo: String
    package let revision: String
    package let path: String
    package let size: UInt64
    package let sha256: String

    package var url: URL? {
        URL(string: "https://huggingface.co/\(repo)/resolve/\(revision)/\(path)")
    }
}

/// Google's quantization-aware-trained Q4_0 release of Gemma 4 26B-A4B, and
/// the tokenizer files from the same model's QAT release.
package enum SupportedModel {
    package static let modelID = "google/gemma-4-26B-A4B-it-qat-q4_0"
    package static let directoryName = "gemma-4-26b-a4b-it-qat.fieldfare"

    static let ggufRepo = "google/gemma-4-26B-A4B-it-qat-q4_0-gguf"
    static let ggufRevision = "d1c082be9cf3c8a514acf63b8761f4b41935842e"
    static let tokenizerRepo = "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized"
    static let tokenizerRevision = "f1e06dc520982d9b9edd76859fdb7ab209449949"

    package static let gguf = PinnedFile(
        repo: ggufRepo, revision: ggufRevision, path: "gemma-4-26B_q4_0-it.gguf",
        size: 14_439_363_584,
        sha256: "3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d")

    package static let mmproj = PinnedFile(
        repo: ggufRepo, revision: ggufRevision, path: "gemma-4-26B-it-mmproj.gguf",
        size: 1_194_828_160,
        sha256: "a359953a076b877db30c31dbbb4c6d93b4a6e017ee5db5784247e4d4c0dd4f3b")

    package static let tokenizer: [PinnedFile] = [
        PinnedFile(repo: tokenizerRepo, revision: tokenizerRevision, path: "tokenizer.json",
                   size: 32_169_626,
                   sha256: "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"),
        PinnedFile(repo: tokenizerRepo, revision: tokenizerRevision, path: "tokenizer_config.json",
                   size: 3_729,
                   sha256: "3ab5c7b94dc97d65ca7064496fa69b88ff875378e1cb7ee3e43070c3a8170999"),
        PinnedFile(repo: tokenizerRepo, revision: tokenizerRevision, path: "chat_template.jinja",
                   size: 18_683,
                   sha256: "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4"),
        PinnedFile(repo: tokenizerRepo, revision: tokenizerRevision, path: "config.json",
                   size: 3_810,
                   sha256: "ece3392c07744553f4e8bb2b5905bc68b0a7d7ab2927133bb621875b8e4a3289"),
    ]
}

/// Random-access reads of one source file.
package protocol ByteSource: Sendable {
    var size: UInt64 { get }
    /// How many reads may run at once.
    var concurrency: Int { get }
    func read(_ range: Range<UInt64>) async throws -> [UInt8]
}

package enum SourceError: Error, Equatable, CustomStringConvertible, Sendable {
    case sizeMismatch(path: String, expected: UInt64, actual: UInt64)
    case http(path: String, detail: String)
    case digestMismatch(path: String, expected: String, actual: String)

    package var description: String {
        switch self {
        case let .sizeMismatch(path, expected, actual): "\(path): \(actual) bytes, expected \(expected)"
        case let .http(path, detail): "\(path): \(detail)"
        case let .digestMismatch(path, expected, actual): "\(path): sha256 \(actual), expected \(expected)"
        }
    }
}

package struct LocalFileSource: ByteSource {
    package let path: String
    package let size: UInt64
    package var concurrency: Int { 2 }

    package init(path: String) throws {
        var st = stat()
        guard unsafe stat(path, &st) == 0, (st.st_mode & S_IFMT) == S_IFREG else {
            throw InstallError.posix(call: "stat(\(path))", errno: errno)
        }
        self.path = path
        self.size = UInt64(st.st_size)
    }

    package func read(_ range: Range<UInt64>) async throws -> [UInt8] {
        let path = path
        return try await BlockingIO.run {
            let fd = unsafe open(path, O_RDONLY | O_CLOEXEC)
            guard fd >= 0 else { throw InstallError.posix(call: "open(\(path))", errno: errno) }
            defer { close(fd) }
            return try POSIXFile.read(fd: fd, range: range, path: path)
        }
    }
}

/// Ranged GETs against a pinned Hugging Face URL. Each range must come back
/// as a 206 of exactly the requested length; failures retry with backoff.
package struct HuggingFaceSource: ByteSource {
    package let file: PinnedFile
    package var size: UInt64 { file.size }
    package let concurrency: Int
    private let session: URLSession

    package init(file: PinnedFile, concurrency: Int = 8) {
        self.file = file
        self.concurrency = concurrency
        let configuration = URLSessionConfiguration.ephemeral
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.urlCache = nil
        configuration.timeoutIntervalForRequest = 120
        configuration.httpMaximumConnectionsPerHost = concurrency
        self.session = URLSession(configuration: configuration)
    }

    package func read(_ range: Range<UInt64>) async throws -> [UInt8] {
        guard let url = file.url else { throw SourceError.http(path: file.path, detail: "bad URL") }
        var request = URLRequest(url: url)
        request.setValue("bytes=\(range.lowerBound)-\(range.upperBound - 1)", forHTTPHeaderField: "Range")
        request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
        var delay: UInt64 = 1
        for attempt in 1...8 {
            do {
                let (data, response) = try await session.data(for: request)
                guard let http = response as? HTTPURLResponse else {
                    throw SourceError.http(path: file.path, detail: "not an HTTP response")
                }
                guard http.statusCode == 206, UInt64(data.count) == range.upperBound - range.lowerBound else {
                    throw SourceError.http(path: file.path,
                                           detail: "HTTP \(http.statusCode), \(data.count) bytes for \(range)")
                }
                return [UInt8](data)
            } catch {
                if attempt == 8 || error is CancellationError { throw error }
            }
            try await Task.sleep(nanoseconds: delay * 1_000_000_000)
            delay = min(delay * 2, 60)
        }
        throw SourceError.http(path: file.path, detail: "unreachable")
    }
}

/// Downloads one small pinned file whole and checks its digest.
package func fetchPinned(_ file: PinnedFile) async throws -> [UInt8] {
    let bytes = try await HuggingFaceSource(file: file, concurrency: 1).read(0..<file.size)
    var hash = ResumableSHA256()
    hash.update(bytes[...])
    let digest = hash.finalize()
    guard digest == file.sha256 else {
        throw SourceError.digestMismatch(path: file.path, expected: file.sha256, actual: digest)
    }
    return bytes
}
