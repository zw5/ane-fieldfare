package import ANEFieldfareModelFormat
import Darwin
package import Foundation

package struct InstallOptions: Sendable {
    package enum Tokenizer: Sendable {
        case huggingFace
        /// A directory holding the pinned tokenizer files; each is still checked.
        case directory(URL)
    }

    /// The `.fieldfare` directory to create.
    package var destination: URL
    package var source: any ByteSource
    /// What the source must hash to, and its provenance.
    package var sourceFile: PinnedFile
    package var tokenizer: Tokenizer
    package var tokenizerFiles: [PinnedFile]
    package var modelID: String
    package var overwrite = false
    package var restart = false
    package var chunkBytes: UInt64 = 16 * 1024 * 1024
    package var checkpointBytes: UInt64 = 256 * 1024 * 1024
    /// First read of the source; grown until the GGUF header fits.
    package var headerPrefixBytes: UInt64 = 32 * 1024 * 1024
    package var log: @Sendable (String) -> Void = { _ in }

    package init(destination: URL, source: any ByteSource, sourceFile: PinnedFile,
                 tokenizer: Tokenizer, tokenizerFiles: [PinnedFile], modelID: String) {
        self.destination = destination
        self.source = source
        self.sourceFile = sourceFile
        self.tokenizer = tokenizer
        self.tokenizerFiles = tokenizerFiles
        self.modelID = modelID
    }
}

/// Builds a `.fieldfare` directory from a GGUF source in one ordered pass.
///
/// Everything is written under `<destination>.partial/`; the directory is
/// renamed into place only after the source matched its SHA-256 and every
/// output was hashed into the manifest. An interrupted install resumes from
/// its last checkpoint: outputs are synced, then the checkpoint records the
/// source offset and the running source hash.
package enum Installer {
    static let headerFile = ".source-header"
    static let checkpointFile = ".checkpoint.json"
    static let headerMaxBytes: UInt64 = 256 * 1024 * 1024

    struct Checkpoint: Codable {
        let sourceSHA256: String
        let headerSHA256: String
        let offset: UInt64
        let hash: Data
    }

    @discardableResult
    package static func install(_ options: InstallOptions) async throws -> FieldfareManifest {
        let destination = options.destination.standardizedFileURL
        let fm = FileManager.default
        try fm.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        let lock = try InstallLock(path: destination.path + ".install.lock")
        defer { withExtendedLifetime(lock) {} }

        if fm.fileExists(atPath: destination.path), !options.overwrite {
            throw InstallError.alreadyInstalled(destination.path)
        }
        guard options.source.size == options.sourceFile.size else {
            throw SourceError.sizeMismatch(path: options.sourceFile.path, expected: options.sourceFile.size,
                                           actual: options.source.size)
        }
        let partial = URL(fileURLWithPath: destination.path + ".partial", isDirectory: true)
        if options.restart, fm.fileExists(atPath: partial.path) { try fm.removeItem(at: partial) }
        try fm.createDirectory(at: partial, withIntermediateDirectories: true)

        let (prefix, header) = try await sourceHeader(options, partial: partial)
        let headerDigest = digest(prefix[...])
        let plan = try Gemma4Plan.plan(header)
        let checkpoint = try loadCheckpoint(partial: partial, options: options, headerDigest: headerDigest)
        let outputs = try OutputFiles(root: partial, files: plan.files, resuming: checkpoint != nil)
        defer { outputs.close() }

        var converter = StreamingConverter(segments: plan.segments, resumingAt: checkpoint)
        if converter.offset == 0 {
            converter = try await consume(converter, prefix, at: 0, outputs: outputs)
        } else {
            options.log("resuming at \(gigabytes(converter.offset)) of \(gigabytes(options.source.size))")
        }

        var saved = converter.checkpoint.offset
        let start = Date()
        let startOffset = converter.offset
        var lastLog = start
        try await stream(options.source, from: converter.offset, chunk: options.chunkBytes) { bytes, offset in
            converter = try await consume(converter, bytes, at: offset, outputs: outputs)
            if converter.checkpoint.offset - saved >= options.checkpointBytes {
                try await outputs.sync()
                try saveCheckpoint(converter.checkpoint, partial: partial, options: options, headerDigest: headerDigest)
                saved = converter.checkpoint.offset
            }
            if Date().timeIntervalSince(lastLog) >= 15 {
                lastLog = Date()
                let rate = Double(converter.offset - startOffset) / max(1, lastLog.timeIntervalSince(start)) / 1e6
                options.log("\(gigabytes(converter.offset)) of \(gigabytes(options.source.size)), \(rate.formatted(.number.precision(.fractionLength(1)))) MB/s")
            }
        }
        let sourceDigest = try converter.finish(sourceSize: options.source.size)
        guard sourceDigest == options.sourceFile.sha256 else {
            try? fm.removeItem(at: partial)
            throw SourceError.digestMismatch(path: options.sourceFile.path, expected: options.sourceFile.sha256,
                                             actual: sourceDigest)
        }
        try await outputs.sync()
        options.log("source verified (sha256 \(sourceDigest.prefix(12))…)")

        try await installTokenizer(options, partial: partial)
        let manifest = FieldfareManifest(
            modelID: options.modelID,
            source: FieldfareSource(files: ([options.sourceFile] + options.tokenizerFiles).map {
                .init(repo: $0.repo, revision: $0.revision, path: $0.path, size: $0.size, sha256: $0.sha256)
            }),
            arch: plan.arch,
            files: try await hashFiles(plan.files.map(\.path)
                + options.tokenizerFiles.map { "\(FieldfareFormat.tokenizerDirectory)/\($0.path)" },
                root: partial),
            resident: plan.resident,
            experts: plan.experts,
            vision: nil)
        try POSIXFile.atomicWrite(try FieldfareManifestCodec.encode(manifest),
                                  to: partial.appendingPathComponent(FieldfareFormat.manifestPath))
        try? fm.removeItem(at: partial.appendingPathComponent(checkpointFile))
        try? fm.removeItem(at: partial.appendingPathComponent(headerFile))
        try POSIXFile.syncDirectory(partial)
        try commit(partial: partial, destination: destination)
        options.log("installed \(destination.path)")
        return manifest
    }

    /// Re-hashes every file of an installed model against its manifest.
    package static func verify(_ directory: URL) async throws -> FieldfareManifest {
        let root = directory.standardizedFileURL
        let data = try Data(contentsOf: root.appendingPathComponent(FieldfareFormat.manifestPath))
        let manifest = try FieldfareManifestCodec.decode(data)
        let actual = try await hashFiles(Array(manifest.files.keys), root: root)
        for (path, expected) in manifest.files where actual[path] != expected {
            throw InstallError.verificationFailed("\(path) does not match the manifest")
        }
        return manifest
    }

    // MARK: - Steps

    private static func sourceHeader(_ options: InstallOptions, partial: URL) async throws
        -> (prefix: [UInt8], header: GGUFHeader) {
        let url = partial.appendingPathComponent(headerFile)
        if let saved = try? Data(contentsOf: url),
           let header = try? GGUFReader.parse(saved, fileSize: options.source.size) {
            return ([UInt8](saved), header)
        }
        var length = min(options.headerPrefixBytes, options.source.size)
        while true {
            let prefix = try await options.source.read(0..<length)
            do {
                let header = try GGUFReader.parse(Data(prefix), fileSize: options.source.size)
                try POSIXFile.atomicWrite(Data(prefix), to: url)
                return (prefix, header)
            } catch let GGUFError.needMoreBytes(needed) where needed <= headerMaxBytes && length < options.source.size {
                length = min(max(needed, length * 2), options.source.size)
            }
        }
    }

    private static func loadCheckpoint(partial: URL, options: InstallOptions,
                                       headerDigest: String) throws -> ConversionCheckpoint? {
        let url = partial.appendingPathComponent(checkpointFile)
        guard let data = try? Data(contentsOf: url) else { return nil }
        guard let saved = try? JSONDecoder().decode(Checkpoint.self, from: data),
              saved.sourceSHA256 == options.sourceFile.sha256, saved.headerSHA256 == headerDigest,
              saved.offset <= options.source.size, let hash = ResumableSHA256(state: saved.hash) else {
            throw InstallError.invalidPartial("checkpoint does not match this source")
        }
        return ConversionCheckpoint(offset: saved.offset, hash: hash)
    }

    private static func saveCheckpoint(_ checkpoint: ConversionCheckpoint, partial: URL,
                                       options: InstallOptions, headerDigest: String) throws {
        let saved = Checkpoint(sourceSHA256: options.sourceFile.sha256, headerSHA256: headerDigest,
                               offset: checkpoint.offset, hash: checkpoint.hash.state)
        try POSIXFile.atomicWrite(try JSONEncoder().encode(saved), to: partial.appendingPathComponent(checkpointFile))
    }

    private static func consume(_ converter: StreamingConverter, _ bytes: [UInt8], at offset: UInt64,
                                outputs: OutputFiles) async throws -> StreamingConverter {
        try await BlockingIO.run {
            var converter = converter
            try converter.consume(bytes[...], at: offset) { file, at, data in
                try outputs.write(file: file, bytes: data, at: at)
            }
            return converter
        }
    }

    /// Reads `[start, size)` in chunks, several at once, and hands them to
    /// `body` strictly in order. At most `concurrency` chunks are in memory
    /// beyond the one being consumed.
    static func stream(_ source: any ByteSource, from start: UInt64, chunk: UInt64,
                       body: ([UInt8], UInt64) async throws -> Void) async throws {
        let ranges = Swift.stride(from: start, to: source.size, by: Int(chunk)).map {
            $0..<min($0 + chunk, source.size)
        }
        guard !ranges.isEmpty else { return }
        try await withThrowingTaskGroup(of: (Int, [UInt8]).self) { group in
            var launched = 0
            var ready: [Int: [UInt8]] = [:]
            while launched < min(source.concurrency, ranges.count) {
                let (i, range) = (launched, ranges[launched])
                group.addTask { (i, try await source.read(range)) }
                launched += 1
            }
            for i in ranges.indices {
                while ready[i] == nil {
                    guard let (j, bytes) = try await group.next() else { return }
                    ready[j] = bytes
                }
                guard let bytes = ready.removeValue(forKey: i) else { return }
                if launched < ranges.count {
                    let (k, range) = (launched, ranges[launched])
                    group.addTask { (k, try await source.read(range)) }
                    launched += 1
                }
                try await body(bytes, ranges[i].lowerBound)
            }
        }
    }

    private static func installTokenizer(_ options: InstallOptions, partial: URL) async throws {
        let directory = partial.appendingPathComponent(FieldfareFormat.tokenizerDirectory, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        for file in options.tokenizerFiles {
            let bytes: [UInt8]
            switch options.tokenizer {
            case .huggingFace:
                bytes = try await fetchPinned(file)
            case let .directory(local):
                bytes = [UInt8](try Data(contentsOf: local.appendingPathComponent(file.path)))
                let actual = digest(bytes[...])
                guard actual == file.sha256 else {
                    throw SourceError.digestMismatch(path: file.path, expected: file.sha256, actual: actual)
                }
            }
            try POSIXFile.atomicWrite(Data(bytes), to: directory.appendingPathComponent(file.path))
        }
    }

    private static func hashFiles(_ paths: [String], root: URL) async throws -> [String: FieldfareFile] {
        try await withThrowingTaskGroup(of: (String, FieldfareFile).self) { group in
            var result: [String: FieldfareFile] = [:]
            var pending = paths[...]
            func launch(_ group: inout ThrowingTaskGroup<(String, FieldfareFile), any Error>) {
                guard let path = pending.popFirst() else { return }
                let full = root.appendingPathComponent(path).path
                group.addTask {
                    let (size, digest) = try await BlockingIO.run { try POSIXFile.sha256(path: full) }
                    return (path, FieldfareFile(size: size, sha256: digest))
                }
            }
            for _ in 0..<4 { launch(&group) }
            while let (path, file) = try await group.next() {
                result[path] = file
                launch(&group)
            }
            return result
        }
    }

    private static func commit(partial: URL, destination: URL) throws {
        if FileManager.default.fileExists(atPath: destination.path) {
            guard unsafe renamex_np(partial.path, destination.path, UInt32(RENAME_SWAP)) == 0 else {
                throw InstallError.posix(call: "renamex_np(\(destination.path))", errno: errno)
            }
            try FileManager.default.removeItem(at: partial)
        } else {
            guard unsafe rename(partial.path, destination.path) == 0 else {
                throw InstallError.posix(call: "rename(\(partial.path))", errno: errno)
            }
        }
        try POSIXFile.syncDirectory(destination.deletingLastPathComponent())
    }

    static func digest(_ bytes: ArraySlice<UInt8>) -> String {
        var hash = ResumableSHA256()
        hash.update(bytes)
        return hash.finalize()
    }

    static func gigabytes(_ bytes: UInt64) -> String {
        "\((Double(bytes) / 1e9).formatted(.number.precision(.fractionLength(2)))) GB"
    }
}

/// The plan's output files, opened for positional writes.
final class OutputFiles: Sendable {
    private let fds: [Int32]
    private let paths: [String]

    init(root: URL, files: [PlannedFile], resuming: Bool) throws {
        var fds: [Int32] = []
        var paths: [String] = []
        do {
            for file in files {
                let url = root.appendingPathComponent(file.path)
                try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                        withIntermediateDirectories: true)
                let fd = unsafe open(url.path, O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0o644)
                guard fd >= 0 else { throw InstallError.posix(call: "open(\(url.path))", errno: errno) }
                fds.append(fd)
                paths.append(url.path)
                if resuming {
                    guard try POSIXFile.size(fd: fd, path: url.path) == file.size else {
                        throw InstallError.invalidPartial("\(file.path) has the wrong size")
                    }
                } else if ftruncate(fd, off_t(file.size)) != 0 {
                    throw InstallError.posix(call: "ftruncate(\(url.path))", errno: errno)
                }
            }
        } catch {
            for fd in fds { Darwin.close(fd) }
            throw error
        }
        self.fds = fds
        self.paths = paths
    }

    func write(file: Int, bytes: [UInt8], at offset: UInt64) throws {
        try POSIXFile.write(fd: fds[file], bytes: bytes, at: offset, path: paths[file])
    }

    func sync() async throws {
        let (fds, paths) = (fds, paths)
        try await BlockingIO.run {
            for (fd, path) in zip(fds, paths) { try POSIXFile.sync(fd: fd, path: path) }
        }
    }

    func close() {
        for fd in fds { Darwin.close(fd) }
    }
}
