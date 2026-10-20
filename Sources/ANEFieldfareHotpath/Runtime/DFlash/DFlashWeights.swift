import Foundation
import Darwin
import Metal

public enum DFlashError: Error, CustomStringConvertible {
    case invalidCheckpoint(String)
    case invalidState(String)

    public var description: String {
        switch self {
        case .invalidCheckpoint(let detail): return "DFlash drafter checkpoint: \(detail)"
        case .invalidState(let detail): return "DFlash drafter: \(detail)"
        }
    }
}

/// Configuration read from a z-lab DFlash drafter's `config.json`.
public struct DFlashDrafterConfig: Sendable, Equatable, Codable {
    public let hiddenSize: Int
    public let layers: Int
    public let heads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let intermediateSize: Int
    public let blockSize: Int
    public let contextLayers: [Int]
    public let maskTokenID: Int32
    public let slidingWindow: Int
    public let slidingLayers: [Bool]
    public let ropeTheta: Float
    public let rmsEps: Float

    public var qWidth: Int { heads * headDim }
    public var kvWidth: Int { kvHeads * headDim }
    /// One row of the fused Q/K/V projection's output.
    public var qkvWidth: Int { qWidth + 2 * kvWidth }

    static func load(_ url: URL) throws -> DFlashDrafterConfig {
        guard let json = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any],
              let dflash = json["dflash_config"] as? [String: Any],
              let hidden = json["hidden_size"] as? Int, let layers = json["num_hidden_layers"] as? Int,
              let heads = json["num_attention_heads"] as? Int, let kvHeads = json["num_key_value_heads"] as? Int,
              let headDim = json["head_dim"] as? Int, let intermediate = json["intermediate_size"] as? Int,
              let contextLayers = dflash["target_layer_ids"] as? [Int], let mask = dflash["mask_token_id"] as? Int,
              let types = json["layer_types"] as? [String], types.count == layers else {
            throw DFlashError.invalidCheckpoint("config.json is missing DFlash fields")
        }
        guard headDim == 128, heads % kvHeads == 0, json["model_type"] as? String == "qwen3" else {
            throw DFlashError.invalidCheckpoint("unsupported drafter shape")
        }
        return DFlashDrafterConfig(
            hiddenSize: hidden, layers: layers, heads: heads, kvHeads: kvHeads, headDim: headDim,
            intermediateSize: intermediate, blockSize: json["block_size"] as? Int ?? 16,
            contextLayers: contextLayers, maskTokenID: Int32(mask),
            slidingWindow: json["sliding_window"] as? Int ?? Int.max,
            slidingLayers: types.map { $0 == "sliding_attention" },
            ropeTheta: Float((json["rope_theta"] as? Double) ?? 1_000_000),
            rmsEps: Float((json["rms_norm_eps"] as? Double) ?? 1e-6))
    }
}

/// An INT4 affine matrix inside `DFlashWeights.buffer`: `rows` rows of
/// `columns`, group 64, BF16 scales and biases.
struct DFlashMatrix {
    let weights: Int
    let scales: Int
    let biases: Int
    let rows: Int
    let columns: Int

    /// Rows `range` of this matrix, which are contiguous in all three arrays.
    func rows(_ range: Range<Int>) -> DFlashMatrix {
        let groups = columns / Quantization.groupSize
        return DFlashMatrix(weights: weights + range.lowerBound * columns / 2,
                            scales: scales + range.lowerBound * groups * 2,
                            biases: biases + range.lowerBound * groups * 2,
                            rows: range.count, columns: columns)
    }
}

/// The drafter's weights as installed for this runtime: every projection in
/// the target's INT4 format (group 64, BF16 scale and bias), Q/K/V and
/// gate/up concatenated by rows so each is one dispatch, the norms in BF16.
///
/// They live in one file next to the checkpoint, built once from
/// `model.safetensors`, and the GPU reads them in place from the page cache:
/// a file-backed no-copy buffer streams as fast as a private allocation once
/// resident, the drafter costs no load-time quantization or BF16 staging
/// copy, and its pages stay clean, so the OS can reclaim them like any
/// cached file.
///
/// Layout: 8-byte magic `GFDFI4\0\1`, UInt32 header length, a JSON header
/// (source fingerprint, config, tensor table) padded to `dataAlignment`, then
/// the tensors, each 256-byte aligned within the data region.
struct DFlashWeights {
    static let fileName = "fieldfare-dflash-int4.bin"
    static let magic: [UInt8] = Array("GFDFI4".utf8) + [0, 1]
    static let dataAlignment = 16384

    struct Layer {
        let inputNorm: Int
        let postAttentionNorm: Int
        let qNorm: Int
        let kNorm: Int
        let qkv: DFlashMatrix
        let o: DFlashMatrix
        let gateUp: DFlashMatrix
        let down: DFlashMatrix
    }

    let buffer: any MTLBuffer
    let fc: DFlashMatrix
    let hiddenNorm: Int
    let norm: Int
    let layers: [Layer]

    /// The installed file's tensor table, as `install` leaves it.
    struct Layout: Sendable {
        var tensors: [String: Int]
        var dataBytes: Int
    }

    private struct Header: Codable {
        var sourceBytes: Int
        var sourceModified: Double
        var config: DFlashDrafterConfig
        var tensors: [String: Int]
        var dataBytes: Int
    }

    /// Makes sure `directory` holds installed weights for its checkpoint,
    /// building them if they are missing or older than `model.safetensors`,
    /// and returns their layout. Blocking file I/O: callers on the Swift
    /// concurrency pool run it through `BlockingIO`.
    static func install(directory: URL, config: DFlashDrafterConfig) throws -> Layout {
        let source = directory.appendingPathComponent("model.safetensors")
        let installed = directory.appendingPathComponent(fileName)
        let attributes = try FileManager.default.attributesOfItem(atPath: source.path)
        let sourceBytes = (attributes[.size] as? NSNumber)?.intValue ?? -1
        let sourceModified = (attributes[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        if let header = try? readHeader(installed), header.sourceBytes == sourceBytes,
           header.sourceModified == sourceModified, header.config == config {
            return Layout(tensors: header.tensors, dataBytes: header.dataBytes)
        }
        FileHandle.standardError.write(Data("[dflash] installing INT4 drafter weights to \(installed.path)\n".utf8))
        removeStalePartials(in: directory)
        return try build(source: source, into: installed, config: config,
                         fingerprint: (sourceBytes, sourceModified))
    }

    /// Maps installed weights with `layout` (from `install`) as a no-copy
    /// buffer over the page cache.
    static func open(directory: URL, layout: Layout, config: DFlashDrafterConfig,
                     device: any MTLDevice) throws -> DFlashWeights {
        let mapped = try ResidentBuffer(fileURL: directory.appendingPathComponent(fileName),
                                        fileOffset: UInt64(dataAlignment),
                                        residentSize: UInt64(layout.dataBytes), device: device)
        mapped.buffer.label = "dflash.weights"
        return try DFlashWeights(buffer: mapped.buffer, tensors: layout.tensors, config: config)
    }

    /// Deletes partial installs left by processes that no longer run; a
    /// live install's partial file belongs to it.
    private static func removeStalePartials(in directory: URL) {
        let prefix = ".\(fileName)."
        let names = (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []
        for name in names where name.hasPrefix(prefix) && name.hasSuffix(".partial") {
            let pid = Int32(name.dropFirst(prefix.count).dropLast(".partial".count)) ?? 0
            guard pid <= 0 || kill(pid, 0) != 0 else { continue }
            try? FileManager.default.removeItem(at: directory.appendingPathComponent(name))
        }
    }

    private static func readHeader(_ url: URL) throws -> Header {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        guard let prefix = try handle.read(upToCount: 12), prefix.count == 12,
              Array(prefix.prefix(8)) == magic else {
            throw DFlashError.invalidCheckpoint("\(url.lastPathComponent): bad magic")
        }
        let length = prefix.suffix(4).enumerated().reduce(0) { $0 | Int($1.element) << (8 * $1.offset) }
        guard let json = try handle.read(upToCount: length), json.count == length else {
            throw DFlashError.invalidCheckpoint("\(url.lastPathComponent): short header")
        }
        return try JSONDecoder().decode(Header.self, from: json)
    }

    private init(buffer: any MTLBuffer, tensors: [String: Int], config: DFlashDrafterConfig) throws {
        // Every tensor must lie inside the mapping: a corrupt table would
        // otherwise send the GPU past the end of the buffer.
        func offset(_ name: String, bytes: Int) throws -> Int {
            guard let value = tensors[name], value >= 0, value + bytes <= buffer.length else {
                throw DFlashError.invalidCheckpoint("installed weights lack \(name) or it overruns the file")
            }
            return value
        }
        func vector(_ name: String, length: Int) throws -> Int { try offset(name, bytes: length * 2) }
        func matrix(_ name: String, rows: Int, columns: Int) throws -> DFlashMatrix {
            let groups = rows * columns / Quantization.groupSize
            return DFlashMatrix(weights: try offset(name + ".q4", bytes: rows * columns / 2),
                                scales: try offset(name + ".scales", bytes: groups * 2),
                                biases: try offset(name + ".biases", bytes: groups * 2), rows: rows, columns: columns)
        }
        let D = config.hiddenSize, F = config.intermediateSize
        self.buffer = buffer
        fc = try matrix("fc", rows: D, columns: config.contextLayers.count * D)
        hiddenNorm = try vector("hidden_norm", length: D)
        norm = try vector("norm", length: D)
        layers = try (0..<config.layers).map { l in
            Layer(inputNorm: try vector("L\(l).input_norm", length: D),
                  postAttentionNorm: try vector("L\(l).post_attention_norm", length: D),
                  qNorm: try vector("L\(l).q_norm", length: config.headDim),
                  kNorm: try vector("L\(l).k_norm", length: config.headDim),
                  qkv: try matrix("L\(l).qkv", rows: config.qkvWidth, columns: D),
                  o: try matrix("L\(l).o", rows: D, columns: config.qWidth),
                  gateUp: try matrix("L\(l).gate_up", rows: 2 * F, columns: D),
                  down: try matrix("L\(l).down", rows: D, columns: F))
        }
    }

    // MARK: - Install

    /// A safetensors file mapped read-only, its BF16 tensors by name.
    /// `@safe`: the mapping lives until `close`, which `build` defers right
    /// after opening; `bf16` bounds every view by the tensor's header entry,
    /// checked against the file length at open.
    @safe private struct Safetensors {
        let base: UnsafeRawPointer
        let length: Int
        let tensors: [String: (offset: Int, shape: [Int])]

        init(_ url: URL) throws {
            let fd = unsafe Darwin.open(url.path, O_RDONLY | O_CLOEXEC)
            guard fd >= 0 else { throw ModelError.posixFailed(call: "open(\(url.path))", errno: errno) }
            defer { Darwin.close(fd) }
            var info = stat()
            guard unsafe fstat(fd, &info) == 0, info.st_size > 8 else {
                throw DFlashError.invalidCheckpoint("unreadable \(url.lastPathComponent)")
            }
            length = Int(info.st_size)
            guard let mapped = unsafe mmap(nil, length, PROT_READ, MAP_PRIVATE, fd, 0),
                  unsafe mapped != MAP_FAILED else {
                throw ModelError.posixFailed(call: "mmap(\(url.path))", errno: errno)
            }
            unsafe base = UnsafeRawPointer(mapped)
            _ = unsafe posix_madvise(mapped, length, POSIX_MADV_SEQUENTIAL)
            let headerLength = unsafe Int(base.loadUnaligned(as: UInt64.self))
            guard headerLength > 0, 8 + headerLength <= length,
                  let header = unsafe try JSONSerialization.jsonObject(
                    with: Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: base + 8), count: headerLength,
                               deallocator: .none)) as? [String: Any] else {
                unsafe munmap(mapped, length)
                throw DFlashError.invalidCheckpoint("bad safetensors header")
            }
            var tensors: [String: (offset: Int, shape: [Int])] = [:]
            for (name, value) in header where name != "__metadata__" {
                guard let entry = value as? [String: Any], entry["dtype"] as? String == "BF16",
                      let range = entry["data_offsets"] as? [Int], range.count == 2,
                      let shape = entry["shape"] as? [Int],
                      8 + headerLength + range[1] <= length,
                      range[1] - range[0] == shape.reduce(2, *) else {
                    unsafe munmap(mapped, length)
                    throw DFlashError.invalidCheckpoint("tensor \(name) is not BF16")
                }
                tensors[name] = (8 + headerLength + range[0], shape)
            }
            self.tensors = tensors
        }

        func close() { unsafe munmap(UnsafeMutableRawPointer(mutating: base), length) }

        func bf16(_ name: String, shape: [Int]) throws -> UnsafePointer<UInt16> {
            guard let tensor = tensors[name], tensor.shape == shape else {
                throw DFlashError.invalidCheckpoint("tensor \(name) is not \(shape) BF16")
            }
            return unsafe (base + tensor.offset).assumingMemoryBound(to: UInt16.self)
        }
    }

    /// Quantizes the checkpoint into `destination`, writing a temporary file
    /// mapped read-write and renaming it into place once complete, so a
    /// crash never leaves a partial install behind.
    private static func build(source: URL, into destination: URL, config: DFlashDrafterConfig,
                              fingerprint: (bytes: Int, modified: Double)) throws -> Layout {
        let st = try Safetensors(source)
        defer { st.close() }
        let D = config.hiddenSize, F = config.intermediateSize
        let q = config.qWidth, kv = config.kvWidth
        // (name, source tensors concatenated by rows, columns)
        var matrices: [(String, [String], Int)] = [("fc", ["fc.weight"], config.contextLayers.count * D)]
        var vectors: [(String, String)] = [("hidden_norm", "hidden_norm.weight"), ("norm", "norm.weight")]
        for l in 0..<config.layers {
            let a = "layers.\(l).self_attn.", m = "layers.\(l).mlp."
            matrices += [("L\(l).qkv", [a + "q_proj.weight", a + "k_proj.weight", a + "v_proj.weight"], D),
                         ("L\(l).o", [a + "o_proj.weight"], q),
                         ("L\(l).gate_up", [m + "gate_proj.weight", m + "up_proj.weight"], D),
                         ("L\(l).down", [m + "down_proj.weight"], F)]
            vectors += [("L\(l).input_norm", "layers.\(l).input_layernorm.weight"),
                        ("L\(l).post_attention_norm", "layers.\(l).post_attention_layernorm.weight"),
                        ("L\(l).q_norm", a + "q_norm.weight"), ("L\(l).k_norm", a + "k_norm.weight")]
        }
        let rowsOf: [String: Int] = [
            "fc.weight": D, "q_proj.weight": q, "k_proj.weight": kv, "v_proj.weight": kv, "o_proj.weight": D,
            "gate_proj.weight": F, "up_proj.weight": F, "down_proj.weight": D,
        ]
        func rows(_ name: String) -> Int { rowsOf[String(name.split(separator: ".").suffix(2).joined(separator: "."))] ?? 0 }
        func vectorLength(_ name: String) -> Int { name.hasSuffix("_norm.weight") && name.contains("self_attn") ? config.headDim : D }

        var tensors: [String: Int] = [:]
        var cursor = 0
        func take(_ name: String, _ bytes: Int) -> Int {
            tensors[name] = cursor
            defer { cursor += (bytes + 255) / 256 * 256 }
            return cursor
        }
        var placedMatrices: [(sources: [String], columns: Int, weights: Int, scales: Int, biases: Int)] = []
        for (name, sources, columns) in matrices {
            guard columns % Quantization.groupSize == 0 else {
                throw DFlashError.invalidCheckpoint("\(name) has \(columns) columns")
            }
            let total = sources.map(rows).reduce(0, +)
            placedMatrices.append((sources, columns,
                                   take(name + ".q4", total * columns / 2),
                                   take(name + ".scales", total * columns / Quantization.groupSize * 2),
                                   take(name + ".biases", total * columns / Quantization.groupSize * 2)))
        }
        let placedVectors = vectors.map { name, source in (source: source, offset: take(name, vectorLength(source) * 2)) }
        let dataBytes = cursor
        let header = try JSONEncoder().encode(Header(sourceBytes: fingerprint.bytes,
                                                     sourceModified: fingerprint.modified,
                                                     config: config, tensors: tensors, dataBytes: dataBytes))
        guard 12 + header.count <= dataAlignment else {
            throw DFlashError.invalidCheckpoint("installed weights header exceeds \(dataAlignment) bytes")
        }
        let pageSize = Int(getpagesize())
        let fileBytes = (dataAlignment + dataBytes + pageSize - 1) / pageSize * pageSize

        let temporary = destination.deletingLastPathComponent()
            .appendingPathComponent(".\(fileName).\(getpid()).partial")
        let fd = unsafe Darwin.open(temporary.path, O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0o644)
        guard fd >= 0 else { throw ModelError.posixFailed(call: "open(\(temporary.path))", errno: errno) }
        var renamed = false
        defer {
            close(fd)
            if !renamed { unsafe unlink(temporary.path) }
        }
        // Allocate every block before writing through the mapping: a write
        // to a sparse page the disk cannot back raises SIGBUS instead of an
        // error.
        var store = fstore_t(fst_flags: UInt32(F_ALLOCATEALL), fst_posmode: F_PEOFPOSMODE, fst_offset: 0,
                             fst_length: off_t(fileBytes), fst_bytesalloc: 0)
        guard unsafe fcntl(fd, F_PREALLOCATE, &store) != -1 else {
            throw ModelError.posixFailed(call: "F_PREALLOCATE(\(temporary.path), \(fileBytes) bytes)", errno: errno)
        }
        guard ftruncate(fd, off_t(fileBytes)) == 0 else {
            throw ModelError.posixFailed(call: "ftruncate(\(temporary.path))", errno: errno)
        }
        guard let mapped = unsafe mmap(nil, fileBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0),
              unsafe mapped != MAP_FAILED else {
            throw ModelError.posixFailed(call: "mmap(\(temporary.path))", errno: errno)
        }
        defer { unsafe munmap(mapped, fileBytes) }
        unsafe mapped.copyMemory(from: magic, byteCount: magic.count)
        unsafe mapped.storeBytes(of: UInt32(header.count).littleEndian, toByteOffset: 8, as: UInt32.self)
        unsafe header.copyBytes(to: UnsafeMutableRawBufferPointer(start: mapped + 12, count: header.count))
        let data = unsafe mapped + dataAlignment

        for (sources, columns, w, s, b) in placedMatrices {
            var row0 = 0
            for sourceName in sources {
                let n = rows(sourceName)
                let source = unsafe try st.bf16(sourceName, shape: [n, columns])
                let groups = columns / Quantization.groupSize
                unsafe quantize(source, rows: n, columns: columns,
                                packed: (data + w).assumingMemoryBound(to: UInt8.self) + row0 * columns / 2,
                                scales: (data + s).assumingMemoryBound(to: UInt16.self) + row0 * groups,
                                biases: (data + b).assumingMemoryBound(to: UInt16.self) + row0 * groups)
                row0 += n
            }
        }
        for (source, offset) in placedVectors {
            let n = vectorLength(source)
            let values = unsafe try st.bf16(source, shape: [n])
            unsafe (data + offset).copyMemory(from: values, byteCount: n * 2)
        }
        guard unsafe msync(mapped, fileBytes, MS_SYNC) == 0, fsync(fd) == 0 else {
            throw ModelError.posixFailed(call: "msync(\(temporary.path))", errno: errno)
        }
        guard unsafe rename(temporary.path, destination.path) == 0 else {
            throw ModelError.posixFailed(call: "rename(\(destination.path))", errno: errno)
        }
        renamed = true
        return Layout(tensors: tensors, dataBytes: dataBytes)
    }

    /// `[rows][columns]` BF16 to INT4 affine with the target installer's
    /// rounding: per 64-wide group, scale = (max - min) / 15 and bias = min,
    /// both rounded to BF16 first.
    private static func quantize(_ source: UnsafePointer<UInt16>, rows: Int, columns: Int,
                                 packed: UnsafeMutablePointer<UInt8>, scales: UnsafeMutablePointer<UInt16>,
                                 biases: UnsafeMutablePointer<UInt16>) {
        let groupsPerRow = columns / Quantization.groupSize
        // SAFETY: iteration `row` reads only row `row` of the source and
        // writes only that row's packed weights, scales and biases, so no two
        // iterations touch the same bytes, and every write lands before
        // `concurrentPerform` returns.
        nonisolated(unsafe) let (wp, pp, sp, bp) = unsafe (source, packed, scales, biases)
        // SAFETY: one-time CPU quantization at install, fanned over GCD
        // workers. `install` runs on `BlockingIO`'s queue, so neither the
        // caller nor the workers are Swift concurrency pool threads.
        DispatchQueue.concurrentPerform(iterations: rows) { row in
            for g in 0..<groupsPerRow {
                let base = row * columns + g * Quantization.groupSize
                var lo = Float.infinity, hi = -Float.infinity
                for i in 0..<Quantization.groupSize {
                    let v = unsafe Quantization.bf16ToFloat(wp[base + i])
                    lo = min(lo, v); hi = max(hi, v)
                }
                let sBits = Quantization.bf16Bits(hi == lo ? 1 : (hi - lo) / 15)
                let bBits = Quantization.bf16Bits(lo)
                unsafe sp[row * groupsPerRow + g] = sBits
                unsafe bp[row * groupsPerRow + g] = bBits
                let scale = Quantization.bf16ToFloat(sBits), bias = Quantization.bf16ToFloat(bBits)
                let inv = scale == 0 ? 0 : 1 / scale
                for i in stride(from: 0, to: Quantization.groupSize, by: 2) {
                    func q(_ v: UInt16) -> UInt8 {
                        UInt8(max(0, min(15, Int(((Quantization.bf16ToFloat(v) - bias) * inv).rounded()))))
                    }
                    unsafe pp[(base + i) / 2] = unsafe q(wp[base + i]) | (q(wp[base + i + 1]) << 4)
                }
            }
        }
    }
}
