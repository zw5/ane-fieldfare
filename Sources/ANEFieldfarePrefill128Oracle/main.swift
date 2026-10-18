import Darwin
import Foundation
import Metal
import ANEFieldfareHotpath

private struct ArmReceipt: Codable {
    let schema: String
    let status: String
    let exactCommand: [String]
    let host: String
    let os: String
    let promptIndex: Int
    let tokenCount: Int
    let chunkTokens: Int
    let expertCacheSlots: Int
    let maxContext: Int
    let prefillMilliseconds: Double
    let greedyToken: UInt32
    let logitsBytes: Int
    let continuationLogitsBytes: Int
    let processPeakResidentBytes: Int64
    let claimBoundary: String

    enum CodingKeys: String, CodingKey {
        case schema, status, host, os
        case exactCommand = "exact_command"
        case promptIndex = "prompt_index"
        case tokenCount = "token_count"
        case chunkTokens = "chunk_tokens"
        case expertCacheSlots = "expert_cache_slots"
        case maxContext = "max_context"
        case prefillMilliseconds = "prefill_milliseconds"
        case greedyToken = "greedy_token"
        case logitsBytes = "logits_bytes"
        case continuationLogitsBytes = "continuation_logits_bytes"
        case processPeakResidentBytes = "process_peak_resident_bytes"
        case claimBoundary = "claim_boundary"
    }
}

private let prompts = [
    Array(repeating: """
    The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome.
    The capital of Spain is Madrid. The capital of Portugal is Lisbon. The capital of Japan is Tokyo.
    The capital of Canada is Ottawa. The capital of Brazil is Brasilia. The capital of Colombia is Bogota.
    The capital of Australia is Canberra. The capital of India is New Delhi. The capital of Egypt is Cairo.
    """, count: 2).joined(separator: " ")
    + " Now answer briefly and with no explanation: the capital of France is",
]

private func fp16Values(_ data: Data) -> [Float] {
    data.withUnsafeBytes { raw in
        raw.bindMemory(to: UInt16.self).map { Float(Float16(bitPattern: $0)) }
    }
}

private func compareFP16(_ left: Data, _ right: Data) -> [String: Any] {
    guard left.count == right.count else {
        return ["exact": false, "left_bytes": left.count, "right_bytes": right.count]
    }
    let a = fp16Values(left)
    let b = fp16Values(right)
    var mismatchCount = 0
    var maxAbsolute: Float = 0
    for index in a.indices {
        let byte = index * MemoryLayout<UInt16>.stride
        if left[byte] != right[byte] || left[byte + 1] != right[byte + 1] {
            mismatchCount += 1
        }
        maxAbsolute = max(maxAbsolute, abs(a[index] - b[index]))
    }
    return [
        "exact": mismatchCount == 0,
        "element_count": a.count,
        "mismatch_count": mismatchCount,
        "max_absolute_error": Double(maxAbsolute),
    ]
}

private func argmax(_ logits: Data) -> UInt32 {
    let values = fp16Values(logits)
    var best = 0
    for index in 1..<values.count where values[index] > values[best] { best = index }
    return UInt32(best)
}

private func milliseconds(_ duration: Duration) -> Double {
    Double(duration.components.seconds) * 1_000
        + Double(duration.components.attoseconds) / 1e15
}

private func peakResidentBytes() -> Int64 {
    var usage = rusage()
    guard getrusage(RUSAGE_SELF, &usage) == 0 else { return -1 }
    return Int64(usage.ru_maxrss)
}

private func writeJSON<T: Encodable>(_ value: T, to url: URL) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    var data = try encoder.encode(value)
    data.append(0x0A)
    try data.write(to: url, options: .atomic)
}

/// Runs exactly one authentic arm. The process must exit before another arm starts;
/// process isolation is the teardown contract for the model, streamer, KV, and Metal state.
private func runArm(modelURL: URL, chunk: Int, promptIndex: Int,
                    outputPrefix: URL) async throws {
    guard chunk == 64 || chunk == 128 else {
        throw PrefillError.chunkedUnsupported("oracle chunk must be 64 or 128")
    }
    guard prompts.indices.contains(promptIndex) else {
        throw PrefillError.chunkedUnsupported("oracle prompt index is out of range")
    }
    try FileManager.default.createDirectory(
        at: outputPrefix.deletingLastPathComponent(),
        withIntermediateDirectories: true)

    let tokenizer = try await GFTokenizer.load(forModelDirectory: modelURL)
    let tokenIDs = tokenizer.encode(prompts[promptIndex], addBOS: true)
    guard tokenIDs.count > 128, tokenIDs.count <= 384 else {
        throw PrefillError.chunkedUnsupported(
            "representative prompt must contain 129...384 tokens, got \(tokenIDs.count)")
    }

    let context = try MetalContext()
    let model = try Model.load(directoryURL: modelURL,
                               device: context.device,
                               streamingMode: .pread(slotCount: 8),
                               expertCachePolicy: .lfu,
                               integrityPolicy: .sizeCheckTrustedReceipt)
    let runtime = RuntimeConfiguration(expertCacheSlots: 8,
                                       prefillChunkTokens: chunk,
                                       forceLogitsHead: true)
    let runner = try RealForwardRunner(model: model, context: context,
                                       maxContext: 512,
                                       runtimeConfiguration: runtime)
    guard let logits = context.device.makeBuffer(
        length: model.config.vocabSize * MemoryLayout<Float16>.stride,
        options: .storageModeShared),
          let continuation = context.device.makeBuffer(
        length: model.config.vocabSize * MemoryLayout<Float16>.stride,
        options: .storageModeShared) else {
        throw PrefillError.chunkedUnsupported("oracle logits allocation failed")
    }

    memset(logits.contents(), 0xA5, logits.length)
    let started = ContinuousClock.now
    _ = try await runner.prefillChunked(tokens: tokenIDs[...], startPosition: 0,
                                        outputMode: .logits,
                                        config: .production(chunkTokens: chunk),
                                        into: logits, onProgress: { _ in })
    let prefillMilliseconds = milliseconds(ContinuousClock.now - started)
    let logitsData = Data(bytes: logits.contents(), count: logits.length)
    let greedyToken = argmax(logitsData)

    memset(continuation.contents(), 0x5A, continuation.length)
    try await runner.produce(token: Int32(bitPattern: greedyToken),
                             position: tokenIDs.count, into: continuation)
    let continuationData = Data(bytes: continuation.contents(), count: continuation.length)

    try logitsData.write(to: URL(fileURLWithPath: outputPrefix.path + ".logits.bin"),
                         options: .atomic)
    try continuationData.write(
        to: URL(fileURLWithPath: outputPrefix.path + ".continuation.bin"),
        options: .atomic)
    let receipt = ArmReceipt(
        schema: "ane-fieldfare.prefill-128-causal-kv-arm.v1",
        status: "accepted",
        exactCommand: CommandLine.arguments,
        host: ProcessInfo.processInfo.hostName,
        os: ProcessInfo.processInfo.operatingSystemVersionString,
        promptIndex: promptIndex,
        tokenCount: tokenIDs.count,
        chunkTokens: chunk,
        expertCacheSlots: 8,
        maxContext: 512,
        prefillMilliseconds: prefillMilliseconds,
        greedyToken: greedyToken,
        logitsBytes: logitsData.count,
        continuationLogitsBytes: continuationData.count,
        processPeakResidentBytes: peakResidentBytes(),
        claimBoundary: "One process ran one authentic prompt-prefill arm, wrote complete FP16 final logits and one continuation KV-read result, and then exited. Accepted is not a cross-arm numerical validation.")
    try writeJSON(receipt, to: URL(fileURLWithPath: outputPrefix.path + ".json"))
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    FileHandle.standardOutput.write(try encoder.encode(receipt))
    FileHandle.standardOutput.write(Data("\n".utf8))
}

private func compareArms(referencePrefix: URL, candidatePrefix: URL,
                         outputURL: URL) throws {
    let decoder = JSONDecoder()
    let reference = try decoder.decode(
        ArmReceipt.self,
        from: Data(contentsOf: URL(fileURLWithPath: referencePrefix.path + ".json")))
    let candidate = try decoder.decode(
        ArmReceipt.self,
        from: Data(contentsOf: URL(fileURLWithPath: candidatePrefix.path + ".json")))
    guard reference.chunkTokens == 64, candidate.chunkTokens == 128,
          reference.promptIndex == candidate.promptIndex,
          reference.tokenCount == candidate.tokenCount else {
        throw PrefillError.chunkedUnsupported("arm receipts do not describe a comparable 64/128 pair")
    }
    let referenceLogits = try Data(contentsOf: URL(
        fileURLWithPath: referencePrefix.path + ".logits.bin"))
    let candidateLogits = try Data(contentsOf: URL(
        fileURLWithPath: candidatePrefix.path + ".logits.bin"))
    let referenceContinuation = try Data(contentsOf: URL(
        fileURLWithPath: referencePrefix.path + ".continuation.bin"))
    let candidateContinuation = try Data(contentsOf: URL(
        fileURLWithPath: candidatePrefix.path + ".continuation.bin"))
    let logits = compareFP16(referenceLogits, candidateLogits)
    let continuation = compareFP16(referenceContinuation, candidateContinuation)
    let logitsExact = logits["exact"] as? Bool == true
    let continuationExact = continuation["exact"] as? Bool == true
    let greedyExact = reference.greedyToken == candidate.greedyToken
    let withinEngineCap = reference.processPeakResidentBytes <= 3_000_000_000
        && candidate.processPeakResidentBytes <= 3_000_000_000
    let speedup = reference.prefillMilliseconds / candidate.prefillMilliseconds
    let promoted = logitsExact && continuationExact && greedyExact
        && withinEngineCap && speedup > 1
    let receipt: [String: Any] = [
        "schema": "ane-fieldfare.prefill-128-causal-kv-oracle.v2",
        "status": promoted ? "validated" : "rejected",
        "exact_command": CommandLine.arguments,
        "exit_status": 0,
        "host": ProcessInfo.processInfo.hostName,
        "os": ProcessInfo.processInfo.operatingSystemVersionString,
        "reference_arm": try JSONSerialization.jsonObject(with: JSONEncoder().encode(reference)),
        "candidate_arm": try JSONSerialization.jsonObject(with: JSONEncoder().encode(candidate)),
        "oracle": [
            "final_logits": logits,
            "continuation_logits_after_kv_read": continuation,
            "greedy_token_exact": greedyExact,
        ],
        "phase_specific_prefill": [
            "chunk64_milliseconds": reference.prefillMilliseconds,
            "chunk128_milliseconds": candidate.prefillMilliseconds,
            "chunk128_speedup": speedup,
            "chunk128_wins": speedup > 1,
        ],
        "resource_boundary": [
            "engine_peak_cap_bytes": 3_000_000_000,
            "chunk64_process_peak_resident_bytes": reference.processPeakResidentBytes,
            "chunk128_process_peak_resident_bytes": candidate.processPeakResidentBytes,
            "both_arms_within_cap": withinEngineCap,
            "isolation": "one arm per process; the first process exited before the second began",
        ],
        "promotion": promoted,
        "claim_boundary": "One representative >128-token authentic prompt compares chunk-64 and chunk-128 final FP16 logits, greedy token, one continuation KV-read step, prefill-only elapsed time, and per-process peak resident memory. It is an admission screen, not a multi-prompt throughput claim. A mismatch requires a separate one-arm boundary diagnostic; this gate retains no tensor-boundary snapshots.",
    ]
    let data = try JSONSerialization.data(withJSONObject: receipt,
                                           options: [.prettyPrinted, .sortedKeys])
    try (data + Data("\n".utf8)).write(to: outputURL, options: .atomic)
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
}

@main
private enum Prefill128OracleMain {
    static func main() async {
        do {
            switch CommandLine.arguments.dropFirst().first {
            case "arm":
                guard CommandLine.arguments.count == 6,
                      let chunk = Int(CommandLine.arguments[3]),
                      let promptIndex = Int(CommandLine.arguments[4]) else {
                    throw PrefillError.chunkedUnsupported(
                        "usage: ANEFieldfarePrefill128Oracle arm <model-directory> <64|128> <prompt-index> <output-prefix>")
                }
                try await runArm(
                    modelURL: URL(fileURLWithPath: CommandLine.arguments[2]),
                    chunk: chunk,
                    promptIndex: promptIndex,
                    outputPrefix: URL(fileURLWithPath: CommandLine.arguments[5]))
            case "compare":
                guard CommandLine.arguments.count == 5 else {
                    throw PrefillError.chunkedUnsupported(
                        "usage: ANEFieldfarePrefill128Oracle compare <chunk64-prefix> <chunk128-prefix> <receipt-path>")
                }
                try compareArms(
                    referencePrefix: URL(fileURLWithPath: CommandLine.arguments[2]),
                    candidatePrefix: URL(fileURLWithPath: CommandLine.arguments[3]),
                    outputURL: URL(fileURLWithPath: CommandLine.arguments[4]))
            default:
                throw PrefillError.chunkedUnsupported(
                    "expected arm or compare mode; each arm must run in a separate process")
            }
        } catch {
            FileHandle.standardError.write(Data("error: \(error)\n".utf8))
            Foundation.exit(1)
        }
    }
}
