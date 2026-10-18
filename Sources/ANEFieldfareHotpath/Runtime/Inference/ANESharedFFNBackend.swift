import ANEFieldfareAOTExpertRuntime
import Foundation
import IOSurface
import Metal

final class SharedSurfaceOwner: @unchecked Sendable {
    let surface: IOSurfaceRef
    init(_ surface: IOSurfaceRef) { self.surface = surface }
}

enum ANEExecutionError: Error, CustomStringConvertible {
    case rejected(String)
    var description: String {
        switch self { case .rejected(let message): return "ANE execution rejected: \(message)" }
    }
}

/// Immutable checkpoint FFNs on ANE; Metal handles layout and the routed branch.
/// Prefill uses distinct position lanes; decode uses a single useful lane.
/// Retained for reproducible experiments; resident FP16 constants lost overall.
final class ANESharedFFNBackend {
    let layers: [ANESharedFFNLayer]
    let mode: String

    /// Construct guest layers `0..<count`. Placement (layer count and which
    /// phases use the guest) is resolved by `ExecutionPlacementPlan`.
    init(model: Model, context: MetalContext, count: Int, prefill: Bool, decode: Bool, prefillTokens: Int) throws {
        guard model.config.hiddenSize == 2816, model.config.intermediateSize == 2112 else {
            throw ANEExecutionError.rejected("unsupported checkpoint dimensions")
        }
        guard count > 0, count <= model.config.numLayers, prefill || decode else {
            throw ANEExecutionError.rejected("shared FFN guest needs 1...\(model.config.numLayers) layers and a phase")
        }
        mode = prefill && decode ? "both" : (prefill ? "prefill" : "decode")
        let positions = mode == "decode" ? 64 : max(64, prefillTokens)
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void ane_shared_pack(const device half *x [[buffer(0)]], device half *y [[buffer(1)]],
                                    constant uint &positions [[buffer(2)]], constant uint &rows [[buffer(3)]],
                                    uint i [[thread_position_in_grid]]) {
            y[i] = (i % positions < rows) ? x[(i % positions) * 2816 + i / positions] : half(0);
        }
        kernel void ane_shared_gather(const device half *x [[buffer(0)]], device half *y [[buffer(1)]],
                                      constant uint &positions [[buffer(2)]],
                                      uint i [[thread_position_in_grid]]) { y[i] = x[(i % 2816) * positions + i / 2816]; }
        """
        let library = try context.device.makeLibrary(source: source, options: nil)
        let pack = try context.device.makeComputePipelineState(function: library.makeFunction(name: "ane_shared_pack")!)
        let gather = try context.device.makeComputePipelineState(function: library.makeFunction(name: "ane_shared_gather")!)
        let path = model.directoryURL.appendingPathComponent("model_weights.bin").path
        var built: [ANESharedFFNLayer] = []
        for layer in 0..<count {
            built.append(try ANESharedFFNLayer(path: path, layer: layer, bits: model.sharedExpertWeightBits,
                                             device: context.device, positions: positions, pack: pack, gather: gather))
        }
        layers = built
    }

    func layer(_ index: Int, prefill: Bool) -> SharedFFNGuestLayer? {
        let enabled = prefill ? mode != "decode" : mode != "prefill"
        return enabled && index < layers.count ? layers[index] : nil
    }

    var profile: [String: Any] {
        let info = layers.map { ANEFieldfareANESharedFFNGetInfo($0.handle) }
        return ["backend": "ANE immutable shared FFN + Metal attention/routed experts/head",
                "mode": mode, "useful_rows": layers.reduce(0) { $0 + $1.evaluatedRows },
                "layers": layers.count, "compile_count": info.reduce(0) { $0 + Int($1.compileCount) },
                "load_count": info.reduce(0) { $0 + Int($1.loadCount) },
                "evaluations": info.reduce(UInt64(0)) { $0 + $1.evaluations },
                "constant_bytes": info.reduce(UInt64(0)) { $0 + $1.constantBytes },
                "compile_ms": info.reduce(0.0) { $0 + $1.compileMilliseconds },
                "load_ms": info.reduce(0.0) { $0 + $1.loadMilliseconds },
                "evaluation_ms": info.reduce(0.0) { $0 + $1.evaluationMilliseconds },
                "claim_boundary": "Actual shared FFN dispatches, immutable FP16 constants expanded from checkpoint affine weights; experimental numerical behavior."]
    }
}

/// The serial submission queue owns native Begin/Wait. The consumer queue waits
/// on ANE's output event after routed work, so neither CPU I/O nor routed GPU work
/// serializes behind the shared FFN. Completion is checked before returning a token.
final class ANESharedFFNLayer: @unchecked Sendable {
    let handle: OpaquePointer
    private let input: MTLBuffer
    private let output: MTLBuffer
    private let event: MTLSharedEvent
    private let packPipeline: MTLComputePipelineState
    private let gatherPipeline: MTLComputePipelineState
    private let queue = DispatchQueue(label: "ane-fieldfare.shared-ffn", qos: .userInitiated)
    private let completion = DispatchSemaphore(value: 0)
    private let positions: Int
    private var inputRows = 1
    private(set) var evaluatedRows = 0
    private var generation: UInt64 = 0
    private var pending = false
    private var settled = false
    private var failure: String?

    init(path: String, layer: Int, bits: Int, device: MTLDevice,
         positions: Int, pack: MTLComputePipelineState, gather: MTLComputePipelineState) throws {
        var error: UnsafePointer<CChar>?
        guard let pointer = ANEFieldfareANESharedFFNCreate(path, UInt32(layer), 2816, 2112, UInt32(bits), UInt32(positions), &error) else {
            let message = error.map { String(cString: $0) } ?? "program construction failed"
            free(UnsafeMutableRawPointer(mutating: error))
            throw ANEExecutionError.rejected("layer \(layer): \(message)")
        }
        func wrap(_ surface: IOSurfaceRef?) -> MTLBuffer? {
            guard let surface else { return nil }
            let base = IOSurfaceGetBaseAddress(surface)
            let bytes = IOSurfaceGetAllocSize(surface)
            guard bytes >= 2816 * positions * 2, bytes % Int(getpagesize()) == 0,
                  Int(bitPattern: base) % Int(getpagesize()) == 0 else { return nil }
            let owner = SharedSurfaceOwner(surface)
            return device.makeBuffer(bytesNoCopy: base, length: bytes, options: .storageModeShared,
                                     deallocator: { [owner] _, _ in _ = owner })
        }
        guard let input = wrap(ANEFieldfareANESharedFFNInput(pointer)?.takeUnretainedValue()),
              let output = wrap(ANEFieldfareANESharedFFNOutput(pointer)?.takeUnretainedValue()), let event = device.makeSharedEvent() else {
            ANEFieldfareANESharedFFNDestroy(pointer)
            throw ANEExecutionError.rejected("IOSurface/Metal carrier construction failed")
        }
        self.handle = pointer; self.input = input; self.output = output; self.event = event
        packPipeline = pack; gatherPipeline = gather; self.positions = positions
    }

    deinit {
        if pending && !settled { completion.wait() }
        ANEFieldfareANESharedFFNDestroy(handle)
    }

    func encodeInput(_ cb: MTLCommandBuffer, source: MTLBuffer, rows: Int = 1) throws {
        guard !pending, rows > 0, rows <= positions, source.length >= rows * 2816 * 2,
              let encoder = cb.makeComputeCommandEncoder() else {
            throw ANEExecutionError.rejected("input carrier is busy or encoder unavailable")
        }
        encoder.setComputePipelineState(packPipeline)
        encoder.setBuffer(source, offset: 0, index: 0); encoder.setBuffer(input, offset: 0, index: 1)
        inputRows = rows
        var p = UInt32(positions), r = UInt32(rows)
        encoder.setBytes(&p, length: 4, index: 2); encoder.setBytes(&r, length: 4, index: 3)
        encoder.dispatchThreads(MTLSize(width: 2816 * positions, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Caller has completed the input producer command buffer.
    func begin() throws {
        guard !pending else { throw ANEExecutionError.rejected("request already pending") }
        generation += 1
        var error: UnsafePointer<CChar>?
        guard ANEFieldfareANESharedFFNSetOutputEvent(handle, event as CFTypeRef, generation, &error) else {
            let message = error.map { String(cString: $0) } ?? "output event failed"
            free(UnsafeMutableRawPointer(mutating: error))
            throw ANEExecutionError.rejected(message)
        }
        pending = true; settled = false; failure = nil
        queue.async { [self] in
            var error: UnsafePointer<CChar>?
            let valid = ANEFieldfareANESharedFFNBegin(handle, &error) &&
                ANEFieldfareANESharedFFNWait(handle, 5_000_000_000, &error)
            if !valid {
                failure = error.map { String(cString: $0) } ?? "evaluation failed"
                // Release the consumer on failure so it can drain; finish()
                // rejects the token before generation can publish it.
                event.signaledValue = generation
            }
            free(UnsafeMutableRawPointer(mutating: error))
            completion.signal()
        }
    }

    func encodeOutput(_ cb: MTLCommandBuffer, destination: MTLBuffer) throws {
        guard pending else { throw ANEExecutionError.rejected("no pending ANE output") }
        cb.encodeWaitForEvent(event, value: generation)
        guard let encoder = cb.makeComputeCommandEncoder() else { throw ANEExecutionError.rejected("output encoder unavailable") }
        encoder.setComputePipelineState(gatherPipeline)
        encoder.setBuffer(output, offset: 0, index: 0); encoder.setBuffer(destination, offset: 0, index: 1)
        var p = UInt32(positions)
        encoder.setBytes(&p, length: 4, index: 2)
        encoder.dispatchThreads(MTLSize(width: 2816 * inputRows, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Block until the pending request settles and report its failure, if
    /// any. The request stays pending; `finish` or `abandon` releases it.
    func awaitOutcome() -> String? {
        guard pending else { return nil }
        if !settled { completion.wait(); settled = true }
        return failure
    }

    func finish() throws {
        guard pending else { return }
        let failure = awaitOutcome()
        pending = false
        if let failure { throw ANEExecutionError.rejected(failure) }
        evaluatedRows += inputRows
    }

    /// Release a request whose output the host has replaced.
    func abandon() {
        _ = awaitOutcome()
        pending = false
    }
}
