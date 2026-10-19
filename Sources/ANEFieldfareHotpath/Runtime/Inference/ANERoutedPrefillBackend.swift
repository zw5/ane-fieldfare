import ANEFieldfareANECarrier
import Foundation
import Metal
import IOSurface
import Synchronization

enum ANEExecutionError: Error, CustomStringConvertible {
    case rejected(String)
    var description: String {
        switch self { case .rejected(let message): return "ANE execution rejected: \(message)" }
    }
}

/// Keeps an IOSurface alive while a Metal buffer aliases its memory.
private final class SharedSurfaceOwner: @unchecked Sendable {
    // SAFETY: a lifetime token for Metal's `@Sendable` deallocator. Nothing
    // reads or writes through `surface`; it is only released with the owner.
    let surface: IOSurfaceRef
    init(_ surface: IOSurfaceRef) { self.surface = surface }
}

/// The compiled carrier program. Destroyed when the backend and every
/// in-flight evaluation have let go of it.
private final class ANECarrierProgram {
    let handle: OpaquePointer
    init(_ handle: OpaquePointer) { self.handle = handle }
    deinit { ANEExpertCarrierDestroy(handle) }
}

/// How one bank's evaluation ended, signalled once by the submission queue.
private final class ANEEvaluationOutcome: Sendable {
    private let failure = Mutex<String?>(nil)
    private let settled = DispatchSemaphore(value: 0)

    func settle(failure message: String?) {
        failure.withLock { $0 = message }
        settled.signal()
    }

    /// Blocks until `settle`, then returns its failure message, if any.
    func wait() -> String? {
        settled.wait()
        return failure.withLock { $0 }
    }
}

/// Everything the submission queue needs to evaluate one bank.
private struct ANEEvaluation: @unchecked Sendable {
    // SAFETY: `submit` builds one per evaluation and hands it to the
    // submission queue exactly once. Until `outcome` settles, the bank is
    // `busy`: `prepareBatch` and `submit` refuse it and `finish` waits on
    // `outcome` first, so the issuing side never touches `producer`,
    // `validation` or `event` while the queue does. Carrier evaluations are
    // serialized by that queue, as `ANEExpertCarrierEvaluate` requires.
    let program: ANECarrierProgram
    let bank: Int
    let producer: any MTLCommandBuffer
    let validation: any MTLBuffer
    let event: any MTLSharedEvent
    let generation: UInt64
    let outcome: ANEEvaluationOutcome

    /// Waits for the input producer, then evaluates on the ANE. A failed
    /// evaluation still signals `event`, so the consumer never hangs on it.
    func run() {
        producer.waitUntilCompleted()
        var error: UnsafePointer<CChar>?
        let valid = producer.error == nil
            && validation.contents().load(as: UInt32.self) == 0
            && ANEExpertCarrierEvaluate(program.handle, UInt32(bank), &error)
        var failure: String?
        if !valid {
            failure = producer.error.map(String.init(describing:))
                ?? error.map { String(cString: $0) }
                ?? "non-finite ANE input or failed evaluation"
            event.signaledValue = generation
        }
        free(UnsafeMutableRawPointer(mutating: error))
        outcome.settle(failure: failure)
    }
}

final class ANERoutedPrefillBackend {
    let layerCount: Int
    let batchExperts = 4
    /// Activation positions per expert in one request. Each request streams
    /// all four experts' expanded weights (about 0.95 ms), so more lanes
    /// amortize that transfer over more rows; past 128 the ANE is compute
    /// bound and padding outweighs the saving.
    let lanes: Int
    /// Upcoming expert groups read concurrently. SSD throughput rises from
    /// 4.7 GB/s at queue depth 1 to 6.7 GB/s at depth 2.
    let fetchLookahead = 4
    private let argumentEncoder: any MTLArgumentEncoder
    struct Job {
        let view: TensorView
        let start: Int, count: Int
    }
    private let program: ANECarrierProgram
    private let context: MetalContext
    /// Carrier expansion and unpack run on their own queue so they overlap
    /// the host queue's shared FFN. The host clears `partials` and completes
    /// the producer of `hidden` before `runLayer`, and reads `partials` only
    /// after `runLayer` has waited for every consumer.
    private let carrierQueue: any MTLCommandQueue
    private let weightsPipeline: any MTLComputePipelineState
    private let inputsPipeline: any MTLComputePipelineState
    private let outputsPipeline: any MTLComputePipelineState
    // SAFETY: a dedicated serial queue, not a task. Each evaluation blocks
    // for up to the carrier's 5 s timeout and evaluations must not overlap;
    // a serial queue gives both without parking a cooperative-pool thread.
    private let submission = DispatchQueue(label: "ane-fieldfare.routed-prefill", qos: .userInitiated)
    let banks: [Bank]
    private(set) var evaluatedRows: UInt64 = 0
    private(set) var readBytes: UInt64 = 0
    private(set) var readNanos: UInt64 = 0
    private var inputGPUMilliseconds = 0.0, consumerGPUMilliseconds = 0.0
    private var terminalFailure: String?

    final class Bank {
        let input: any MTLBuffer, output: any MTLBuffer, event: any MTLSharedEvent
        let arguments: any MTLBuffer, metadata: any MTLBuffer, validation: any MTLBuffer
        var producer: any MTLCommandBuffer?
        var consumer: any MTLCommandBuffer?
        var pairs: any MTLBuffer?
        var rows = 0
        var generation: UInt64 = 0
        /// Set while an evaluation is out on the submission queue.
        fileprivate var evaluation: ANEEvaluationOutcome?
        var busy: Bool { evaluation != nil }
        init(input: any MTLBuffer, output: any MTLBuffer, event: any MTLSharedEvent, arguments: any MTLBuffer, metadata: any MTLBuffer, validation: any MTLBuffer) {
            self.input = input; self.output = output; self.event = event
            self.arguments = arguments; self.metadata = metadata
            self.validation = validation
        }
    }

    /// Shape contract for the device carrier. Placement (how many layers) is
    /// resolved by `DeviceSupervisor`.
    static func validate(config: ArchConfig) throws {
        guard config.hiddenSize == 2816, config.moeIntermediateSize == 704, config.topKExperts == 8 else {
            throw ANEExecutionError.rejected("ANE routed prefill requires H=2816, F_moe=704, K=8")
        }
    }

    init(context: MetalContext, layers: Int, lanes: Int = 128) throws {
        self.lanes = lanes
        let library = try MetalContext.privateLibrary(device: context.device, module: "ane_prefill")
        let constants = MTLFunctionConstantValues()
        var laneCount = UInt32(lanes)
        constants.setConstantValue(&laneCount, type: .uint, index: 80)
        func function(_ name: String) throws -> any MTLFunction {
            try library.makeFunction(name: name, constantValues: constants)
        }
        func pipeline(_ name: String) throws -> any MTLComputePipelineState {
            try context.device.makeComputePipelineState(function: try function(name))
        }
        weightsPipeline = try pipeline("ane_expert_weights")
        argumentEncoder = try function("ane_expert_weights").makeArgumentEncoder(bufferIndex: 0)
        inputsPipeline = try pipeline("ane_expert_inputs")
        outputsPipeline = try pipeline("ane_expert_outputs")
        var error: UnsafePointer<CChar>?
        guard let handle = ANEExpertCarrierCreate(UInt32(lanes), &error) else {
            let message = error.map { String(cString: $0) } ?? "device carrier construction failed"
            free(UnsafeMutableRawPointer(mutating: error)); throw ANEExecutionError.rejected(message)
        }
        func wrap(_ raw: Unmanaged<IOSurfaceRef>?, minimum: Int) throws -> any MTLBuffer {
            guard let surface = raw?.takeUnretainedValue() else { throw ModelError.residentBufferWrapFailed }
            let base = IOSurfaceGetBaseAddress(surface), bytes = IOSurfaceGetAllocSize(surface)
            guard bytes >= minimum, bytes % Int(getpagesize()) == 0,
                  Int(bitPattern: base) % Int(getpagesize()) == 0 else { throw ModelError.residentBufferWrapFailed }
            let owner = SharedSurfaceOwner(surface)
            guard let buffer = context.device.makeBuffer(bytesNoCopy: base, length: bytes, options: .storageModeShared,
                deallocator: { [owner] _, _ in _ = owner }) else { throw ModelError.residentBufferWrapFailed }
            return buffer
        }
        var built: [Bank] = []
        do {
            for index in 0..<2 {
                let input = try wrap(ANEExpertCarrierInput(handle, UInt32(index)), minimum: batchExperts * 2816 * (lanes + 2112) * 2)
                let output = try wrap(ANEExpertCarrierOutput(handle, UInt32(index)), minimum: batchExperts * 2816 * lanes * 4)
                guard let event = context.device.makeSharedEvent() else { throw ModelError.residentBufferWrapFailed }
                guard let arguments = context.device.makeBuffer(length: argumentEncoder.encodedLength, options: .storageModeShared),
                      let metadata = context.device.makeBuffer(length: 4 * 8, options: .storageModeShared),
                      let validation = context.device.makeBuffer(length: 4, options: .storageModeShared) else {
                    throw ModelError.residentBufferWrapFailed
                }
                built.append(Bank(input: input, output: output, event: event, arguments: arguments, metadata: metadata, validation: validation))
            }
        } catch { ANEExpertCarrierDestroy(handle); throw error }
        guard let carrierQueue = context.device.makeCommandQueue() else {
            ANEExpertCarrierDestroy(handle); throw ModelError.residentBufferWrapFailed
        }
        carrierQueue.label = "ane-fieldfare.routed-prefill.carrier"
        self.carrierQueue = carrierQueue
        self.program = ANECarrierProgram(handle); self.banks = built; self.context = context; self.layerCount = layers
    }

    var profile: [String: Any] {
        let info = ANEExpertCarrierGetInfo(program.handle)
        return ["layers": layerCount, "compile_count": 1, "load_count": 1,
            "compile_ms": info.compileMilliseconds, "load_ms": info.loadMilliseconds,
            "evaluations": info.evaluations, "useful_rows": evaluatedRows,
            "evaluation_ms": info.evaluationMilliseconds, "read_bytes": readBytes,
            "read_seconds": Double(readNanos) / 1e9, "request_banks": banks.count, "experts_per_request": batchExperts,
            "input_gpu_ms": inputGPUMilliseconds, "consumer_gpu_ms": consumerGPUMilliseconds,
            "carrier_bytes": banks.reduce(0) { $0 + $1.input.length + $1.output.length },
            "input_scale": 64, "lanes": lanes, "weight_pack_tile": "32x32", "fetch_lookahead": fetchLookahead,
            "claim_boundary": "Routed top-8 prefill on ANE with ordinary runtime-input weights expanded by Metal; no mutable-weight or resident-expert claim."]
    }

    func prepareBatch(bank index: Int, jobs: [Job], offsets: MoEExpertOffsets,
                      hidden: any MTLBuffer, pairs: any MTLBuffer) throws -> any MTLCommandBuffer {
        if let terminalFailure { throw ANEExecutionError.rejected("backend requires reconstruction after: \(terminalFailure)") }
        guard banks.indices.contains(index), !banks[index].busy, banks[index].producer == nil,
              !jobs.isEmpty, jobs.count <= batchExperts else {
            throw ANEExecutionError.rejected("routed bank busy or batch invalid")
        }
        let offsetsArray = [offsets.gateWOff, offsets.gateSOff, offsets.gateBOff,
                            offsets.upWOff, offsets.upSOff, offsets.upBOff,
                            offsets.downWOff, offsets.downSOff, offsets.downBOff]
        let pairValues = pairs.contents().assumingMemoryBound(to: PrefillTokenExpertPair.self)
        var destinations = Set<UInt64>()
        for job in jobs {
            guard (1...lanes).contains(job.count), job.start >= 0,
                  job.start <= pairs.length / MemoryLayout<PrefillTokenExpertPair>.stride - job.count,
                  job.view.offset <= UInt64(job.view.buffer.length), job.view.length >= 3_358_720,
                  job.view.length <= UInt64(job.view.buffer.length) - job.view.offset,
                  offsetsArray.enumerated().allSatisfy({ UInt64($0.element) + ($0.offset % 3 == 0 ? 991232 : 61952) <= job.view.length }) else {
                throw ANEExecutionError.rejected("invalid routed carrier spans")
            }
            for p in job.start..<(job.start + job.count) {
                guard (UInt64(pairValues[p].token) + 1) * 2816 * 2 <= hidden.length, pairValues[p].rank < 8 else {
                    throw ANEExecutionError.rejected("invalid routed input row")
                }
                guard destinations.insert(UInt64(pairValues[p].token) * 8 + UInt64(pairValues[p].rank)).inserted else {
                    throw ANEExecutionError.rejected("duplicate routed output row")
                }
            }
        }
        let bank = banks[index]
        bank.validation.contents().storeBytes(of: UInt32(0), as: UInt32.self)
        let metadata = bank.metadata.contents().assumingMemoryBound(to: UInt32.self)
        argumentEncoder.setArgumentBuffer(bank.arguments, offset: 0)
        for slot in 0..<4 {
            let job = slot < jobs.count ? jobs[slot] : nil
            metadata[slot * 2] = UInt32(job?.start ?? 0); metadata[slot * 2 + 1] = UInt32(job?.count ?? 0)
            argumentEncoder.setBuffer(job?.view.buffer, offset: Int(job?.view.offset ?? 0), index: slot)
        }
        guard let cb = carrierQueue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw ModelError.residentBufferWrapFailed
        }
        enc.setComputePipelineState(weightsPipeline)
        enc.setBuffer(bank.arguments, offset: 0, index: 0); enc.setBuffer(bank.input, offset: 0, index: 1)
        enc.setBytes(offsetsArray, length: offsetsArray.count * MemoryLayout<UInt32>.stride, index: 2)
        enc.setBuffer(bank.metadata, offset: 0, index: 3)
        for job in jobs { enc.useResource(job.view.buffer, usage: .read) }
        enc.dispatchThreadgroups(MTLSize(width: 88, height: 22, depth: batchExperts),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.setComputePipelineState(inputsPipeline)
        enc.setBuffer(hidden, offset: 0, index: 0); enc.setBuffer(bank.input, offset: 0, index: 1)
        enc.setBuffer(pairs, offset: 0, index: 2); enc.setBuffer(bank.metadata, offset: 0, index: 3)
        enc.setBuffer(bank.validation, offset: 0, index: 4)
        enc.dispatchThreads(MTLSize(width: batchExperts * 2816 * lanes, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
        bank.producer = cb; bank.pairs = pairs; bank.rows = jobs.reduce(0) { $0 + $1.count }
        cb.commit()
        return cb
    }

    func submit(bank index: Int, producer: any MTLCommandBuffer, partials: any MTLBuffer,
                pairs: any MTLBuffer) throws -> any MTLCommandBuffer {
        guard banks.indices.contains(index), !banks[index].busy,
              banks[index].producer === producer, banks[index].pairs === pairs else {
            throw ANEExecutionError.rejected("routed request bank is busy or invalid")
        }
        let bank = banks[index]
        defer {
            if !bank.busy {
                producer.waitUntilCompleted()
                bank.producer = nil; bank.pairs = nil
            }
        }
        let metadata = bank.metadata.contents().assumingMemoryBound(to: UInt32.self)
        let pairValues = pairs.contents().assumingMemoryBound(to: PrefillTokenExpertPair.self)
        for slot in 0..<batchExperts {
            for p in Int(metadata[slot * 2])..<(Int(metadata[slot * 2]) + Int(metadata[slot * 2 + 1])) {
                let pair = pairValues[p]
                guard pair.rank < 8,
                      ((UInt64(pair.token) * 8 + UInt64(pair.rank)) + 1) * 2816 * 2 <= partials.length else {
                    throw ANEExecutionError.rejected("invalid routed output row")
                }
            }
        }
        bank.generation += 1
        var error: UnsafePointer<CChar>?
        guard ANEExpertCarrierSetOutputEvent(program.handle, UInt32(index), bank.event as CFTypeRef,
                                                           bank.generation, &error) else {
            let message = error.map { String(cString: $0) } ?? "output signal failed"
            free(UnsafeMutableRawPointer(mutating: error)); throw ANEExecutionError.rejected(message)
        }
        guard let cb = carrierQueue.makeCommandBuffer() else { throw ModelError.residentBufferWrapFailed }
        cb.encodeWaitForEvent(bank.event, value: bank.generation)
        guard let enc = cb.makeComputeCommandEncoder() else { throw ModelError.residentBufferWrapFailed }
        enc.setComputePipelineState(outputsPipeline)
        enc.setBuffer(bank.output, offset: 0, index: 0); enc.setBuffer(partials, offset: 0, index: 1)
        enc.setBuffer(pairs, offset: 0, index: 2); enc.setBuffer(bank.metadata, offset: 0, index: 3)
        enc.setBuffer(bank.validation, offset: 0, index: 4)
        enc.dispatchThreads(MTLSize(width: batchExperts * 2816 * lanes, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
        let outcome = ANEEvaluationOutcome()
        bank.evaluation = outcome; bank.consumer = cb
        let evaluation = ANEEvaluation(
            program: program, bank: index, producer: producer, validation: bank.validation,
            event: bank.event, generation: bank.generation, outcome: outcome)
        submission.async { evaluation.run() }
        return cb // The scheduler commits this AFTER the next input producer.
    }

    func finish(bank index: Int, consumer: any MTLCommandBuffer) throws {
        guard banks.indices.contains(index), let evaluation = banks[index].evaluation,
              banks[index].consumer === consumer else {
            throw ANEExecutionError.rejected("routed bank is not pending for this consumer")
        }
        consumer.waitUntilCompleted()
        let bank = banks[index]
        let failure = evaluation.wait()
        bank.evaluation = nil
        defer { bank.producer = nil; bank.consumer = nil; bank.pairs = nil }
        if let failure { terminalFailure = failure; throw ANEExecutionError.rejected(failure) }
        if let error = consumer.error { terminalFailure = String(describing: error); throw error }
        guard bank.validation.contents().load(as: UInt32.self) == 0 else {
            terminalFailure = "non-finite ANE routed output"
            throw ANEExecutionError.rejected("non-finite ANE routed output")
        }
        if let producer = bank.producer {
            inputGPUMilliseconds += max(0, producer.gpuEndTime - producer.gpuStartTime) * 1000
        }
        consumerGPUMilliseconds += max(0, consumer.gpuEndTime - consumer.gpuStartTime) * 1000
        evaluatedRows += UInt64(bank.rows)
    }

    func runLayer(model: Model, layer: Int, hidden: any MTLBuffer, partials: any MTLBuffer,
                  routes: PrefillMoEGroupedRoutes, pairs: any MTLBuffer) async throws {
        struct Pending {
            let bank: Int, consumer: any MTLCommandBuffer, jobs: [Job]
            var committed = false
        }
        var pending: [Pending] = []
        var staging: [Job] = []
        var ordinal = 0
        let offsets = model.routedExpertOffsets(layer: layer)
        func commitLast() {
            if let last = pending.indices.last, !pending[last].committed {
                pending[last].consumer.commit(); pending[last].committed = true
            }
        }
        func drain() throws {
            let first = pending.removeFirst()
            if !first.committed { first.consumer.commit() }
            try withExtendedLifetime(first.jobs) { try finish(bank: first.bank, consumer: first.consumer) }
        }
        func issue() throws {
            guard !staging.isEmpty else { return }
            if pending.count == banks.count { try drain() }
            let bank = ordinal % banks.count
            let producer = try prepareBatch(bank: bank, jobs: staging, offsets: offsets, hidden: hidden, pairs: pairs)
            commitLast()
            let consumer = try submit(bank: bank, producer: producer, partials: partials, pairs: pairs)
            pending.append(Pending(bank: bank, consumer: consumer, jobs: staging))
            staging.removeAll(keepingCapacity: true); ordinal += 1
        }
        // Ordered lookahead: up to `fetchLookahead` upcoming groups read
        // concurrently, consumed in group order so request batches are
        // identical to a sequential fetch. The reads are child tasks that
        // capture only the layer mapping, so leaving this scope, by returning
        // or by throwing, waits for every read still in flight.
        let mapped = try model.expertLayer(layer)
        try await withThrowingTaskGroup(of: Int.self) { reads in
            var inflight: [Int] = []
            var landed = Set<Int>()
            var nextGroup = 0
            func launch() throws {
                while inflight.count < fetchLookahead, nextGroup < routes.groups.count {
                    let absent = try model.nonResidentRoutedExperts(
                        layer: layer, experts: [Int(routes.groups[nextGroup].expert)])
                    readBytes += UInt64(absent.count * model.routedExpertBytes)
                    let ordinal = nextGroup
                    reads.addTask {
                        try await Model.read(absent, of: mapped)
                        return ordinal
                    }
                    inflight.append(ordinal)
                    nextGroup += 1
                }
            }
            defer {
                while !pending.isEmpty { try? drain() }
            }
            for group in routes.groups {
                try Task.checkCancellation()
                try launch()
                let ordinal = inflight.removeFirst()
                let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                while landed.remove(ordinal) == nil {
                    guard let read = try await reads.next() else {
                        throw ANEExecutionError.rejected("routed expert read was never started")
                    }
                    landed.insert(read)
                }
                readNanos += clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started
                guard let view = try model.routedExpertViews(
                    layer: layer, experts: [Int(group.expert)]).first else {
                    throw ModelError.residentBufferWrapFailed
                }
                for offset in stride(from: 0, to: Int(group.pairCount), by: lanes) {
                    staging.append(Job(view: view, start: Int(group.pairStart) + offset,
                                       count: min(lanes, Int(group.pairCount) - offset)))
                    if staging.count == batchExperts { try issue() }
                }
            }
            try issue(); commitLast()
            while !pending.isEmpty { try drain() }
        }
    }
}
