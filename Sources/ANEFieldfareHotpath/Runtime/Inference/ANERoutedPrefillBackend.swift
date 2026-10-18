import ANEFieldfareANECarrier
import Foundation
import Metal
import IOSurface

enum ANEExecutionError: Error, CustomStringConvertible {
    case rejected(String)
    var description: String {
        switch self { case .rejected(let message): return "ANE execution rejected: \(message)" }
    }
}

/// Keeps an IOSurface alive while a Metal buffer aliases its memory.
private final class SharedSurfaceOwner: @unchecked Sendable {
    let surface: IOSurfaceRef
    init(_ surface: IOSurfaceRef) { self.surface = surface }
}

/// Expert fetches are internally locked per layer streamer, so concurrent
/// lookahead reads may share the model value.
private struct ConcurrentExpertReader: @unchecked Sendable {
    let model: Model
}

private final class SubmittedANEInput: @unchecked Sendable {
    let command: MTLCommandBuffer
    init(_ command: MTLCommandBuffer) { self.command = command }
}

final class ANERoutedPrefillBackend: @unchecked Sendable {
    let layerCount: Int
    let batchExperts = 4
    /// Upcoming expert groups read concurrently. SSD throughput rises from
    /// 4.7 GB/s at queue depth 1 to 6.7 GB/s at depth 2.
    let fetchLookahead = 4
    private let argumentEncoder: MTLArgumentEncoder
    struct Job {
        let view: TensorView
        let start: Int, count: Int
        let slots: [Int]
    }
    private let handle: OpaquePointer
    private let context: MetalContext
    /// Carrier expansion and unpack run on their own queue so they overlap
    /// the host queue's shared FFN. The host clears `partials` and completes
    /// the producer of `hidden` before `runLayer`, and reads `partials` only
    /// after `runLayer` has waited for every consumer.
    private let carrierQueue: MTLCommandQueue
    private let weightsPipeline: MTLComputePipelineState
    private let inputsPipeline: MTLComputePipelineState
    private let outputsPipeline: MTLComputePipelineState
    private let submission = DispatchQueue(label: "ane-fieldfare.routed-prefill", qos: .userInitiated)
    let banks: [Bank]
    private(set) var evaluatedRows: UInt64 = 0
    private(set) var readBytes: UInt64 = 0
    private(set) var readNanos: UInt64 = 0
    private var inputGPUMilliseconds = 0.0, consumerGPUMilliseconds = 0.0
    private var terminalFailure: String?

    final class Bank: @unchecked Sendable {
        let input: MTLBuffer, output: MTLBuffer, event: MTLSharedEvent
        let arguments: MTLBuffer, metadata: MTLBuffer, validation: MTLBuffer
        var producer: MTLCommandBuffer?
        var consumer: MTLCommandBuffer?
        var pairs: MTLBuffer?
        var rows = 0
        let completed = DispatchSemaphore(value: 0)
        var generation: UInt64 = 0
        var failure: String?
        var busy = false
        init(input: MTLBuffer, output: MTLBuffer, event: MTLSharedEvent, arguments: MTLBuffer, metadata: MTLBuffer, validation: MTLBuffer) {
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

    init(context: MetalContext, layers: Int) throws {
        let library = try MetalContext.privateLibrary(device: context.device, module: "ane_prefill")
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let f = library.makeFunction(name: name) else { throw MetalError.missingFunction(name) }
            return try context.device.makeComputePipelineState(function: f)
        }
        weightsPipeline = try pipeline("ane_expert_weights")
        argumentEncoder = library.makeFunction(name: "ane_expert_weights")!.makeArgumentEncoder(bufferIndex: 0)
        inputsPipeline = try pipeline("ane_expert_inputs")
        outputsPipeline = try pipeline("ane_expert_outputs")
        var error: UnsafePointer<CChar>?
        guard let handle = ANEExpertCarrierCreate(&error) else {
            let message = error.map { String(cString: $0) } ?? "device carrier construction failed"
            free(UnsafeMutableRawPointer(mutating: error)); throw ANEExecutionError.rejected(message)
        }
        func wrap(_ raw: Unmanaged<IOSurfaceRef>?, minimum: Int) throws -> MTLBuffer {
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
                let input = try wrap(ANEExpertCarrierInput(handle, UInt32(index)), minimum: batchExperts * 2816 * 2176 * 2)
                let output = try wrap(ANEExpertCarrierOutput(handle, UInt32(index)), minimum: batchExperts * 2816 * 64 * 4)
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
        self.handle = handle; self.banks = built; self.context = context; self.layerCount = layers
    }

    deinit { ANEExpertCarrierDestroy(handle) }

    var profile: [String: Any] {
        let info = ANEExpertCarrierGetInfo(handle)
        return ["layers": layerCount, "compile_count": 1, "load_count": 1,
            "compile_ms": info.compileMilliseconds, "load_ms": info.loadMilliseconds,
            "evaluations": info.evaluations, "useful_rows": evaluatedRows,
            "evaluation_ms": info.evaluationMilliseconds, "read_bytes": readBytes,
            "read_seconds": Double(readNanos) / 1e9, "request_banks": banks.count, "experts_per_request": batchExperts,
            "input_gpu_ms": inputGPUMilliseconds, "consumer_gpu_ms": consumerGPUMilliseconds,
            "carrier_bytes": banks.reduce(0) { $0 + $1.input.length + $1.output.length },
            "input_scale": 64, "weight_pack_tile": "32x32", "fetch_lookahead": fetchLookahead,
            "claim_boundary": "Routed top-8 prefill on ANE with ordinary runtime-input weights expanded by Metal; no mutable-weight or resident-expert claim."]
    }

    func prepareBatch(bank index: Int, jobs: [Job], offsets: MoEExpertOffsets,
                      hidden: MTLBuffer, pairs: MTLBuffer) throws -> MTLCommandBuffer {
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
            guard (1...64).contains(job.count), job.start >= 0,
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
        offsetsArray.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: 2) }
        enc.setBuffer(bank.metadata, offset: 0, index: 3)
        for job in jobs { enc.useResource(job.view.buffer, usage: .read) }
        enc.dispatchThreadgroups(MTLSize(width: 88, height: 22, depth: batchExperts),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.setComputePipelineState(inputsPipeline)
        enc.setBuffer(hidden, offset: 0, index: 0); enc.setBuffer(bank.input, offset: 0, index: 1)
        enc.setBuffer(pairs, offset: 0, index: 2); enc.setBuffer(bank.metadata, offset: 0, index: 3)
        enc.setBuffer(bank.validation, offset: 0, index: 4)
        enc.dispatchThreads(MTLSize(width: batchExperts * 2816 * 64, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
        bank.producer = cb; bank.pairs = pairs; bank.rows = jobs.reduce(0) { $0 + $1.count }
        cb.commit()
        return cb
    }

    func submit(bank index: Int, producer: MTLCommandBuffer, partials: MTLBuffer,
                pairs: MTLBuffer) throws -> MTLCommandBuffer {
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
        bank.generation += 1; bank.failure = nil
        var error: UnsafePointer<CChar>?
        guard ANEExpertCarrierSetOutputEvent(handle, UInt32(index), bank.event as CFTypeRef,
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
        enc.dispatchThreads(MTLSize(width: batchExperts * 2816 * 64, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
        bank.busy = true; bank.consumer = cb
        let input = SubmittedANEInput(producer)
        submission.async { [self, bank, input] in
            input.command.waitUntilCompleted()
            var error: UnsafePointer<CChar>?
            let valid = input.command.error == nil
                && bank.validation.contents().load(as: UInt32.self) == 0
                && ANEExpertCarrierEvaluate(handle, UInt32(index), &error)
            if !valid {
                bank.failure = input.command.error.map(String.init(describing:)) ?? error.map { String(cString: $0) } ?? "non-finite ANE input or failed evaluation"
                bank.event.signaledValue = bank.generation
            }
            free(UnsafeMutableRawPointer(mutating: error)); bank.completed.signal()
        }
        return cb // The scheduler commits this AFTER the next input producer.
    }

    func finish(bank index: Int, consumer: MTLCommandBuffer) throws {
        guard banks.indices.contains(index), banks[index].busy, banks[index].consumer === consumer else {
            throw ANEExecutionError.rejected("routed bank is not pending for this consumer")
        }
        consumer.waitUntilCompleted()
        let bank = banks[index]
        bank.completed.wait()
        bank.busy = false
        defer { bank.producer = nil; bank.consumer = nil; bank.pairs = nil }
        if let failure = bank.failure { terminalFailure = failure; throw ANEExecutionError.rejected(failure) }
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

    func runLayer(model: Model, layer: Int, hidden: MTLBuffer, partials: MTLBuffer,
                  routes: PrefillMoEGroupedRoutes, pairs: MTLBuffer) async throws {
        struct Pending {
            let bank: Int, consumer: MTLCommandBuffer, jobs: [Job]
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
            if pending.count == 2 { try drain() }
            let bank = ordinal % 2
            let producer = try prepareBatch(bank: bank, jobs: staging, offsets: offsets, hidden: hidden, pairs: pairs)
            commitLast()
            let consumer = try submit(bank: bank, producer: producer, partials: partials, pairs: pairs)
            pending.append(Pending(bank: bank, consumer: consumer, jobs: staging))
            staging.removeAll(keepingCapacity: true); ordinal += 1
        }
        // Ordered lookahead: up to `fetchLookahead` upcoming groups read
        // concurrently into reserved cache slots, consumed in group order so
        // request batches are identical to a sequential fetch.
        struct Fetch {
            let plan: RoutedExpertFetchPlan
            let task: Task<[TensorView], Error>
        }
        var inflight: [Fetch] = []
        var nextGroup = 0
        func reservedSlots() -> Set<Int> {
            Set(pending.flatMap { $0.jobs.flatMap(\.slots) } + staging.flatMap(\.slots)
                + inflight.flatMap(\.plan.assignedSlots))
        }
        func launch() throws {
            while inflight.count < fetchLookahead, nextGroup < routes.groups.count {
                let expert = Int(routes.groups[nextGroup].expert)
                guard let plan = try model.planRoutedExpertsIfPossible(
                    layer: layer, experts: [expert], avoidingSlots: reservedSlots()) else { return }
                let reader = ConcurrentExpertReader(model: model)
                inflight.append(Fetch(plan: plan, task: Task { [reader, plan] in
                    try await reader.model.fetchRoutedExperts(plan: plan)
                }))
                readBytes += UInt64(plan.misses.count) * 3_358_720
                nextGroup += 1
            }
        }
        defer {
            for fetch in inflight { fetch.task.cancel() }
            while !pending.isEmpty { try? drain() }
        }
        for group in routes.groups {
            try Task.checkCancellation()
            try launch()
            while inflight.isEmpty, !pending.isEmpty { try drain(); try launch() }
            if inflight.isEmpty && !staging.isEmpty { try issue(); commitLast(); try drain(); try launch() }
            guard !inflight.isEmpty else { throw ANEExecutionError.rejected("routed cache cannot place ANE expert") }
            let fetch = inflight.removeFirst()
            let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let views = try await fetch.task.value
            readNanos += clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started
            guard let view = views.first else { throw ModelError.residentBufferWrapFailed }
            for offset in stride(from: 0, to: Int(group.pairCount), by: 64) {
                staging.append(Job(view: view, start: Int(group.pairStart) + offset,
                                   count: min(64, Int(group.pairCount) - offset), slots: fetch.plan.assignedSlots))
                if staging.count == batchExperts { try issue() }
            }
        }
        try issue(); commitLast()
        while !pending.isEmpty { try drain() }
    }
}
