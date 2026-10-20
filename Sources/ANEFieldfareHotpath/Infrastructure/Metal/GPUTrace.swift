import Darwin
import Foundation
import Metal
import Synchronization

/// A timeline of every compute and blit encoder's GPU execution and of
/// named host spans, on one clock, for `ANE_FIELDFARE_GPU_TRACE=<path>`.
///
/// Encoders made through `makeTracedComputeEncoder` / `makeTracedBlitEncoder`
/// carry a start and end timestamp sampled at the encoder's stage
/// boundaries, so each interval is the encoder's own GPU time inside the
/// real, unsplit command buffers. Metal's timestamps are `CLOCK_UPTIME_RAW`
/// nanoseconds, the clock host spans use. `finish` writes a Chrome trace
/// (open it at ui.perfetto.dev); `Scripts/profile/summarize.py` reduces it.
///
/// Sampling costs the GPU about 4 µs between encoders, so gaps between
/// encoders read slightly long. With the variable unset, every entry point
/// is one nil check.
public enum GPUTrace {
    static let sink: GPUTraceSink? = ProcessInfo.processInfo
        .environment["ANE_FIELDFARE_GPU_TRACE"].map(GPUTraceSink.init(path:))

    /// The host-side lane a span is drawn on.
    public enum Track: String, Sendable {
        case host
        case ane
        case io
    }

    /// Labels encoders made from now on, e.g. "decode L5".
    public static func setScope(_ scope: @autoclosure () -> String) {
        sink?.setScope(scope())
    }

    /// A span start for `end`; zero when tracing is off.
    public static func begin() -> UInt64 {
        sink == nil ? 0 : clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
    }

    public static func end(_ name: @autoclosure () -> String, since start: UInt64,
                           track: Track = .host) {
        guard let sink else { return }
        sink.span(name: name(), start: start, end: clock_gettime_nsec_np(CLOCK_UPTIME_RAW),
                  track: track)
    }

    /// Moves encoders the GPU has finished into the timeline, freeing their
    /// sample slots. Call it after waiting on submitted work.
    public static func collect() {
        sink?.collect()
    }

    /// Writes the trace; returns its path, or nil when tracing is off. Every
    /// traced command buffer must have completed.
    @discardableResult
    public static func finish() throws -> String? {
        try sink?.write()
    }

    static func callSiteName(_ fileID: StaticString, _ function: StaticString, _ line: UInt) -> String {
        let file = "\(fileID)".split(separator: "/").last.map { $0.split(separator: ".").first ?? $0 } ?? ""
        let fn = "\(function)".split(separator: "(").first ?? ""
        return "\(file).\(fn):\(line)"
    }
}

extension MTLCommandBuffer {
    /// `makeComputeCommandEncoder()`, timestamped when GPU tracing is on.
    func makeTracedComputeEncoder(fileID: StaticString = #fileID,
                                  function: StaticString = #function,
                                  line: UInt = #line) -> (any MTLComputeCommandEncoder)? {
        guard let sink = GPUTrace.sink,
              let slot = sink.reserve(device: device,
                                      name: GPUTrace.callSiteName(fileID, function, line)) else {
            return makeComputeCommandEncoder()
        }
        let pass = MTLComputePassDescriptor()
        guard let attachment = pass.sampleBufferAttachments[0] else { return makeComputeCommandEncoder() }
        attachment.sampleBuffer = slot.samples.buffer
        attachment.startOfEncoderSampleIndex = slot.index
        attachment.endOfEncoderSampleIndex = slot.index + 1
        return makeComputeCommandEncoder(descriptor: pass)
    }

    /// `makeBlitCommandEncoder()`, timestamped when GPU tracing is on.
    func makeTracedBlitEncoder(fileID: StaticString = #fileID,
                               function: StaticString = #function,
                               line: UInt = #line) -> (any MTLBlitCommandEncoder)? {
        guard let sink = GPUTrace.sink,
              let slot = sink.reserve(device: device,
                                      name: GPUTrace.callSiteName(fileID, function, line)) else {
            return makeBlitCommandEncoder()
        }
        let pass = MTLBlitPassDescriptor()
        guard let attachment = pass.sampleBufferAttachments[0] else { return makeBlitCommandEncoder() }
        attachment.sampleBuffer = slot.samples.buffer
        attachment.startOfEncoderSampleIndex = slot.index
        attachment.endOfEncoderSampleIndex = slot.index + 1
        return makeBlitCommandEncoder(descriptor: pass)
    }
}

/// One counter sample buffer, shared between the encoders that sample into
/// it and the trace that resolves it.
struct GPUTraceSamples: @unchecked Sendable {
    // SAFETY: the host never mutates the buffer. Encoders only store a
    // reference to it in their pass descriptor; the GPU writes each index
    // pair once, for the one encoder that reserved it; `resolve` runs only
    // under the trace's mutex and treats an unwritten (zero) timestamp as
    // still pending, so a sample is never read before it is written.
    let buffer: any MTLCounterSampleBuffer
}

final class GPUTraceSink: Sendable {
    /// Metal allows 32 live counter sample buffers per process; leave room
    /// for anything else that samples.
    static let maxBuffers = 24
    static let samplesPerBuffer = 4096

    struct Slot: Sendable {
        let samples: GPUTraceSamples
        let index: Int
    }

    struct Event: Sendable {
        let name: String
        let scope: String
        let start: UInt64
        let end: UInt64
        let track: String
    }

    private struct Buffer {
        let samples: GPUTraceSamples
        var used = 0
        var pending = 0
    }

    private struct Pending {
        let buffer: Int
        let index: Int
        let name: String
        let scope: String
        let reserved: UInt64
    }

    private struct State {
        var buffers: [Int: Buffer] = [:]
        var current: Int?
        var nextBufferID = 0
        var pending: [Pending] = []
        var events: [Event] = []
        var scope = ""
        var dropped = 0
    }

    let path: String
    private let state = Mutex(State())

    init(path: String) {
        self.path = path
    }

    func setScope(_ scope: String) {
        state.withLock { $0.scope = scope }
    }

    func span(name: String, start: UInt64, end: UInt64, track: GPUTrace.Track) {
        state.withLock { state in
            state.events.append(Event(name: name, scope: state.scope, start: start, end: end,
                                      track: track.rawValue))
        }
    }

    /// Reserves a start/end sample pair, or nil when every buffer is full of
    /// samples the GPU has not written yet (the encoder then goes untraced).
    func reserve(device: any MTLDevice, name: String) -> Slot? {
        let reserved = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        return state.withLock { state -> Slot? in
            if state.current.flatMap({ state.buffers[$0] }).map({ $0.used + 2 > Self.samplesPerBuffer }) ?? true {
                Self.collect(&state)
                guard state.buffers.count < Self.maxBuffers,
                      let samples = Self.makeSamples(device: device) else {
                    state.current = nil
                    state.dropped += 1
                    return nil
                }
                state.buffers[state.nextBufferID] = Buffer(samples: samples)
                state.current = state.nextBufferID
                state.nextBufferID += 1
            }
            guard let id = state.current, var buffer = state.buffers[id] else { return nil }
            let index = buffer.used
            buffer.used += 2
            buffer.pending += 1
            state.buffers[id] = buffer
            state.pending.append(Pending(buffer: id, index: index, name: name, scope: state.scope,
                                         reserved: reserved))
            return Slot(samples: buffer.samples, index: index)
        }
    }

    func collect() {
        state.withLock { Self.collect(&$0) }
    }

    func write() throws -> String {
        let (events, dropped, unresolved) = state.withLock { state in
            Self.collect(&state)
            return (state.events, state.dropped, state.pending.count)
        }
        var trace: [[String: Any]] = [
            ["name": "process_name", "ph": "M", "pid": 1, "args": ["name": "ANE Fieldfare"]],
        ]
        // Encoders can overlap on the GPU; give each a lane where it does not.
        var laneEnds: [UInt64] = []
        let tracks = ["host": 1, "ane": 2, "io": 3]
        for (name, tid) in tracks {
            trace.append(["name": "thread_name", "ph": "M", "pid": 1, "tid": tid, "args": ["name": name]])
        }
        for event in events.sorted(by: { $0.start < $1.start }) {
            let tid: Int
            if let host = tracks[event.track] {
                tid = host
            } else {
                let lane = laneEnds.firstIndex { $0 <= event.start } ?? laneEnds.count
                if lane == laneEnds.count {
                    laneEnds.append(0)
                    trace.append(["name": "thread_name", "ph": "M", "pid": 1, "tid": 10 + lane,
                                  "args": ["name": "gpu \(lane)"]])
                }
                laneEnds[lane] = event.end
                tid = 10 + lane
            }
            trace.append([
                "name": event.name, "cat": event.track, "ph": "X", "pid": 1, "tid": tid,
                "ts": Double(event.start) / 1000, "dur": Double(event.end - event.start) / 1000,
                "args": ["scope": event.scope],
            ])
        }
        let root: [String: Any] = [
            "traceEvents": trace,
            "displayTimeUnit": "ns",
            "otherData": ["dropped_encoders": dropped, "unresolved_encoders": unresolved],
        ]
        let data = try JSONSerialization.data(withJSONObject: root)
        try data.write(to: URL(fileURLWithPath: path))
        return path
    }

    private static func makeSamples(device: any MTLDevice) -> GPUTraceSamples? {
        guard let set = device.counterSets?.first(where: { $0.name == MTLCommonCounterSet.timestamp.rawValue })
        else { return nil }
        let descriptor = MTLCounterSampleBufferDescriptor()
        descriptor.counterSet = set
        descriptor.sampleCount = samplesPerBuffer
        descriptor.storageMode = .shared
        return (try? device.makeCounterSampleBuffer(descriptor: descriptor)).map(GPUTraceSamples.init)
    }

    /// Records every pending encoder whose timestamps are written, then drops
    /// buffers that are full and fully resolved.
    private static func collect(_ state: inout State) {
        guard !state.pending.isEmpty else { return }
        var resolved: [Int: [UInt64]] = [:]
        var still: [Pending] = []
        for item in state.pending {
            if resolved[item.buffer] == nil, let buffer = state.buffers[item.buffer] {
                resolved[item.buffer] = timestamps(buffer.samples, count: buffer.used)
            }
            let values = resolved[item.buffer] ?? []
            guard item.index + 1 < values.count else { still.append(item); continue }
            let start = values[item.index], end = values[item.index + 1]
            // Zero is unwritten; anything before the reservation is stale.
            guard start >= item.reserved, end >= start, end != MTLCounterErrorValue else {
                still.append(item)
                continue
            }
            state.events.append(Event(name: item.name, scope: item.scope, start: start, end: end,
                                      track: "gpu"))
            state.buffers[item.buffer]?.pending -= 1
        }
        state.pending = still
        for (id, buffer) in state.buffers
        where buffer.pending == 0 && buffer.used + 2 > samplesPerBuffer {
            state.buffers[id] = nil
            if state.current == id { state.current = nil }
        }
    }

    private static func timestamps(_ samples: GPUTraceSamples, count: Int) -> [UInt64] {
        guard count > 0, let data = try? samples.buffer.resolveCounterRange(0..<count) else { return [] }
        return unsafe data.withUnsafeBytes { raw in
            unsafe Array(raw.bindMemory(to: MTLCounterResultTimestamp.self)).map(\.timestamp)
        }
    }
}
