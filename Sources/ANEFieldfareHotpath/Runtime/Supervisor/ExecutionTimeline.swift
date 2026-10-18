import Foundation
import Metal

/// Opt-in interval trace of the resources the routed prefill pipeline
/// contends for: disk reads, GPU command buffers and ANE evaluations.
/// Enabled by `ANE_FIELDFARE_TIMELINE=<path>`; the runner writes the trace
/// when its profile is read. All times share the host uptime clock that
/// Metal's `gpuStartTime`/`gpuEndTime` use.
final class ExecutionTimeline: @unchecked Sendable {
    static let shared: ExecutionTimeline? = ProcessInfo.processInfo.environment["ANE_FIELDFARE_TIMELINE"]
        .map { ExecutionTimeline(path: $0) }

    struct Interval { let lane: String; let label: String; let start: Double; let end: Double }

    let path: String
    private let lock = NSLock()
    private var intervals: [Interval] = []

    init(path: String) { self.path = path }

    static func now() -> Double { Double(clock_gettime_nsec_np(CLOCK_UPTIME_RAW)) / 1e9 }

    func record(_ lane: String, _ label: String, start: Double, end: Double) {
        lock.lock(); intervals.append(Interval(lane: lane, label: label, start: start, end: end)); lock.unlock()
    }

    /// Record a command buffer's GPU interval once it completes.
    func track(_ commandBuffer: MTLCommandBuffer, _ label: String) {
        commandBuffer.addCompletedHandler { [self] cb in
            record("gpu", label, start: cb.gpuStartTime, end: cb.gpuEndTime)
        }
    }

    func write() {
        lock.lock(); let snapshot = intervals; lock.unlock()
        let rows = snapshot.map { ["lane": $0.lane, "label": $0.label, "start": $0.start, "end": $0.end] as [String: Any] }
        if let data = try? JSONSerialization.data(withJSONObject: rows) {
            try? data.write(to: URL(fileURLWithPath: path))
        }
    }
}
