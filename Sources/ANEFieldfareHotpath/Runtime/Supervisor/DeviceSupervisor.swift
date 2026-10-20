import Foundation
import Metal

/// What the supervisor does when the ANE guest reports a fault.
public enum RuntimeFaultPolicy: String, Codable, Sendable, CaseIterable {
    /// Abort the request. A guest fault can never publish a token.
    case failClosed = "fail-closed"
    /// Evict the guest for the rest of the runner's life and execute the
    /// faulted layer, and every later one, on Metal. The profile labels output
    /// after a failover as mixed ANE and Metal arithmetic.
    case failoverMetal = "failover-metal"
}

/// Routed top-8 prefill for one layer, on a rail other than Metal. On return
/// every routed `(token, rank)` row named by `pairs` has been written to
/// `partials`. On throw the guest has drained its own in-flight work, so the
/// host may overwrite `partials` immediately.
protocol RoutedPrefillGuest: AnyObject {
    func runLayer(model: Model, layer: Int, hidden: any MTLBuffer, partials: any MTLBuffer,
                  routes: PrefillMoEGroupedRoutes, pairs: any MTLBuffer) async throws
    var profile: [String: Any] { get }
}

extension ANERoutedPrefillBackend: RoutedPrefillGuest {}

/// One guest fault the supervisor observed, and what it did about it.
struct GuestFault: Equatable {
    enum Action: String {
        case failedClosed = "failed-closed"
        case evictedToMetal = "evicted-to-metal"
    }
    let layer: Int
    let message: String
    let action: Action
}

/// Owns the ANE guest for one runner. Metal is the host rail: it runs
/// attention, routing, the shared FFN, layer tails, logits and decode, and
/// keeps a complete routed-expert implementation. The guest runs routed
/// prefill experts for layers `0..<aneLayers`.
///
/// The forward loop asks `routedPrefill(layer:)` whether the guest is
/// admitted, and runs it through `supervise`, which applies the fault policy.
/// An evicted guest is never admitted again, because a faulted ANE program
/// may have lost its carrier state. Construction fails closed: a requested
/// placement whose program or shape is unavailable aborts startup instead of
/// silently running on Metal.
final class DeviceSupervisor {
    let aneLayers: Int
    let faultPolicy: RuntimeFaultPolicy
    private let guest: any RoutedPrefillGuest?
    private(set) var evicted = false
    private(set) var faults: [GuestFault] = []

    /// `--backend hybrid-ane` places every layer, or layers `0..<n` under
    /// `ANE_FIELDFARE_ANE_ROUTED_PREFILL_LAYERS=n`; `metal` places none.
    static func aneLayerCount(backend: RuntimeBackend, numLayers: Int) throws -> Int {
        guard backend == .hybridANE else { return 0 }
        let variable = "ANE_FIELDFARE_ANE_ROUTED_PREFILL_LAYERS"
        guard let raw = ProcessInfo.processInfo.environment[variable] else { return numLayers }
        guard let count = Int(raw), (0...numLayers).contains(count) else {
            throw RuntimeEnvironmentError(variable: variable, value: raw, expected: "an integer in 0...\(numLayers)")
        }
        return count
    }

    convenience init(configuration: RuntimeConfiguration, model: Model, context: MetalContext) throws {
        let layers = try Self.aneLayerCount(backend: configuration.backend, numLayers: model.config.numLayers)
        var guest: any RoutedPrefillGuest?
        if layers > 0 {
            try ANERoutedPrefillBackend.validate(config: model.config)
            guard configuration.activeRoutedExperts == model.config.topKExperts,
                  configuration.prefillExpertUnionCap == nil else {
                throw ANEExecutionError.rejected("ANE routed prefill requires exact uncapped top-8")
            }
            guest = try ANERoutedPrefillBackend(context: context, layers: layers)
        }
        self.init(aneLayers: layers, faultPolicy: configuration.faultPolicy, guest: guest)
    }

    /// Places `guest` on layers `0..<aneLayers`; tests inject faulting guests.
    init(aneLayers: Int, faultPolicy: RuntimeFaultPolicy, guest: any RoutedPrefillGuest?) {
        self.aneLayers = guest == nil ? 0 : aneLayers
        self.faultPolicy = faultPolicy
        self.guest = guest
    }

    func routedPrefill(layer: Int) -> any RoutedPrefillGuest? {
        !evicted && layer < aneLayers ? guest : nil
    }

    /// Run the guest under the fault policy. Returns true when the guest
    /// completed, false when it was evicted and the host must execute the
    /// layer. Fail-closed rethrows; cancellation always propagates.
    func supervise(layer: Int, _ body: () async throws -> Void) async throws -> Bool {
        do {
            try await body()
            return true
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            let closed = faultPolicy == .failClosed
            faults.append(GuestFault(layer: layer, message: String(describing: error),
                                     action: closed ? .failedClosed : .evictedToMetal))
            if closed { throw error }
            evicted = true
            return false
        }
    }

    /// The ANE guest's profile; nil when no layer runs on the ANE.
    var routedPrefillProfile: [String: Any]? { guest?.profile }

    var profile: [String: Any] {
        [
            "fault_policy": faultPolicy.rawValue,
            "ane_layers": aneLayers,
            "evicted": evicted,
            "faults": faults.map { ["layer": $0.layer, "message": $0.message, "action": $0.action.rawValue] },
            "claim_boundary": evicted
                ? "The ANE guest was evicted; layers from the fault on ran on Metal, so output mixes ANE and Metal arithmetic."
                : "Placement as planned; no guest eviction.",
        ]
    }
}
