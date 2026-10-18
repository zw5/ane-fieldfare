import Foundation

/// One guest fault the supervisor observed, and what it did about it.
public struct GuestFault: Sendable, Equatable {
    public enum Action: String, Sendable {
        /// The request was aborted (fail-closed policy).
        case failedClosed = "failed-closed"
        /// The guest was evicted and the layer re-executed on Metal.
        case evictedToMetal = "evicted-to-metal"
    }
    public let operation: PlacedOperation
    public let layer: Int
    public let message: String
    public let action: Action
}

/// The device supervisor owns every guest rail for one runner.
///
/// The forward loop never holds a guest directly. For each placeable
/// operation it asks the supervisor which guest, if any, is admitted for the
/// layer, runs the guest, and reports the outcome. The supervisor applies the
/// fault policy: fail closed (rethrow), or evict the guest and tell the caller
/// to execute the layer on the Metal host rail. An evicted guest is never
/// admitted again, because a faulted ANE program may have lost its carrier
/// state; reconstruction means building a new runner.
///
/// Guests are constructed eagerly and fail closed: an explicit placement
/// whose program, artifact or shape is unavailable aborts runner
/// construction instead of silently running on Metal.
final class DeviceSupervisor {
    let plan: ExecutionPlacementPlan
    let faultPolicy: RuntimeFaultPolicy

    private let routedPrefillGuest: RoutedPrefillGuest?
    private let sharedFFNGuest: SharedFFNGuest?
    private let routedDecodeGuest: RoutedDecodeGuest?
    private var evicted: Set<PlacedOperation> = []
    private(set) var faults: [GuestFault] = []

    /// Build the guests the plan places, after checking the runtime
    /// constraints that each guest's arithmetic depends on.
    convenience init(plan: ExecutionPlacementPlan,
                     faultPolicy: RuntimeFaultPolicy,
                     model: Model,
                     context: MetalContext,
                     configuration: RuntimeConfiguration,
                     routedReuseEnabled: Bool) throws {
        let cfg = model.config
        let exactTop8 = configuration.activeRoutedExperts == cfg.topKExperts
        var decode: RoutedDecodeGuest?
        if let root = plan.aotExpertRoot {
            guard exactTop8 else {
                throw AOTExpertBackendError.incompatibleShape(
                    "AOT expert backend requires exact top-8 routing")
            }
            decode = try AOTExpertBackend(root: root, config: cfg)
        }
        var routed: RoutedPrefillGuest?
        let routedLayers = plan.aneLayerCount(.routedExpertsPrefill)
        if routedLayers > 0 {
            try ANERoutedPrefillBackend.validate(config: cfg)
            guard exactTop8, configuration.prefillExpertUnionCap == nil else {
                throw ANEExecutionError.rejected("ANE routed prefill requires exact uncapped top-8")
            }
            routed = try ANERoutedPrefillBackend(context: context, layers: routedLayers)
        }
        var shared: SharedFFNGuest?
        let sharedPrefill = plan.aneLayerCount(.sharedFFNPrefill)
        let sharedDecode = plan.aneLayerCount(.sharedFFNDecode)
        if sharedPrefill > 0 || sharedDecode > 0 {
            guard decode == nil, !routedReuseEnabled else {
                throw ANEExecutionError.rejected(
                    "shared FFN overlap cannot combine with experimental routed AOT/reuse")
            }
            shared = try ANESharedFFNBackend(
                model: model, context: context, count: max(sharedPrefill, sharedDecode),
                prefill: sharedPrefill > 0, decode: sharedDecode > 0,
                prefillTokens: configuration.prefillChunkTokens)
        }
        self.init(plan: plan, faultPolicy: faultPolicy,
                  routedPrefill: routed, sharedFFN: shared, routedDecode: decode)
    }

    /// Inject guests directly. Tests use this to place fault-injecting guests
    /// on checkpoints the ANE carriers cannot run.
    init(plan: ExecutionPlacementPlan,
         faultPolicy: RuntimeFaultPolicy,
         routedPrefill: RoutedPrefillGuest? = nil,
         sharedFFN: SharedFFNGuest? = nil,
         routedDecode: RoutedDecodeGuest? = nil) {
        self.plan = plan
        self.faultPolicy = faultPolicy
        self.routedPrefillGuest = routedPrefill
        self.sharedFFNGuest = sharedFFN
        self.routedDecodeGuest = routedDecode
    }

    static func metalOnly() -> DeviceSupervisor {
        DeviceSupervisor(plan: .metalOnly, faultPolicy: .failClosed)
    }

    private func admitted(_ operation: PlacedOperation, layer: Int) -> Bool {
        !evicted.contains(operation) && plan.rail(for: operation, layer: layer) == .ane
    }

    // MARK: Admission

    func routedPrefill(layer: Int) -> RoutedPrefillGuest? {
        admitted(.routedExpertsPrefill, layer: layer) ? routedPrefillGuest : nil
    }

    func sharedFFN(layer: Int, prefill: Bool) -> SharedFFNGuestLayer? {
        let operation: PlacedOperation = prefill ? .sharedFFNPrefill : .sharedFFNDecode
        guard admitted(operation, layer: layer) else { return nil }
        return sharedFFNGuest?.layer(layer, prefill: prefill)
    }

    func routedDecode(layer: Int) -> RoutedDecodeGuest? {
        admitted(.routedExpertsDecode, layer: layer) ? routedDecodeGuest : nil
    }

    /// True when the host should check a shared-FFN guest's outcome before
    /// consuming its output, so a fault can be replaced by Metal arithmetic.
    /// Fail-closed keeps the original overlap: the check happens at `finish`.
    var checksSharedOutcomeBeforeConsume: Bool { faultPolicy == .failoverMetal }

    // MARK: Faults

    /// Record a guest fault. Under fail-closed this rethrows `error`; under
    /// failover it evicts the operation (both phases, for the shared FFN,
    /// since they share one ANE program set) and returns, and the caller must
    /// execute the layer on Metal.
    func fault(_ operation: PlacedOperation, layer: Int, error: Error) throws {
        try fault(operation, layer: layer, message: String(describing: error), rethrowing: error)
    }

    func fault(_ operation: PlacedOperation, layer: Int, message: String) throws {
        try fault(operation, layer: layer, message: message,
                  rethrowing: ANEExecutionError.rejected(message))
    }

    private func fault(_ operation: PlacedOperation, layer: Int, message: String,
                       rethrowing error: Error) throws {
        switch faultPolicy {
        case .failClosed:
            faults.append(GuestFault(operation: operation, layer: layer,
                                     message: message, action: .failedClosed))
            throw error
        case .failoverMetal:
            faults.append(GuestFault(operation: operation, layer: layer,
                                     message: message, action: .evictedToMetal))
            evicted.insert(operation)
            if operation == .sharedFFNPrefill || operation == .sharedFFNDecode {
                evicted.formUnion([.sharedFFNPrefill, .sharedFFNDecode])
            }
        }
    }

    /// Run a guest operation under the fault policy. Returns true when the
    /// guest completed, false when it was evicted and the host must execute.
    func supervise(_ operation: PlacedOperation, layer: Int,
                   _ body: () async throws -> Void) async throws -> Bool {
        do {
            try await body()
            return true
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            try fault(operation, layer: layer, error: error)
            return false
        }
    }

    // MARK: Reporting

    var routedPrefillProfile: [String: Any] {
        routedPrefillGuest?.profile ?? ["layers": 0, "evaluations": 0]
    }

    var sharedFFNProfile: [String: Any] {
        sharedFFNGuest?.profile ?? ["layers": 0, "evaluations": 0]
    }

    var profile: [String: Any] {
        [
            "fault_policy": faultPolicy.rawValue,
            "placement": plan.profile,
            "evicted": evicted.map(\.rawValue).sorted(),
            "faults": faults.map {
                ["operation": $0.operation.rawValue, "layer": $0.layer,
                 "message": $0.message, "action": $0.action.rawValue]
            },
            "claim_boundary": faults.contains { $0.action == .evictedToMetal }
                ? "A guest was evicted; layers after the fault ran on the Metal host rail, so output mixes ANE and Metal arithmetic."
                : "Placement as planned; no guest eviction.",
        ]
    }
}
