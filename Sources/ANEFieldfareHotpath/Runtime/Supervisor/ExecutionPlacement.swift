import Foundation

/// A device rail the supervisor can place work on. Metal is the host rail: it
/// owns attention, routing, layer tails and logits, and is the numerical
/// reference every guest placement falls back to.
public enum ExecutionRail: String, Codable, Sendable {
    case metal
    case ane
}

/// The operations whose placement is negotiable. Everything else in the
/// forward pass is pinned to the Metal host rail.
public enum PlacedOperation: String, CaseIterable, Codable, Sendable {
    /// Routed top-8 experts during chunked prefill.
    case routedExpertsPrefill = "routed-experts-prefill"
    /// The dense shared FFN during chunked prefill.
    case sharedFFNPrefill = "shared-ffn-prefill"
    /// The dense shared FFN during recurrent decode.
    case sharedFFNDecode = "shared-ffn-decode"
    /// Routed top-8 experts during recurrent decode (compiler-authored AOT
    /// procedures).
    case routedExpertsDecode = "routed-experts-decode"
}

/// What the supervisor does when a guest rail reports a fault.
public enum RuntimeFaultPolicy: String, Codable, Sendable, CaseIterable {
    /// Abort the request, as the runtime always has. A guest fault can never
    /// publish a token.
    case failClosed = "fail-closed"
    /// Evict the faulted guest for the rest of the runner's life and execute
    /// the affected layer, and every later one, on the Metal host rail. Output
    /// after a failover is Metal arithmetic and is labelled in the profile.
    case failoverMetal = "failover-metal"
}

public enum ExecutionPlacementError: Error, CustomStringConvertible, Equatable {
    case invalidLayerCount(variable: String, value: String, maximum: Int)
    case invalidSharedMode(String)

    public var description: String {
        switch self {
        case .invalidLayerCount(let variable, let value, let maximum):
            return "\(variable)=\(value) must be an integer in 0...\(maximum)"
        case .invalidSharedMode(let value):
            return "ANE_FIELDFARE_ANE_SHARED_MODE=\(value) must be prefill, decode, or both"
        }
    }
}

/// The resolved, immutable answer to "which rail runs operation X on layer L".
/// Guests always cover a prefix of layers `0..<n`; the remaining layers stay
/// on Metal. This is the only place that reads process environment for
/// placement, so the forward loop consumes typed decisions only.
public struct ExecutionPlacementPlan: Equatable, Sendable {
    /// Number of leading layers each operation runs on ANE. Absent means 0.
    public let aneLayers: [PlacedOperation: Int]
    /// Artifact root for compiler-authored decode experts, when placed.
    public let aotExpertRoot: String?

    public static let metalOnly = ExecutionPlacementPlan(aneLayers: [:], aotExpertRoot: nil)

    public init(aneLayers: [PlacedOperation: Int], aotExpertRoot: String? = nil) {
        self.aneLayers = aneLayers.filter { $0.value > 0 }
        self.aotExpertRoot = aotExpertRoot
    }

    public func rail(for operation: PlacedOperation, layer: Int) -> ExecutionRail {
        layer < aneLayers[operation, default: 0] ? .ane : .metal
    }

    public func aneLayerCount(_ operation: PlacedOperation) -> Int {
        aneLayers[operation, default: 0]
    }

    public var usesANE: Bool { !aneLayers.isEmpty }

    /// Resolve placement from the CLI backend and the experiment variables.
    ///
    /// - `--backend hybrid-ane` places routed prefill on ANE for every layer.
    /// - `ANE_FIELDFARE_ANE_ROUTED_PREFILL_LAYERS=n` overrides that count, for
    ///   either backend.
    /// - `ANE_FIELDFARE_ANE_SHARED_LAYERS=n` with
    ///   `ANE_FIELDFARE_ANE_SHARED_MODE=prefill|decode|both` (default prefill)
    ///   places the shared FFN.
    /// - `ANE_FIELDFARE_AOT_EXPERT_ROOT=<dir>` places decode experts on every
    ///   layer through the AOT provider.
    public static func resolve(backend: RuntimeBackend,
                               numLayers: Int,
                               environment: [String: String] = ProcessInfo.processInfo.environment) throws -> ExecutionPlacementPlan {
        func layerCount(_ variable: String) throws -> Int? {
            guard let raw = environment[variable] else { return nil }
            guard let count = Int(raw), (0...numLayers).contains(count) else {
                throw ExecutionPlacementError.invalidLayerCount(
                    variable: variable, value: raw, maximum: numLayers)
            }
            return count
        }
        var layers: [PlacedOperation: Int] = [:]
        layers[.routedExpertsPrefill] = try layerCount("ANE_FIELDFARE_ANE_ROUTED_PREFILL_LAYERS")
            ?? (backend == .hybridANE ? numLayers : 0)
        if let shared = try layerCount("ANE_FIELDFARE_ANE_SHARED_LAYERS") {
            let mode = environment["ANE_FIELDFARE_ANE_SHARED_MODE"] ?? "prefill"
            guard ["prefill", "decode", "both"].contains(mode) else {
                throw ExecutionPlacementError.invalidSharedMode(mode)
            }
            if mode != "decode" { layers[.sharedFFNPrefill] = shared }
            if mode != "prefill" { layers[.sharedFFNDecode] = shared }
        }
        let aotRoot = environment["ANE_FIELDFARE_AOT_EXPERT_ROOT"].flatMap { $0.isEmpty ? nil : $0 }
        if aotRoot != nil { layers[.routedExpertsDecode] = numLayers }
        return ExecutionPlacementPlan(aneLayers: layers, aotExpertRoot: aotRoot)
    }

    var profile: [String: Any] {
        Dictionary(uniqueKeysWithValues: PlacedOperation.allCases.map {
            ($0.rawValue, ["rail": aneLayerCount($0) > 0 ? "ane" : "metal",
                           "ane_layers": aneLayerCount($0)] as [String: Any])
        })
    }
}
