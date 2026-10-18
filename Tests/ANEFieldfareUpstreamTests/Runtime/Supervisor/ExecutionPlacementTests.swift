import Foundation
import Testing

@testable import ANEFieldfareHotpath

@Suite struct ExecutionPlacementTests {
    @Test func metalBackendPlacesNothingOnANE() throws {
        let plan = try ExecutionPlacementPlan.resolve(backend: .metal, numLayers: 30, environment: [:])
        #expect(plan == .metalOnly)
        #expect(!plan.usesANE)
        for operation in PlacedOperation.allCases {
            #expect(plan.rail(for: operation, layer: 0) == .metal)
        }
    }

    @Test func hybridBackendPlacesEveryRoutedPrefillLayer() throws {
        let plan = try ExecutionPlacementPlan.resolve(backend: .hybridANE, numLayers: 30, environment: [:])
        #expect(plan.aneLayerCount(.routedExpertsPrefill) == 30)
        #expect(plan.rail(for: .routedExpertsPrefill, layer: 29) == .ane)
        #expect(plan.rail(for: .sharedFFNPrefill, layer: 0) == .metal)
        #expect(plan.rail(for: .routedExpertsDecode, layer: 0) == .metal)
    }

    @Test func environmentOverridesAndPrefixPlacement() throws {
        let plan = try ExecutionPlacementPlan.resolve(backend: .metal, numLayers: 30, environment: [
            "ANE_FIELDFARE_ANE_ROUTED_PREFILL_LAYERS": "12",
            "ANE_FIELDFARE_ANE_SHARED_LAYERS": "4",
            "ANE_FIELDFARE_ANE_SHARED_MODE": "both",
        ])
        #expect(plan.rail(for: .routedExpertsPrefill, layer: 11) == .ane)
        #expect(plan.rail(for: .routedExpertsPrefill, layer: 12) == .metal)
        #expect(plan.aneLayerCount(.sharedFFNPrefill) == 4)
        #expect(plan.aneLayerCount(.sharedFFNDecode) == 4)

        let zero = try ExecutionPlacementPlan.resolve(backend: .hybridANE, numLayers: 30,
            environment: ["ANE_FIELDFARE_ANE_ROUTED_PREFILL_LAYERS": "0"])
        #expect(zero == .metalOnly)

        let decodeOnly = try ExecutionPlacementPlan.resolve(backend: .metal, numLayers: 30,
            environment: ["ANE_FIELDFARE_ANE_SHARED_LAYERS": "30", "ANE_FIELDFARE_ANE_SHARED_MODE": "decode"])
        #expect(decodeOnly.aneLayerCount(.sharedFFNPrefill) == 0)
        #expect(decodeOnly.aneLayerCount(.sharedFFNDecode) == 30)

        let aot = try ExecutionPlacementPlan.resolve(backend: .metal, numLayers: 30,
            environment: ["ANE_FIELDFARE_AOT_EXPERT_ROOT": "/artifacts"])
        #expect(aot.aotExpertRoot == "/artifacts")
        #expect(aot.rail(for: .routedExpertsDecode, layer: 29) == .ane)
        let emptyAOT = try ExecutionPlacementPlan.resolve(backend: .metal, numLayers: 30,
            environment: ["ANE_FIELDFARE_AOT_EXPERT_ROOT": ""])
        #expect(emptyAOT == .metalOnly)
    }

    @Test func invalidPlacementFailsClosed() {
        #expect(throws: ExecutionPlacementError.self) {
            try ExecutionPlacementPlan.resolve(backend: .metal, numLayers: 30,
                environment: ["ANE_FIELDFARE_ANE_ROUTED_PREFILL_LAYERS": "31"])
        }
        #expect(throws: ExecutionPlacementError.self) {
            try ExecutionPlacementPlan.resolve(backend: .metal, numLayers: 30,
                environment: ["ANE_FIELDFARE_ANE_SHARED_LAYERS": "x"])
        }
        #expect(throws: ExecutionPlacementError.self) {
            try ExecutionPlacementPlan.resolve(backend: .metal, numLayers: 30,
                environment: ["ANE_FIELDFARE_ANE_SHARED_LAYERS": "2", "ANE_FIELDFARE_ANE_SHARED_MODE": "always"])
        }
    }

    @Test func supervisorEvictsOnlyUnderFailover() async throws {
        let plan = ExecutionPlacementPlan(aneLayers: [.sharedFFNPrefill: 2, .sharedFFNDecode: 2])
        struct Injected: Error {}

        let closed = DeviceSupervisor(plan: plan, faultPolicy: .failClosed)
        await #expect(throws: Injected.self) {
            _ = try await closed.supervise(.sharedFFNPrefill, layer: 0) { throw Injected() }
        }
        #expect(closed.faults.map(\.action) == [.failedClosed])

        let failover = DeviceSupervisor(plan: plan, faultPolicy: .failoverMetal)
        let completed = try await failover.supervise(.sharedFFNPrefill, layer: 1) { throw Injected() }
        #expect(!completed)
        #expect(failover.faults.map(\.action) == [.evictedToMetal])
        // Both shared phases share one program set, so both are evicted.
        let evicted = try #require(failover.profile["evicted"] as? [String])
        #expect(evicted == ["shared-ffn-decode", "shared-ffn-prefill"])
        #expect(try await failover.supervise(.sharedFFNPrefill, layer: 0) {})

        await #expect(throws: CancellationError.self) {
            _ = try await failover.supervise(.routedExpertsPrefill, layer: 0) { throw CancellationError() }
        }
    }

    @Test func faultPolicyFlagParses() throws {
        #expect(RuntimeFaultPolicy(rawValue: "failover-metal") == .failoverMetal)
        #expect(RuntimeConfiguration.production.faultPolicy == .failClosed)
    }
}
