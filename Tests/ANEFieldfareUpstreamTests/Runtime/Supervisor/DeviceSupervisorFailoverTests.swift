import Foundation
import Metal
import Testing

@testable import ANEFieldfareHotpath

/// `ANE_FIELDFARE_TEST_CHECKPOINT`, else the repository's install location.
private let installedCheckpointURL: URL? = {
    let environment = ProcessInfo.processInfo.environment["ANE_FIELDFARE_TEST_CHECKPOINT"]
    let url = environment.map { URL(fileURLWithPath: $0) } ?? URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().appendingPathComponent("../../../../checkpoints/gturbo/gemma4.gturbo")
        .standardizedFileURL
    let weights = url.appendingPathComponent("model_weights.bin").path
    return FileManager.default.fileExists(atPath: weights) ? url : nil
}()

/// Fill `buffer` with FP16 NaN on the host queue, ordered after the host's
/// own clears, as a carrier that lost its program mid-layer would leave it.
private func poison(_ buffer: MTLBuffer, on context: MetalContext) throws {
    let commandBuffer = try #require(context.queue.makeCommandBuffer())
    let blit = try #require(commandBuffer.makeBlitCommandEncoder())
    blit.fill(buffer: buffer, range: 0..<buffer.length, value: 0x7E)
    blit.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
}

/// End-to-end supervisor behaviour on the installed checkpoint. The forward
/// kernels are specialised to the production shape, so the toy checkpoint
/// cannot run a full pass; these tests are skipped when the checkpoint is
/// absent. Injected guests corrupt the host's buffers and then fault.
/// Failover must reproduce the matching placement's logits bit for bit, and
/// fail-closed must abort.
@Suite(.serialized, .enabled(if: installedCheckpointURL != nil, "installed checkpoint required"))
struct DeviceSupervisorFailoverTests {
    /// Delegates to a real guest, then faults on `faultLayer` after
    /// corrupting the partials, as a carrier that lost its program would.
    final class FaultAtLayerGuest: RoutedPrefillGuest {
        let inner: RoutedPrefillGuest?
        let faultLayer: Int
        let context: MetalContext
        init(inner: RoutedPrefillGuest?, faultLayer: Int, context: MetalContext) {
            self.inner = inner; self.faultLayer = faultLayer; self.context = context
        }
        func runLayer(model: Model, layer: Int, hidden: MTLBuffer, partials: MTLBuffer,
                      routes: PrefillMoEGroupedRoutes, pairs: MTLBuffer) async throws {
            guard layer == faultLayer else {
                try await inner?.runLayer(model: model, layer: layer, hidden: hidden, partials: partials,
                                          routes: routes, pairs: pairs)
                return
            }
            try poison(partials, on: context)
            throw ANEExecutionError.rejected("injected routed fault at layer \(layer)")
        }
        var profile: [String: Any] { ["injected": true] }
    }

    /// Writes NaN over every routed partial row, then faults, as a carrier
    /// that lost its program mid-layer would.
    /// A guest that faults on its first layer without running anything real.
    final class CorruptingRoutedGuest: RoutedPrefillGuest {
        let context: MetalContext
        var calls = 0
        init(context: MetalContext) { self.context = context }
        func runLayer(model: Model, layer: Int, hidden: MTLBuffer, partials: MTLBuffer,
                      routes: PrefillMoEGroupedRoutes, pairs: MTLBuffer) async throws {
            calls += 1
            try poison(partials, on: context)
            throw ANEExecutionError.rejected("injected routed fault")
        }
        var profile: [String: Any] { ["injected": true] }
    }

    /// A shared-FFN guest whose requests always settle with a failure.
    final class FailingSharedGuest: SharedFFNGuest {
        final class Layer: SharedFFNGuestLayer {
            var pending = false
            var consumed = 0
            func encodeInput(_ cb: MTLCommandBuffer, source: MTLBuffer, rows: Int) throws {}
            func begin() throws { pending = true }
            func awaitOutcome() -> String? { pending ? "injected shared fault" : nil }
            func encodeOutput(_ cb: MTLCommandBuffer, destination: MTLBuffer) throws { consumed += 1 }
            func finish() throws {
                guard pending else { return }
                pending = false
                throw ANEExecutionError.rejected("injected shared fault")
            }
            func abandon() { pending = false }
        }
        let layers: [Layer]
        init(count: Int) { layers = (0..<count).map { _ in Layer() } }
        func layer(_ index: Int, prefill: Bool) -> SharedFFNGuestLayer? {
            index < layers.count ? layers[index] : nil
        }
        var profile: [String: Any] { ["injected": true] }
    }

    struct Fixture {
        let model: Model
        let context: MetalContext
        let config = RuntimeConfiguration(prefillChunkTokens: 64, forceLogitsHead: true)
        let tokens: [Int32] = (0..<40).map { Int32(1_000 + 731 * $0) }

        init() throws {
            context = try MetalContext()
            model = try Model.load(directoryURL: try #require(installedCheckpointURL), device: context.device,
                                   integrityPolicy: .sizeCheckTrustedReceipt)
        }

        func runner(_ supervisor: DeviceSupervisor) throws -> RealForwardRunner {
            try RealForwardRunner(model: model, context: context, maxContext: 256,
                                  runtimeConfiguration: config,
                                  prefillBoundaryOracle: nil, supervisor: supervisor)
        }

        /// Prefill the prompt, then decode two tokens; return every logit row.
        func logits(_ supervisor: DeviceSupervisor) async throws -> [[UInt16]] {
            let runner = try runner(supervisor)
            let vocab = model.config.vocabSize
            let buffer = try #require(context.device.makeBuffer(
                length: vocab * MemoryLayout<Float16>.size, options: .storageModeShared))
            func snapshot() -> [UInt16] {
                Array(UnsafeBufferPointer(start: buffer.contents().bindMemory(to: UInt16.self, capacity: vocab),
                                          count: vocab))
            }
            _ = try await runner.prefillChunked(tokens: tokens[...], startPosition: 0, outputMode: .logits,
                                                config: config.prefillConfig, into: buffer, onProgress: { _ in })
            var rows = [snapshot()]
            for (offset, token) in [Int32(7), 11].enumerated() {
                try await runner.produce(token: token, position: tokens.count + offset, into: buffer)
                rows.append(snapshot())
            }
            return rows
        }
    }

    static func allLayers(_ operations: [PlacedOperation], _ count: Int) -> ExecutionPlacementPlan {
        ExecutionPlacementPlan(aneLayers: Dictionary(uniqueKeysWithValues: operations.map { ($0, count) }))
    }

    @Test func routedPrefillFailoverMatchesMetalBitExactly() async throws {
        let fixture = try Fixture()
        let reference = try await fixture.logits(.metalOnly())
        #expect(reference[0].contains { $0 != 0 })

        let guest = CorruptingRoutedGuest(context: fixture.context)
        let supervisor = DeviceSupervisor(
            plan: Self.allLayers([.routedExpertsPrefill], fixture.model.config.numLayers),
            faultPolicy: .failoverMetal, routedPrefill: guest)
        let recovered = try await fixture.logits(supervisor)
        #expect(recovered == reference)
        #expect(guest.calls == 1, "an evicted guest must never be admitted again")
        #expect(supervisor.faults.map(\.action) == [.evictedToMetal])
        #expect(supervisor.faults.first?.layer == 0)
    }

    @Test func routedPrefillFaultFailsClosedByDefault() async throws {
        let fixture = try Fixture()
        let supervisor = DeviceSupervisor(
            plan: Self.allLayers([.routedExpertsPrefill], fixture.model.config.numLayers),
            faultPolicy: .failClosed, routedPrefill: CorruptingRoutedGuest(context: fixture.context))
        await #expect(throws: ANEExecutionError.self) {
            _ = try await fixture.logits(supervisor)
        }
        #expect(supervisor.faults.map(\.action) == [.failedClosed])
    }

    @Test func sharedFFNFailoverMatchesMetalBitExactly() async throws {
        let fixture = try Fixture()
        let reference = try await fixture.logits(.metalOnly())
        let layers = fixture.model.config.numLayers
        for operation in [PlacedOperation.sharedFFNPrefill, .sharedFFNDecode] {
            let guest = FailingSharedGuest(count: layers)
            let supervisor = DeviceSupervisor(
                plan: Self.allLayers([operation], layers),
                faultPolicy: .failoverMetal, sharedFFN: guest)
            let recovered = try await fixture.logits(supervisor)
            #expect(guest.layers.allSatisfy { $0.consumed == 0 }, "host consumed a failed guest output")
            #expect(recovered == reference, "\(operation.rawValue)")
            #expect(supervisor.faults.count == 1, "\(operation.rawValue)")
            #expect(supervisor.faults.first?.operation == operation)
        }
    }

    @Test func sharedFFNFaultFailsClosedByDefault() async throws {
        let fixture = try Fixture()
        let layers = fixture.model.config.numLayers
        let supervisor = DeviceSupervisor(
            plan: Self.allLayers([.sharedFFNPrefill], layers),
            faultPolicy: .failClosed, sharedFFN: FailingSharedGuest(count: layers))
        await #expect(throws: ANEExecutionError.self) {
            _ = try await fixture.logits(supervisor)
        }
    }

    /// A real ANE guest evicted at layer `k` must leave exactly the output of
    /// a plan that placed only layers `0..<k` on ANE.
    @Test func midRunANEEvictionMatchesPrefixPlacement() async throws {
        let fixture = try Fixture()
        let k = 7
        let prefix = Self.allLayers([.routedExpertsPrefill], k)
        let reference = try await fixture.logits(DeviceSupervisor(
            plan: prefix, faultPolicy: .failClosed,
            routedPrefill: try ANERoutedPrefillBackend(context: fixture.context, layers: k)))

        let layers = fixture.model.config.numLayers
        let guest = FaultAtLayerGuest(
            inner: try ANERoutedPrefillBackend(context: fixture.context, layers: layers), faultLayer: k,
            context: fixture.context)
        let supervisor = DeviceSupervisor(
            plan: Self.allLayers([.routedExpertsPrefill], layers),
            faultPolicy: .failoverMetal, routedPrefill: guest)
        let recovered = try await fixture.logits(supervisor)
        #expect(recovered == reference)
        #expect(supervisor.faults.map(\.layer) == [k])
        #expect(recovered != (try await fixture.logits(.metalOnly())),
                "ANE layers before the fault should differ from pure Metal arithmetic")
    }
}
