import Foundation
import Metal
import Testing

@testable import ANEFieldfareHotpath

/// `ANE_FIELDFARE_TEST_CHECKPOINT`, else the repository's install location.
let installedCheckpointURL: URL? = {
    let environment = ProcessInfo.processInfo.environment["ANE_FIELDFARE_TEST_CHECKPOINT"]
    let url = environment.map { URL(fileURLWithPath: $0) } ?? URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().appendingPathComponent("../../../../checkpoints/gturbo/gemma4.gturbo")
        .standardizedFileURL
    let weights = url.appendingPathComponent("model_weights.bin").path
    return FileManager.default.fileExists(atPath: weights) ? url : nil
}()

/// Fill `buffer` with FP16 NaN on the host queue, ordered after the host's
/// own clears, as a carrier that lost its program mid-layer would leave it.
private func poison(_ buffer: any MTLBuffer, on context: MetalContext) throws {
    let commandBuffer = try #require(context.queue.makeCommandBuffer())
    let blit = try #require(commandBuffer.makeBlitCommandEncoder())
    blit.fill(buffer: buffer, range: 0..<buffer.length, value: 0x7E)
    blit.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
}

@Suite struct DeviceSupervisorPolicyTests {
    final class NoGuest: RoutedPrefillGuest {
        func runLayer(model: Model, layer: Int, hidden: any MTLBuffer, partials: any MTLBuffer,
                      routes: PrefillMoEGroupedRoutes, pairs: any MTLBuffer) async throws {}
        var profile: [String: Any] { [:] }
    }
    struct Injected: Error {}

    @Test func evictsOnlyUnderFailover() async throws {
        let closed = DeviceSupervisor(aneLayers: 2, faultPolicy: .failClosed, guest: NoGuest())
        await #expect(throws: Injected.self) { _ = try await closed.supervise(layer: 0) { throw Injected() } }
        #expect(closed.faults.map(\.action) == [.failedClosed])
        #expect(closed.routedPrefill(layer: 1) != nil)
        #expect(closed.routedPrefill(layer: 2) == nil)

        let failover = DeviceSupervisor(aneLayers: 2, faultPolicy: .failoverMetal, guest: NoGuest())
        #expect(try await !failover.supervise(layer: 1) { throw Injected() })
        #expect(failover.faults.map(\.action) == [.evictedToMetal])
        #expect(failover.routedPrefill(layer: 0) == nil, "an evicted guest is never admitted again")
        await #expect(throws: CancellationError.self) {
            _ = try await failover.supervise(layer: 0) { throw CancellationError() }
        }
        #expect(RuntimeConfiguration().faultPolicy == .failClosed)
    }
}

/// End-to-end supervisor behaviour on the installed checkpoint. The forward
/// kernels are specialised to the production shape, so the toy checkpoint
/// cannot run a full pass; these tests are skipped when the checkpoint is
/// absent. Injected guests corrupt the host's buffers and then fault.
/// Failover must reproduce the matching placement's logits bit for bit.
@Suite(.serialized, .enabled(if: installedCheckpointURL != nil, "installed checkpoint required"))
struct DeviceSupervisorFailoverTests {
    /// Delegates to a real guest (or does nothing), then faults on `faultLayer`
    /// after corrupting the partials, as a carrier that lost its program would.
    final class FaultAtLayerGuest: RoutedPrefillGuest {
        let inner: any RoutedPrefillGuest?
        let faultLayer: Int
        let context: MetalContext
        init(inner: any RoutedPrefillGuest?, faultLayer: Int, context: MetalContext) {
            self.inner = inner; self.faultLayer = faultLayer; self.context = context
        }
        func runLayer(model: Model, layer: Int, hidden: any MTLBuffer, partials: any MTLBuffer,
                      routes: PrefillMoEGroupedRoutes, pairs: any MTLBuffer) async throws {
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

    struct Fixture {
        let model: Model
        let context: MetalContext
        let config = RuntimeConfiguration(prefillChunkTokens: 64, headPath: .logits)
        let tokens: [Int32] = (0..<40).map { Int32(1_000 + 731 * $0) }

        init() throws {
            context = try MetalContext()
            model = try Model.load(directoryURL: try #require(installedCheckpointURL), device: context.device)
        }

        func runner(_ supervisor: DeviceSupervisor) throws -> RealForwardRunner {
            try RealForwardRunner(model: model, context: context, maxContext: 256,
                                  runtimeConfiguration: config, supervisor: supervisor)
        }

        /// Prefill the prompt, then decode two tokens; return every logit row.
        func logits(_ supervisor: DeviceSupervisor) async throws -> [[UInt16]] {
            let runner = try runner(supervisor)
            let vocab = model.config.vocabSize
            let buffer = try #require(context.device.makeBuffer(
                length: vocab * MemoryLayout<Float16>.size, options: .storageModeShared))
            func snapshot() -> [UInt16] {
                unsafe Array(UnsafeBufferPointer(start: buffer.contents().bindMemory(to: UInt16.self, capacity: vocab),
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

    @Test func routedPrefillFailoverMatchesMetalBitExactly() async throws {
        let fixture = try Fixture()
        let reference = try await fixture.logits(DeviceSupervisor(aneLayers: 0, faultPolicy: .failClosed, guest: nil))
        #expect(reference[0].contains { $0 != 0 })

        // A guest re-admitted after eviction would run as a no-op and leave
        // the routed partials unwritten, so `recovered` would differ.
        let guest = FaultAtLayerGuest(inner: nil, faultLayer: 0, context: fixture.context)
        let supervisor = DeviceSupervisor(aneLayers: fixture.model.config.numLayers,
                                          faultPolicy: .failoverMetal, guest: guest)
        let recovered = try await fixture.logits(supervisor)
        #expect(recovered == reference)
        #expect(supervisor.faults.map(\.action) == [.evictedToMetal])
        #expect(supervisor.faults.first?.layer == 0)
    }

    /// A real ANE guest evicted at layer `k` must leave exactly the output of
    /// a plan that placed only layers `0..<k` on ANE.
    @Test func midRunANEEvictionMatchesPrefixPlacement() async throws {
        let fixture = try Fixture()
        let k = 7
        let reference = try await fixture.logits(DeviceSupervisor(
            aneLayers: k, faultPolicy: .failClosed,
            guest: try ANERoutedPrefillBackend(context: fixture.context, layers: k)))

        let layers = fixture.model.config.numLayers
        let guest = FaultAtLayerGuest(
            inner: try ANERoutedPrefillBackend(context: fixture.context, layers: layers), faultLayer: k,
            context: fixture.context)
        let supervisor = DeviceSupervisor(aneLayers: layers, faultPolicy: .failoverMetal, guest: guest)
        let recovered = try await fixture.logits(supervisor)
        #expect(recovered == reference)
        #expect(supervisor.faults.map(\.layer) == [k])
        #expect(recovered != (try await fixture.logits(DeviceSupervisor(aneLayers: 0, faultPolicy: .failClosed, guest: nil))),
                "ANE layers before the fault should differ from pure Metal arithmetic")
    }
}
