import Testing
import Foundation
import Metal
@testable import ANEFieldfareHotpath

/// `Model.load` integration for routed experts mapped from the page cache.
@Suite struct ModelLoaderMappedExpertTests {

    private static func readBytes(_ view: TensorView) -> [UInt8] {
        let base = unsafe view.buffer.contents().advanced(by: Int(view.offset))
        return unsafe [UInt8](UnsafeRawBufferPointer(start: base, count: Int(view.length)))
    }

    private static func loadToy() throws -> (URL, Model) {
        let dir = try ModelLoaderTests.writeToySynthetic()
        let device = try #require(MTLCreateSystemDefaultDevice())
        return (dir, try Model.load(directoryURL: dir, device: device, expecting: .gemma4Toy()))
    }

    @Test func routedExpertBytesAndLazyOpen() throws {
        let (dir, model) = try Self.loadToy()
        defer { try? FileManager.default.removeItem(at: dir) }

        #expect(model.routedLayers.openLayerCount == 0)
        let view = try model.routedExpertViews(layer: 1, experts: [4])[0]
        #expect(model.routedLayers.openLayerCount == 1)
        _ = try model.routedExpertViews(layer: 1, experts: [0])[0]
        #expect(model.routedLayers.openLayerCount == 1)
        _ = try model.routedExpertViews(layer: 0, experts: [0])[0]
        #expect(model.routedLayers.openLayerCount == 2)

        // Each expert is its own buffer over the mapping, so offset is 0.
        #expect(view.offset == 0)
        #expect(Int(view.length) == model.routedExpertBytes)

        // Tagged bytes written by ModelLoaderTests.writeToySynthetic.
        let b = Self.readBytes(view)
        #expect(b[0] == 1)       // layer 1
        #expect(b[1] == 4)       // expert 4
        #expect(b[2] == 0xC1)
        #expect(b[3] == 0xC2)
    }

    @Test func viewsAreStableAndFetchMakesThemResident() async throws {
        let (dir, model) = try Self.loadToy()
        defer { try? FileManager.default.removeItem(at: dir) }

        let first = try model.routedExpertViews(layer: 1, experts: [2, 4])
        try await model.readRoutedExperts(layer: 1, experts: [2, 4])
        let fetched = try model.routedExpertViews(layer: 1, experts: [2, 4])
        #expect(try model.nonResidentRoutedExperts(layer: 1, experts: [2, 4]).isEmpty)
        for (a, b) in zip(first, fetched) { #expect(a.buffer === b.buffer) }
        #expect(Self.readBytes(fetched[0])[1] == 2)
        #expect(Self.readBytes(fetched[1])[1] == 4)
        try model.prefetchRoutedExperts(layer: 1, experts: [0, 1])
        #expect(throws: (any Error).self) { _ = try model.routedExpertViews(layer: 1, experts: [10_000]) }
    }

    @Test func visionResidencyModesReleaseOrRetainExpertLayers() throws {
        let (dir, model) = try Self.loadToy()
        defer { try? FileManager.default.removeItem(at: dir) }
        _ = try model.routedExpertViews(layer: 1, experts: [4])[0]

        model.prepareExpertResidencyForVision(.keepReady)
        #expect(model.routedLayers.openLayerCount == 1)

        model.prepareExpertResidencyForVision(.onDemand)
        #expect(model.routedLayers.openLayerCount == 0)

        let view = try model.routedExpertViews(layer: 1, experts: [4])[0]
        #expect(model.routedLayers.openLayerCount == 1)
        #expect(Self.readBytes(view)[1] == 4)
    }
}
