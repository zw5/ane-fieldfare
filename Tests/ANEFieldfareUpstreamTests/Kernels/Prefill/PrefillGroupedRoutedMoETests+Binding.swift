import Metal
import Testing

@testable import ANEFieldfareHotpath

extension PrefillGroupedRoutedMoETests {
  @Test func streamedTileBindingMapsGlobalExpertsToLocalSlots() throws {
    let ctx = try MetalContext()
    let expertIDs = [3, 9, 17]
    let views = try Self.fakeTensorViews(device: ctx.device, count: expertIDs.count)
    let binding = try PrefillStreamedTileBinding(expertIDs: expertIDs, views: views)

    #expect(binding.localSlot(for: 3) == 0)
    #expect(binding.localSlot(for: 9) == 1)
    #expect(binding.localSlot(for: 3) == 0)
    #expect(binding.localSlot(for: 17) == 2)
    #expect(binding.localSlot(for: 5) == nil)

    let pairs = [
      Self.pair(token: 0, expert: 3, rank: 0),
      Self.pair(token: 0, expert: 9, rank: 1),
      Self.pair(token: 1, expert: 3, rank: 0),
      Self.pair(token: 1, expert: 17, rank: 1),
    ]
    try binding.validateCoversPairs(pairs, pairStart: 0, pairCount: pairs.count)
  }

  @Test func streamedTileBindingRejectsUnboundRouteExpert() throws {
    let ctx = try MetalContext()
    let binding = try PrefillStreamedTileBinding(
      expertIDs: [3, 9],
      views: Self.fakeTensorViews(device: ctx.device, count: 2))
    let pairs = [
      Self.pair(token: 0, expert: 3, rank: 0),
      Self.pair(token: 0, expert: 17, rank: 1),
    ]

    #expect {
      try binding.validateCoversPairs(pairs, pairStart: 0, pairCount: pairs.count)
    } throws: { error in
      guard case PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(let reason) = error else {
        return false
      }
      return reason.contains("17")
    }

    #expect {
      _ = try PrefillStreamedTileBinding(
        expertIDs: [3, 3],
        views: Self.fakeTensorViews(device: ctx.device, count: 2))
    } throws: { error in
      guard case PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(let reason) = error else {
        return false
      }
      return reason.contains("duplicate")
    }
  }

  @Test func streamedTilePlanningNeverExceedsDefaultSlotCount() throws {
    let pairs = (0..<24).map { index in
      Self.pair(
        token: UInt32(index / 8),
        expert: UInt32(min(index, 19)),
        rank: UInt32(index % 8))
    }
    let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
      pairs,
      queryCount: 3,
      topK: 8,
      numExperts: 128,
      tileExpertCount: 16)

    let firstTile = try PrefillStreamedTileBinding.expertIDs(forTile: 0, routes: routes)
    let secondTile = try PrefillStreamedTileBinding.expertIDs(forTile: 1, routes: routes)

    #expect(firstTile == Array(0..<16))
    #expect(secondTile == Array(16..<20))
    #expect(firstTile.count <= 16)
    #expect(secondTile.count <= 16)
    #expect(firstTile != Array(0..<20))
    #expect(secondTile != Array(0..<20))
  }

  @Test func streamedTileBindingFromModelViewsCarriesEachExpertsBytes() async throws {
    let dir = try ModelLoaderTests.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    let device = try #require(MTLCreateSystemDefaultDevice())
    let model = try Model.load(
      directoryURL: dir,
      device: device,
      expecting: .gemma4Toy())
    let routes = try Self.tileFetchRoutes()
    let expertIDs = try PrefillStreamedTileBinding.expertIDs(forTile: 0, routes: routes)
    #expect(expertIDs == [1, 3, 5])

    let views = try await model.fetchRoutedExperts(layer: 1, experts: expertIDs)
    #expect(try model.nonResidentRoutedExperts(layer: 1, experts: expertIDs).isEmpty)
    let binding = try PrefillStreamedTileBinding(expertIDs: expertIDs, views: views)
    try binding.validateCoversPairs(
      routes.sortedPairs,
      pairStart: 0,
      pairCount: routes.sortedPairs.count)
    // Every expert has one fixed buffer: binding twice yields the same views.
    let again = try PrefillStreamedTileBinding(
      expertIDs: expertIDs, views: try model.routedExpertViews(layer: 1, experts: expertIDs))
    #expect(again == binding)
    for (index, expert) in expertIDs.enumerated() {
      let view = binding.views[index]
      #expect(view.offset == 0)
      #expect(Self.byte(view, at: 0) == 1)
      #expect(Self.byte(view, at: 1) == UInt8(expert))
    }
  }

}
