import Metal
import Testing

@testable import ANEFieldfareHotpath

extension PrefillGroupedRoutedMoETests {
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
  }
}
