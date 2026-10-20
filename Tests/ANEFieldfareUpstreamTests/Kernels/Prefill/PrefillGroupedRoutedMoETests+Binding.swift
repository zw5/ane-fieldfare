import Metal
import Testing

@testable import ANEFieldfareHotpath

extension PrefillGroupedRoutedMoETests {
  @Test func streamedTileBindingRejectsDuplicateExperts() throws {
    let ctx = try MetalContext()
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

    let firstTile = PrefillStreamedTileBinding.expertIDs(forTile: 0, routes: routes)
    let secondTile = PrefillStreamedTileBinding.expertIDs(forTile: 1, routes: routes)

    #expect(firstTile == Array(0..<16))
    #expect(secondTile == Array(16..<20))
  }
}
