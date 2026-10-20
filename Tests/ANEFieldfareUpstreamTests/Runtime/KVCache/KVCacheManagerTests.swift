import Testing
import Foundation
import Metal
@testable import ANEFieldfareHotpath

/// Tests `KVCacheManager` FP16 shape, separate K/V storage, ring, and
/// cursor semantics against the Gemma 4 config.
@Suite struct KVCacheManagerTests {

    private let config = ArchConfig.gemma4_26B_A4B

    private func makeManager(maxContext: Int,
                             maxPrefillChunkTokens: Int = 128) throws -> (MetalContext, KVCacheManager) {
        let ctx = try MetalContext()
        let kv = try KVCacheManager(device: ctx.device,
                                    config: config,
                                    maxContext: maxContext,
                                    maxPrefillChunkTokens: maxPrefillChunkTokens)
        return (ctx, kv)
    }

    @Test func strideAndBufferSizes_matchConfig() throws {
        let (_, kv) = try makeManager(maxContext: 128)

        // SWA: numKVHeads(8) * headDim(256) * 2 = 4096 B/token.
        // Full: numFullKVHeads(2) * fullHeadDim(512) * 2 = 2048 B/token.
        #expect(kv.kRange(layer: 0, start: 0, count: 1).stride == 8 * 256 * 2)
        #expect(kv.kRange(layer: 5, start: 0, count: 1).stride == 2 * 512 * 2)
        #expect(kv.keyBuffer(layer: 0).length == 128 * 4096)
        #expect(kv.keyBuffer(layer: 5).length == 128 * 2048)
    }

    /// Full layers share the raw k_proj output, then diverge: K runs k_norm +
    /// RoPE while V runs no-scale v_norm without RoPE. They therefore require
    /// separate cache slots, as SWA layers do.
    @Test func kAndVBuffersNeverAlias() throws {
        let (_, kv) = try makeManager(maxContext: 16)
        #expect(kv.keyBuffer(layer: 0)
                !== kv.valueBuffer(layer: 0))
        let k = kv.keyBuffer(layer: 5)
        let v = kv.valueBuffer(layer: 5)
        #expect(k !== v, "full-layer K and V must NOT alias")
        let ks = kv.kSlot(layer: 5, position: 3)
        let vs = kv.vSlot(layer: 5, position: 3)
        #expect(ks.buffer !== vs.buffer, "full-layer K/V slots must NOT alias")
        // Offsets are still per-position-strided in both buffers.
        #expect(ks.offset == vs.offset)
    }

    @Test func fp16Ring_capsSWALayersAndLeavesFullLayersLinear() throws {
        let (_, kv) = try makeManager(maxContext: 4096)

        #expect(kv.capacity(layer: 0) == 1152)
        #expect(kv.ringCapacity(layer: 0) == 1152)
        #expect(kv.keyBuffer(layer: 0).length == 1152 * 4096)
        #expect(kv.capacity(layer: 5) == 4096)
        #expect(kv.ringCapacity(layer: 5) == 0)
        #expect(kv.keyBuffer(layer: 5).length == 4096 * 2048)
    }

    @Test func fp16Ring_slotOffsetsWrapOnlyForSWALayers() throws {
        // SWA ring capacity: sliding window 1024 + one 1-token chunk.
        let (_, kv) = try makeManager(maxContext: 2048, maxPrefillChunkTokens: 1)
        #expect(kv.ringCapacity(layer: 0) == 1025)

        #expect(kv.kSlot(layer: 0, position: 0).offset == 0)
        #expect(kv.kSlot(layer: 0, position: 1024).offset == 1024 * 4096)
        #expect(kv.kSlot(layer: 0, position: 1025).offset == 0)
        #expect(kv.vSlot(layer: 0, position: 1028).offset == 3 * 4096)
        // A range starting past the ring wraps as a whole.
        #expect(kv.vRange(layer: 0, start: 1025, count: 3).offset == 0)

        #expect(kv.kSlot(layer: 5, position: 1028).offset == 1028 * 2048)
        #expect(kv.vSlot(layer: 5, position: 1028).offset == 1028 * 2048)
    }

    @Test func cursorAdvancesAndResets() throws {
        let (_, kv) = try makeManager(maxContext: 128)
        kv.advance(by: 31)
        kv.advance(by: 0)
        kv.advance()
        #expect(kv.position == 32)
        kv.reset()
        #expect(kv.position == 0)
        // Cursor reusable after reset.
        kv.advance()
        #expect(kv.position == 1)
    }
}
