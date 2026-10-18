import Foundation
import Metal
import XCTest
@testable import ANEFieldfareHotpath

final class ExpertWindowMappingTests: XCTestCase {
    func testMixedHitMissMoEKernelsMatchPread() throws {
        let context = try MetalContext()
        let device = context.device
        let moe = try MoE(context: context)
        let d = 2816, f = 704, topK = 8
        let weights = d * f / 2, params = d * f / 64 * 2
        let matrixBytes = weights + params * 2, stride = matrixBytes * 3
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("ane-expert-kernel-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: path) }
        var bytes = Data(count: stride * 12)
        bytes.withUnsafeMutableBytes { raw in
            let p = raw.bindMemory(to: UInt8.self)
            for expert in 0..<12 {
                for matrix in 0..<3 {
                    let base = expert * stride + matrix * matrixBytes
                    for i in 0..<weights {
                        p[base + i] = UInt8(truncatingIfNeeded: i * 31 + expert * 17 + matrix)
                    }
                    for i in Swift.stride(from: 0, to: params, by: 2) {
                        p[base + weights + i] = 0x80
                        p[base + weights + i + 1] = 0x3b // BF16 scale 1/256
                        p[base + weights + params + i] = 0x00
                        p[base + weights + params + i + 1] = 0xbc // bias -1/128
                    }
                }
            }
        }
        try bytes.write(to: path)
        let layout = StreamLayout(path: path.path, streamOffset: 0,
                                  streamSize: UInt64(bytes.count), expertsPerLayer: 12,
                                  expertStride: UInt64(stride))
        let offsets = MoEExpertOffsets(
            gateWOff: 0, gateSOff: UInt32(weights), gateBOff: UInt32(weights + params),
            upWOff: UInt32(matrixBytes), upSOff: UInt32(matrixBytes + weights),
            upBOff: UInt32(matrixBytes + weights + params),
            downWOff: UInt32(matrixBytes * 2), downSOff: UInt32(matrixBytes * 2 + weights),
            downBOff: UInt32(matrixBytes * 2 + weights + params))
        func buffer(_ values: [Float16]) throws -> MTLBuffer {
            try values.withUnsafeBytes {
                try XCTUnwrap(device.makeBuffer(bytes: $0.baseAddress!, length: $0.count,
                                                options: .storageModeShared))
            }
        }
        let x = try buffer((0..<d).map { Float16(Float($0 % 7 - 3) * 0.01) })
        let routing = try buffer((0..<topK).map { Float16(Float($0 + 1) / 36) })
        let residual = try buffer(Array(repeating: 0, count: d))
        func run(windowMapped: Bool) throws -> Data {
            let streamer = try PreadExpertStreamer(layout: layout, device: device,
                                                    slotCount: 8, windowMapped: windowMapped,
                                                    fileDescriptor: nil)
            _ = try streamer.loadExpertsCached(experts: Array(0..<8))
            let plan = streamer.planExpertsCached(experts: [0, 1, 2, 3, 8, 9, 10, 11])
            XCTAssertEqual(plan.hits, 4)
            XCTAssertEqual(plan.misses, [4, 5, 6, 7])
            let acts = try buffer(Array(repeating: .nan, count: topK * f))
            let y = try buffer(Array(repeating: .nan, count: d))
            func views(_ entries: [(buffer: MTLBuffer, offset: UInt64, size: UInt64)]) -> [TensorView] {
                entries.map {
                    TensorView(buffer: $0.buffer, offset: $0.offset, length: $0.size,
                               scaleOffset: 0, scaleLength: 0, biasOffset: 0, biasLength: 0,
                               shape: (0, 0, 0, 0), dtype: 0)
                }
            }
            func subset(_ indices: [UInt32], _ v: [TensorView], _ cb: MTLCommandBuffer) throws {
                let active = try indices.withUnsafeBytes {
                    try XCTUnwrap(device.makeBuffer(bytes: $0.baseAddress!, length: $0.count,
                                                    options: .storageModeShared))
                }
                let args = try XCTUnwrap(moe.makeRoutedArgumentBuffer(routedViews: v, topK: 8))
                moe.encodeRoutedPersistentPhase1SubsetU16Load(
                    commandBuffer: cb, routedArgBuffer: args, routedBlobs: v.map(\.buffer),
                    routedOffsets: offsets, x: x, acts: acts, activeSlots: active,
                    activeSlotIndices: indices, activeCount: UInt32(indices.count),
                    d: UInt32(d), f: UInt32(f), topK: 8)
            }
            if windowMapped {
                let hitCB = try XCTUnwrap(context.queue.makeCommandBuffer())
                try subset([0, 1, 2, 3], views(streamer.expertCachePlanBuffers(plan)), hitCB)
                hitCB.commit()
            }
            let fetched = views(try streamer.executeExpertCachePlan(plan))
            let cb = try XCTUnwrap(context.queue.makeCommandBuffer())
            let args = moe.makeReusedRoutedArgumentBuffer(routedViews: fetched, topK: 8)
            if windowMapped {
                try subset([4, 5, 6, 7], fetched, cb)
            } else {
                moe.encodeRoutedPersistentPhase1U16Load(
                    commandBuffer: cb, routedArgBuffer: args, routedBlobs: fetched.map(\.buffer),
                    routedOffsets: offsets, x: x, acts: acts, d: UInt32(d), f: UInt32(f), topK: 8)
            }
            moe.encodeRoutedPersistentPhase2Reduce(
                commandBuffer: cb, routedArgBuffer: args, routedBlobs: fetched.map(\.buffer),
                routedOffsets: offsets, acts: acts, routingWeights: routing,
                residual: residual, y: y, d: UInt32(d), f: UInt32(f), topK: 8)
            cb.commit()
            cb.waitUntilCompleted()
            XCTAssertEqual(cb.status, .completed, "\(String(describing: cb.error))")
            let values = y.contents().bindMemory(to: Float16.self, capacity: d)
            XCTAssertTrue((0..<d).allSatisfy { values[$0].isFinite })
            XCTAssertTrue((0..<d).contains { values[$0] != 0 })
            return Data(bytes: y.contents(), count: d * 2)
        }
        let reference = try run(windowMapped: false)
        XCTAssertEqual(try run(windowMapped: true), reference)
    }

    func testMappedWindowsMatchPreadAcrossEvictionAndGPULifetimes() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("ane-expert-window-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: path) }
        // Neither the stream nor every expert starts on a VM page boundary.
        let prefix = 256
        let stride = 24_576
        let expertCount = 24
        let bytes = Data((0..<(prefix + stride * expertCount)).map {
            UInt8(truncatingIfNeeded: ($0 &* 131) ^ ($0 >> 9))
        })
        try bytes.write(to: path)
        let layout = StreamLayout(path: path.path, streamOffset: UInt64(prefix),
                                  streamSize: UInt64(stride * expertCount),
                                  expertsPerLayer: expertCount,
                                  expertStride: UInt64(stride))
        let mapped = try PreadExpertStreamer(layout: layout, device: device,
                                             slotCount: 16, windowMapped: true,
                                             fileDescriptor: nil)
        let pread = try PreadExpertStreamer(layout: layout, device: device,
                                            slotCount: 16, fileDescriptor: nil)
        let command = try XCTUnwrap(queue.makeCommandBuffer())
        let blit = try XCTUnwrap(command.makeBlitCommandEncoder())
        let output = try XCTUnwrap(device.makeBuffer(length: stride * 8,
                                                    options: .storageModeShared))
        do {
            let held = try mapped.loadExpertsCached(experts: Array(0..<8))
            for (index, entry) in held.enumerated() {
                blit.copy(from: entry.buffer, sourceOffset: Int(entry.offset),
                          to: output, destinationOffset: index * stride, size: stride)
            }
        }
        blit.endEncoding()
        // Evict and release slot ownership before the queued GPU copy executes.
        // The old command's Metal resource retains must keep its windows alive.
        for experts in [Array(8..<16), Array(16..<24), Array(0..<8), Array(8..<16)] {
            let actualPlan = mapped.planExpertsCached(experts: experts)
            let referencePlan = pread.planExpertsCached(experts: experts)
            let actual = try mapped.executeExpertCachePlan(actualPlan)
            let reference = try pread.executeExpertCachePlan(referencePlan)
            for (index, entry) in actual.enumerated() {
                let actualData = Data(bytes: entry.buffer.contents().advanced(by: Int(entry.offset)),
                                      count: stride)
                let referenceData = Data(bytes: reference[index].buffer.contents(), count: stride)
                let begin = prefix + experts[index] * stride
                XCTAssertEqual(actualData, bytes.subdata(in: begin..<(begin + stride)))
                XCTAssertEqual(actualData, referenceData)
            }
        }
        mapped.setActiveSlotCount(8)
        mapped.setActiveSlotCount(16)
        command.commit()
        command.waitUntilCompleted()
        XCTAssertEqual(command.status, .completed, "\(String(describing: command.error))")
        XCTAssertEqual(Data(bytes: output.contents(), count: stride * 8),
                       bytes.subdata(in: prefix..<(prefix + stride * 8)))
        XCTAssertThrowsError(try mapped.loadExpert(layer: 0, expert: expertCount, slot: 0))
        XCTAssertThrowsError(try mapped.loadExpert(layer: 0, expert: 0, slot: 16))
    }
}
