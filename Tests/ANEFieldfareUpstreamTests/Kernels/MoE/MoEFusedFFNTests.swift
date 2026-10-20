import Foundation
import Metal
import Testing
@testable import ANEFieldfareHotpath
import ANEFieldfareValidationSupport

@Suite struct MoEFusedFFNTests {
    private static let dimension = 128
    private static let intermediate = 64
    private static let topK = 8

    private struct RoutedBlob {
        let bytes: [UInt8]
        let offsets: MoEExpertOffsets
    }

    @Test func productionRoutedPipelineAndHitSplitMatchReference() throws {
        var rng = SeedTree(0x2D3).key("production-routed-moe")
        func matrix(rows: Int, columns: Int) -> [[Float]] {
            (0..<rows).map { _ in
                (0..<columns).map { _ in rng.uniform(-0.4, 0.4) }
            }
        }

        var gates = [[[Float]]]()
        var ups = [[[Float]]]()
        var downs = [[[Float]]]()
        for _ in 0..<Self.topK {
            gates.append(matrix(rows: Self.intermediate, columns: Self.dimension))
            ups.append(matrix(rows: Self.intermediate, columns: Self.dimension))
            downs.append(matrix(rows: Self.dimension, columns: Self.intermediate))
        }
        let x = (0..<Self.dimension).map { _ in Float(Float16(rng.uniform(-0.5, 0.5))) }
        let residual = (0..<Self.dimension).map { _ in Float(Float16(rng.uniform(-0.5, 0.5))) }
        let routingWeights = (0..<Self.topK).map { Float(Float16(0.04 + Float($0) * 0.015)) }
        func quantized(_ experts: [[[Float]]]) -> [[Quantization.Int4AffineRow]] {
            experts.map { rows in rows.map { Quantization.quantizeInt4Affine($0) } }
        }
        let expected = MoeRef.applyStreamedRouted(
            x: x,
            residual: residual,
            routedGate: quantized(gates),
            routedUp: quantized(ups),
            routedDown: quantized(downs),
            indices: Array(0..<Self.topK),
            routingWeights: routingWeights,
            d: Self.dimension,
            f: Self.intermediate)
        let blobs = (0..<Self.topK).map {
            Self.makeBlob(gate: gates[$0], up: ups[$0], down: downs[$0])
        }

        let context = try MetalContext()
        let kernel = try MoE(context: context)
        func buffer<T: BitwiseCopyable>(_ values: [T]) throws -> any MTLBuffer {
            try #require(unsafe context.device.makeBuffer(bytes: values,
                                                          length: values.count * MemoryLayout<T>.stride,
                                                          options: .storageModeShared))
        }
        func output(_ count: Int) throws -> any MTLBuffer {
            try #require(Fp16Buffer.make(context.device, count: count))
        }
        let routedBuffers = try blobs.map { try buffer($0.bytes) }
        let xBuffer = try #require(Fp16Buffer.make(context.device, values: x))
        let residualBuffer = try #require(Fp16Buffer.make(context.device, values: residual))
        let routingBuffer = try #require(Fp16Buffer.make(context.device, values: routingWeights))
        let fullActs = try output(Self.topK * Self.intermediate)
        let splitActs = try output(Self.topK * Self.intermediate)
        let fullOutput = try output(Self.dimension)
        let splitOutput = try output(Self.dimension)
        let lowSlots = try buffer([UInt32](0...3))
        let highSlots = try buffer([UInt32](4...7))
        // Decode binds the routed experts through tensor views into the reused
        // argument buffer.
        let views = routedBuffers.map {
            TensorView(buffer: $0, offset: 0, length: UInt64($0.length), scaleOffset: 0, scaleLength: 0,
                       biasOffset: 0, biasLength: 0, shape: (0, 0, 0, 0), dtype: 0)
        }
        let argumentBuffer = kernel.makeReusedRoutedArgumentBuffer(routedViews: views, topK: UInt32(Self.topK))

        let fullCommand = try #require(context.queue.makeCommandBuffer())
        kernel.encodeRoutedPersistentPhase1U16Load(
            commandBuffer: fullCommand,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers,
            routedOffsets: blobs[0].offsets,
            x: xBuffer,
            acts: fullActs,
            d: UInt32(Self.dimension),
            f: UInt32(Self.intermediate),
            topK: UInt32(Self.topK))
        kernel.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: fullCommand,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers,
            routedOffsets: blobs[0].offsets,
            acts: fullActs,
            routingWeights: routingBuffer,
            residual: residualBuffer,
            y: fullOutput,
            d: UInt32(Self.dimension),
            f: UInt32(Self.intermediate),
            topK: UInt32(Self.topK))
        fullCommand.commit()
        fullCommand.waitUntilCompleted()
        #expect(fullCommand.error == nil)

        let splitCommand = try #require(context.queue.makeCommandBuffer())
        for (slots, activeSlots) in [([UInt32](0...3), lowSlots),
                                     ([UInt32](4...7), highSlots)] {
            kernel.encodeRoutedPersistentPhase1SubsetU16Load(
                commandBuffer: splitCommand,
                routedArgBuffer: argumentBuffer,
                routedBlobs: routedBuffers,
                routedOffsets: blobs[0].offsets,
                x: xBuffer,
                acts: splitActs,
                activeSlots: activeSlots,
                activeSlotIndices: slots,
                activeCount: UInt32(slots.count),
                d: UInt32(Self.dimension),
                f: UInt32(Self.intermediate),
                topK: UInt32(Self.topK))
        }
        kernel.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: splitCommand,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers,
            routedOffsets: blobs[0].offsets,
            acts: splitActs,
            routingWeights: routingBuffer,
            residual: residualBuffer,
            y: splitOutput,
            d: UInt32(Self.dimension),
            f: UInt32(Self.intermediate),
            topK: UInt32(Self.topK))
        splitCommand.commit()
        splitCommand.waitUntilCompleted()
        #expect(splitCommand.error == nil)

        let full = Fp16Buffer.read(fullOutput, count: Self.dimension)
        let split = Fp16Buffer.read(splitOutput, count: Self.dimension)
        #expect(full == split)
        #expect(RelError.compute(actual: full, reference: expected)
            < Tolerance.fp16ChainedReduction)
    }

    /// `--expert-top-k` below 8: the reduce must ignore the slots past top-k,
    /// even when they still hold a previous layer's experts and activations.
    @Test func fewerRoutedExpertsIgnoreStaleSlots() throws {
        var rng = SeedTree(0x2D4).key("reduced-top-k")
        func matrix(rows: Int, columns: Int) -> [[Float]] {
            (0..<rows).map { _ in (0..<columns).map { _ in rng.uniform(-0.4, 0.4) } }
        }
        let active = 5
        let experts = (0..<Self.topK).map { _ in
            (gate: matrix(rows: Self.intermediate, columns: Self.dimension),
             up: matrix(rows: Self.intermediate, columns: Self.dimension),
             down: matrix(rows: Self.dimension, columns: Self.intermediate))
        }
        let x = (0..<Self.dimension).map { _ in Float(Float16(rng.uniform(-0.5, 0.5))) }
        let residual = (0..<Self.dimension).map { _ in Float(Float16(rng.uniform(-0.5, 0.5))) }
        let routingWeights = (0..<Self.topK).map { Float(Float16(0.04 + Float($0) * 0.015)) }
        func quantized(_ rows: [[Float]]) -> [Quantization.Int4AffineRow] {
            rows.map { Quantization.quantizeInt4Affine($0) }
        }
        let expected = MoeRef.applyStreamedRouted(
            x: x, residual: residual,
            routedGate: experts.prefix(active).map { quantized($0.gate) },
            routedUp: experts.prefix(active).map { quantized($0.up) },
            routedDown: experts.prefix(active).map { quantized($0.down) },
            indices: Array(0..<active),
            routingWeights: Array(routingWeights.prefix(active)),
            d: Self.dimension, f: Self.intermediate)

        let context = try MetalContext()
        let kernel = try MoE(context: context)
        let blobs = experts.map { Self.makeBlob(gate: $0.gate, up: $0.up, down: $0.down) }
        let routedBuffers = try blobs.map {
            try #require(unsafe context.device.makeBuffer(bytes: $0.bytes, length: $0.bytes.count,
                                                          options: .storageModeShared))
        }
        let views = routedBuffers.map {
            TensorView(buffer: $0, offset: 0, length: UInt64($0.length), scaleOffset: 0, scaleLength: 0,
                       biasOffset: 0, biasLength: 0, shape: (0, 0, 0, 0), dtype: 0)
        }
        let xBuffer = try #require(Fp16Buffer.make(context.device, values: x))
        let residualBuffer = try #require(Fp16Buffer.make(context.device, values: residual))
        let routingBuffer = try #require(Fp16Buffer.make(context.device, values: routingWeights))
        let acts = try #require(Fp16Buffer.make(context.device, count: Self.topK * Self.intermediate))
        let output = try #require(Fp16Buffer.make(context.device, count: Self.dimension))
        let d = UInt32(Self.dimension), f = UInt32(Self.intermediate)

        // A full layer first leaves every slot and its activations live.
        let full = kernel.makeReusedRoutedArgumentBuffer(routedViews: views, topK: UInt32(Self.topK))
        let command = try #require(context.queue.makeCommandBuffer())
        try kernel.encodeRoutedPersistentPhase1U16Load(
            commandBuffer: command, routedArgBuffer: full, routedBlobs: routedBuffers,
            routedOffsets: blobs[0].offsets, x: xBuffer, acts: acts, d: d, f: f, topK: UInt32(Self.topK))
        command.commit()
        command.waitUntilCompleted()
        let reduced = kernel.makeReusedRoutedArgumentBuffer(routedViews: Array(views.prefix(active)),
                                                            topK: UInt32(active))
        let activeBuffers = Array(routedBuffers.prefix(active))
        let reducedCommand = try #require(context.queue.makeCommandBuffer())
        try kernel.encodeRoutedPersistentPhase1U16Load(
            commandBuffer: reducedCommand, routedArgBuffer: reduced, routedBlobs: activeBuffers,
            routedOffsets: blobs[0].offsets, x: xBuffer, acts: acts, d: d, f: f, topK: UInt32(active))
        try kernel.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: reducedCommand, routedArgBuffer: reduced, routedBlobs: activeBuffers,
            routedOffsets: blobs[0].offsets, acts: acts, routingWeights: routingBuffer,
            residual: residualBuffer, y: output, d: d, f: f, topK: UInt32(active))
        reducedCommand.commit()
        reducedCommand.waitUntilCompleted()
        #expect(command.error == nil && reducedCommand.error == nil)

        let actual = Fp16Buffer.read(output, count: Self.dimension)
        #expect(RelError.compute(actual: actual, reference: expected) < Tolerance.fp16ChainedReduction)
    }

    private static func makeBlob(gate: [[Float]],
                                 up: [[Float]],
                                 down: [[Float]]) -> RoutedBlob {
        func packed(_ rows: [[Float]])
            -> (weights: [UInt8], scales: [UInt16], biases: [UInt16]) {
            let quantized = rows.map { Quantization.quantizeInt4Affine($0) }
            return (quantized.flatMap(\.packed),
                    quantized.flatMap(\.scales),
                    quantized.flatMap(\.biases))
        }
        var bytes = [UInt8]()
        func append(_ values: [UInt8]) { bytes.append(contentsOf: values) }
        func append(_ values: [UInt16]) {
            for value in values {
                bytes.append(UInt8(truncatingIfNeeded: value))
                bytes.append(UInt8(truncatingIfNeeded: value >> 8))
            }
        }
        let gateValues = packed(gate)
        let upValues = packed(up)
        let downValues = packed(down)
        let gateW = UInt32(bytes.count); append(gateValues.weights)
        let gateS = UInt32(bytes.count); append(gateValues.scales)
        let gateB = UInt32(bytes.count); append(gateValues.biases)
        let upW = UInt32(bytes.count); append(upValues.weights)
        let upS = UInt32(bytes.count); append(upValues.scales)
        let upB = UInt32(bytes.count); append(upValues.biases)
        let downW = UInt32(bytes.count); append(downValues.weights)
        let downS = UInt32(bytes.count); append(downValues.scales)
        let downB = UInt32(bytes.count); append(downValues.biases)
        return RoutedBlob(
            bytes: bytes,
            offsets: MoEExpertOffsets(
                gateWOff: gateW, gateSOff: gateS, gateBOff: gateB,
                upWOff: upW, upSOff: upS, upBOff: upB,
                downWOff: downW, downSOff: downS, downBOff: downB))
    }
}
