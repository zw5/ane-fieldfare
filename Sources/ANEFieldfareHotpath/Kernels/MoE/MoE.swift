import Foundation
import Metal

@frozen
public struct MoEExpertOffsets {
    public var gateWOff: UInt32
    public var gateSOff: UInt32
    public var gateBOff: UInt32
    public var upWOff: UInt32
    public var upSOff: UInt32
    public var upBOff: UInt32
    public var downWOff: UInt32
    public var downSOff: UInt32
    public var downBOff: UInt32

    public init(gateWOff: UInt32, gateSOff: UInt32, gateBOff: UInt32,
                upWOff: UInt32, upSOff: UInt32, upBOff: UInt32,
                downWOff: UInt32, downSOff: UInt32, downBOff: UInt32) {
        self.gateWOff = gateWOff
        self.gateSOff = gateSOff
        self.gateBOff = gateBOff
        self.upWOff = upWOff
        self.upSOff = upSOff
        self.upBOff = upBOff
        self.downWOff = downWOff
        self.downSOff = downSOff
        self.downBOff = downBOff
    }
}

final class MoE {
    static let maxStreamedExperts = 8

    private static let realDecodeD: UInt32 = 2816
    private static let realDecodeF: UInt32 = 704
    private static let realDecodeTopK: UInt32 = 8
    private static let realDecodeNumExperts: UInt32 = 128
    private static let realDecodeMoEConstants: [MetalFunctionConstant] = [
        MetalFunctionConstant(index: 0, value: .uint32(realDecodeD)),
        MetalFunctionConstant(index: 1, value: .uint32(realDecodeF)),
        MetalFunctionConstant(index: 2, value: .uint32(realDecodeTopK)),
    ]
    private static let realDecodeRouterConstants: [MetalFunctionConstant] = [
        MetalFunctionConstant(index: 40, value: .uint32(realDecodeNumExperts)),
        MetalFunctionConstant(index: 41, value: .uint32(realDecodeD)),
    ]

    private let routerGemvPSO: any MTLComputePipelineState
    private let routerGemvSpecializedPSO: any MTLComputePipelineState
    private let routerSelectK8PSO: any MTLComputePipelineState
    private let routerSelectK8SIMDPSO: any MTLComputePipelineState
    private let routerGemvPairSpecializedPSO: any MTLComputePipelineState
    private let routerGemvPairRowsSpecializedPSO: any MTLComputePipelineState
    private let routerSelectK8SIMDRowsPSO: any MTLComputePipelineState
    /// `[maxRouterRows][256]` FP32 logits for `encodeRouterRowsGemma4`: the
    /// router's, then the next layer's guess.
    private let routerRowsLogits: any MTLBuffer
    /// The last row router's next-layer guess scores, `[rows][numExperts]` FP32.
    let guessRowsLogits: any MTLBuffer
    private static let maxRouterRows = 16
    private let routerLogits: any MTLBuffer
    private let phase1U16PSO: any MTLComputePipelineState
    private let phase1U16SpecializedPSO: any MTLComputePipelineState
    private let phase1SubsetU16PSO: any MTLComputePipelineState
    private let phase1SubsetU16SpecializedPSO: any MTLComputePipelineState
    private let phase2ReduceK8PSO: any MTLComputePipelineState
    private let phase2ReduceK8SpecializedPSO: any MTLComputePipelineState
    private let gatherRoutedSlotsPSO: any MTLComputePipelineState
    private let routedArgEncoder: any MTLArgumentEncoder
    private let reusableRoutedArgBuffer: any MTLBuffer

    init(context: MetalContext) throws {
        let routerName = "router_gemv_gemma4_r4"
        self.routerGemvPSO = try context.pipeline(
            routerName,
            constants: [],
            maxTotalThreadsPerThreadgroup: 512)
        self.routerGemvSpecializedPSO = try context.pipeline(
            routerName,
            constants: Self.realDecodeRouterConstants,
            maxTotalThreadsPerThreadgroup: 512)
        self.routerSelectK8PSO = try context.pipeline("router_topk_select_k8")
        self.routerSelectK8SIMDPSO = try context.pipeline(
            "router_topk_select_k8_simd",
            constants: Self.realDecodeRouterConstants)
        self.routerGemvPairSpecializedPSO = try context.pipeline(
            "router_gemv_gemma4_pair_r4",
            constants: Self.realDecodeRouterConstants,
            maxTotalThreadsPerThreadgroup: 512)
        self.routerGemvPairRowsSpecializedPSO = try context.pipeline(
            "router_gemv_gemma4_pair_rows_r4",
            constants: Self.realDecodeRouterConstants,
            maxTotalThreadsPerThreadgroup: 512)
        self.routerSelectK8SIMDRowsPSO = try context.pipeline(
            "router_topk_select_k8_simd_rows",
            constants: Self.realDecodeRouterConstants)
        guard let rowsLogits = context.device.makeBuffer(
                length: Self.maxRouterRows * 256 * MemoryLayout<Float>.stride, options: .storageModePrivate),
              let guessLogits = context.device.makeBuffer(
                length: Self.maxRouterRows * 256 * MemoryLayout<Float>.stride, options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        self.routerRowsLogits = rowsLogits
        self.guessRowsLogits = guessLogits
        self.phase1U16PSO = try context.pipeline("moe_phase1_gate_up_act_u16load")
        self.phase1U16SpecializedPSO = try context.pipeline(
            "moe_phase1_gate_up_act_u16load",
            constants: Self.realDecodeMoEConstants)
        self.phase1SubsetU16PSO = try context.pipeline("moe_phase1_gate_up_act_subset_u16load")
        self.phase1SubsetU16SpecializedPSO = try context.pipeline(
            "moe_phase1_gate_up_act_subset_u16load",
            constants: Self.realDecodeMoEConstants)
        self.phase2ReduceK8PSO = try context.pipeline("moe_phase2_down_reduce_k8")
        self.gatherRoutedSlotsPSO = try context.pipeline("moe_gather_routed_slots")
        self.phase2ReduceK8SpecializedPSO = try context.pipeline(
            "moe_phase2_down_reduce_k8",
            constants: Self.realDecodeMoEConstants)

        guard let logits = context.device.makeBuffer(
            length: 256 * MemoryLayout<Float>.stride,
            options: .storageModeShared),
              let phase1Function = context.library.makeFunction(
                name: "moe_phase1_gate_up_act_u16load") else {
            throw MetalError.noDevice
        }
        self.routerLogits = logits
        self.routedArgEncoder = phase1Function.makeArgumentEncoder(bufferIndex: 0)
        guard let reusable = context.device.makeBuffer(
            length: routedArgEncoder.encodedLength,
            options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        self.reusableRoutedArgBuffer = reusable
    }

    /// The router GEMV alone: `numExperts` Float32 scores into `outLogits`,
    /// with no top-k.
    private func encodeRouterLogitsGemma4(commandBuffer: any MTLCommandBuffer,
                                  weights: any MTLBuffer, weightsOffset: Int = 0,
                                  scales: any MTLBuffer, scalesOffset: Int = 0,
                                  biases: any MTLBuffer, biasesOffset: Int = 0,
                                  hidden: any MTLBuffer,
                                  effectiveScale: any MTLBuffer, effectiveScaleOffset: Int = 0,
                                  outLogits: any MTLBuffer,
                                  numExperts: UInt32,
                                  d: UInt32) throws {
        precondition(d.isMultiple(of: UInt32(Quantization.groupSize)))
        precondition(numExperts <= 256)
        precondition(outLogits.length >= Int(numExperts) * MemoryLayout<Float>.stride)

        let expertCount = numExperts
        let dimension = d
        let useSpecialized = numExperts == Self.realDecodeNumExperts
            && d == Self.realDecodeD
        let encoder = try commandBuffer.makeTracedComputeEncoder()
        encoder.setComputePipelineState(
            useSpecialized ? routerGemvSpecializedPSO : routerGemvPSO)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(hidden, offset: 0, index: 3)
        encoder.setBuffer(effectiveScale, offset: effectiveScaleOffset, index: 4)
        encoder.setBuffer(outLogits, offset: 0, index: 5)
        encoder.setValue(expertCount, index: 6)
        encoder.setValue(dimension, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(numExperts) + 3) / 4, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// A second router-shaped GEMV over the same input, run with the router.
    struct RouterLogitsTarget {
        let weights: any MTLBuffer, weightsOffset: Int
        let scales: any MTLBuffer, scalesOffset: Int
        let biases: any MTLBuffer, biasesOffset: Int
        let effectiveScale: any MTLBuffer
        let outLogits: any MTLBuffer
    }

    func encodeRouterGemma4(commandBuffer: any MTLCommandBuffer,
                                   weights: any MTLBuffer, weightsOffset: Int = 0,
                                   scales: any MTLBuffer, scalesOffset: Int = 0,
                                   biases: any MTLBuffer, biasesOffset: Int = 0,
                                   hidden: any MTLBuffer,
                                   effectiveScale: any MTLBuffer, effectiveScaleOffset: Int = 0,
                                   perExpertScale: any MTLBuffer, perExpertScaleOffset: Int = 0,
                                   outIndices: any MTLBuffer,
                                   outWeights: any MTLBuffer,
                                   numExperts: UInt32,
                                   d: UInt32,
                                   topK: UInt32,
                                   next: RouterLogitsTarget? = nil) throws {
        precondition(topK == UInt32(Self.maxStreamedExperts))
        let expertCount = numExperts
        let useSpecialized = numExperts == Self.realDecodeNumExperts
            && d == Self.realDecodeD
        if useSpecialized, let next {
            let encoder = try commandBuffer.makeTracedComputeEncoder()
            let dimension = d
            encoder.setComputePipelineState(routerGemvPairSpecializedPSO)
            encoder.setBuffer(weights, offset: weightsOffset, index: 0)
            encoder.setBuffer(scales, offset: scalesOffset, index: 1)
            encoder.setBuffer(biases, offset: biasesOffset, index: 2)
            encoder.setBuffer(hidden, offset: 0, index: 3)
            encoder.setBuffer(effectiveScale, offset: effectiveScaleOffset, index: 4)
            encoder.setBuffer(routerLogits, offset: 0, index: 5)
            encoder.setValue(expertCount, index: 6)
            encoder.setValue(dimension, index: 7)
            encoder.setBuffer(next.weights, offset: next.weightsOffset, index: 8)
            encoder.setBuffer(next.scales, offset: next.scalesOffset, index: 9)
            encoder.setBuffer(next.biases, offset: next.biasesOffset, index: 10)
            encoder.setBuffer(next.effectiveScale, offset: 0, index: 11)
            encoder.setBuffer(next.outLogits, offset: 0, index: 12)
            encoder.dispatchThreadgroups(
                MTLSize(width: 2 * ((Int(numExperts) + 3) / 4), height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            encoder.endEncoding()
        } else {
            if let next {
                try encodeRouterLogitsGemma4(commandBuffer: commandBuffer,
                                             weights: next.weights, weightsOffset: next.weightsOffset,
                                             scales: next.scales, scalesOffset: next.scalesOffset,
                                             biases: next.biases, biasesOffset: next.biasesOffset,
                                             hidden: hidden, effectiveScale: next.effectiveScale,
                                             outLogits: next.outLogits, numExperts: numExperts, d: d)
            }
            try encodeRouterLogitsGemma4(commandBuffer: commandBuffer,
                                 weights: weights, weightsOffset: weightsOffset,
                                 scales: scales, scalesOffset: scalesOffset,
                                 biases: biases, biasesOffset: biasesOffset,
                                 hidden: hidden,
                                 effectiveScale: effectiveScale,
                                 effectiveScaleOffset: effectiveScaleOffset,
                                 outLogits: routerLogits,
                                 numExperts: numExperts,
                                 d: d)
        }

        let encoder = try commandBuffer.makeTracedComputeEncoder()
        encoder.setComputePipelineState(
            useSpecialized ? routerSelectK8SIMDPSO : routerSelectK8PSO)
        encoder.setBuffer(routerLogits, offset: 0, index: 0)
        encoder.setBuffer(perExpertScale, offset: perExpertScaleOffset, index: 1)
        encoder.setBuffer(outIndices, offset: 0, index: 2)
        encoder.setBuffer(outWeights, offset: 0, index: 3)
        encoder.setValue(expertCount, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Whether `encodeRouterRowsGemma4` serves this shape.
    static func routesRows(numExperts: UInt32, d: UInt32, rows: Int) -> Bool {
        numExperts == realDecodeNumExperts && d == realDecodeD && (1...maxRouterRows).contains(rows)
    }

    /// The decode router over `rows` hidden rows of a block: per row the same
    /// GEMV and top-8 select as `encodeRouterGemma4`, into `[rows][8]`
    /// indices and weights. `next` (its `outLogits` unused) also scores each
    /// row with the next layer's guess and selects its top 8 into
    /// `nextIndices`/`nextWeights`.
    func encodeRouterRowsGemma4(commandBuffer: any MTLCommandBuffer,
                                weights: any MTLBuffer, weightsOffset: Int,
                                scales: any MTLBuffer, scalesOffset: Int,
                                biases: any MTLBuffer, biasesOffset: Int,
                                hidden: any MTLBuffer, hiddenStrideElements: UInt32, rows: Int,
                                effectiveScale: any MTLBuffer,
                                perExpertScale: any MTLBuffer, perExpertScaleOffset: Int,
                                outIndices: any MTLBuffer, outWeights: any MTLBuffer,
                                numExperts: UInt32, d: UInt32,
                                next: RouterLogitsTarget?,
                                nextPerExpertScale: any MTLBuffer?, nextPerExpertScaleOffset: Int = 0,
                                nextIndices: any MTLBuffer?, nextWeights: any MTLBuffer?) throws {
        precondition(Self.routesRows(numExperts: numExperts, d: d, rows: rows))
        let expertCount = numExperts
        let dimension = d
        let stride = hiddenStrideElements
        let groups = (Int(numExperts) + 3) / 4
        let encoder = try commandBuffer.makeTracedComputeEncoder()
        encoder.setComputePipelineState(routerGemvPairRowsSpecializedPSO)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(hidden, offset: 0, index: 3)
        encoder.setBuffer(effectiveScale, offset: 0, index: 4)
        encoder.setBuffer(routerRowsLogits, offset: 0, index: 5)
        encoder.setValue(expertCount, index: 6)
        encoder.setValue(dimension, index: 7)
        encoder.setBuffer(next?.weights ?? weights, offset: next?.weightsOffset ?? weightsOffset, index: 8)
        encoder.setBuffer(next?.scales ?? scales, offset: next?.scalesOffset ?? scalesOffset, index: 9)
        encoder.setBuffer(next?.biases ?? biases, offset: next?.biasesOffset ?? biasesOffset, index: 10)
        encoder.setBuffer(next?.effectiveScale ?? effectiveScale, offset: 0, index: 11)
        encoder.setBuffer(guessRowsLogits, offset: 0, index: 12)
        encoder.setValue(stride, index: 13)
        encoder.dispatchThreadgroups(
            MTLSize(width: next == nil ? groups : 2 * groups, height: rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()
        let selects: [(any MTLBuffer, any MTLBuffer?, Int, any MTLBuffer?, any MTLBuffer?)] = [
            (routerRowsLogits, perExpertScale, perExpertScaleOffset, outIndices, outWeights),
            (guessRowsLogits, nextPerExpertScale, nextPerExpertScaleOffset, nextIndices, nextWeights),
        ]
        for (logits, scale, scaleOffset, indices, weightsOut) in selects.prefix(next == nil ? 1 : 2) {
            guard let scale, let indices, let weightsOut else { continue }
            let encoder = try commandBuffer.makeTracedComputeEncoder()
            encoder.setComputePipelineState(routerSelectK8SIMDRowsPSO)
            encoder.setBuffer(logits, offset: 0, index: 0)
            encoder.setBuffer(scale, offset: scaleOffset, index: 1)
            encoder.setBuffer(indices, offset: 0, index: 2)
            encoder.setBuffer(weightsOut, offset: 0, index: 3)
            encoder.setValue(expertCount, index: 4)
            encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            encoder.endEncoding()
        }
    }

    func makeReusedRoutedArgumentBuffer(routedViews: [TensorView],
                                        topK: UInt32) -> any MTLBuffer {
        validate(routedBlobs: routedViews.map(\.buffer), topK: topK)
        routedArgEncoder.setArgumentBuffer(reusableRoutedArgBuffer, offset: 0)
        for (index, view) in routedViews.enumerated() {
            routedArgEncoder.setBuffer(view.buffer, offset: Int(view.offset), index: index)
        }
        return reusableRoutedArgBuffer
    }

    func encodeRoutedPersistentPhase1U16Load(
        commandBuffer: any MTLCommandBuffer,
        routedArgBuffer: any MTLBuffer,
        routedBlobs: [any MTLBuffer],
        routedOffsets: MoEExpertOffsets,
        x: any MTLBuffer,
        acts: any MTLBuffer,
        d: UInt32,
        f: UInt32,
        topK: UInt32
    ) throws {
        validate(routedBlobs: routedBlobs, topK: topK)
        let dimension = d
        let intermediate = f
        let expertCount = topK
        let encoder = try commandBuffer.makeTracedComputeEncoder()
        encoder.setComputePipelineState(
            useRealDecodeConstants(d: d, f: f, topK: topK)
                ? phase1U16SpecializedPSO
                : phase1U16PSO)
        encoder.setBuffer(routedArgBuffer, offset: 0, index: 0)
        for buffer in routedBlobs { encoder.useResource(buffer, usage: .read) }
        let offsets = routedOffsets
        encoder.setValue(offsets, index: 1)
        encoder.setBuffer(x, offset: 0, index: 2)
        encoder.setBuffer(acts, offset: 0, index: 3)
        encoder.setValue(dimension, index: 4)
        encoder.setValue(intermediate, index: 5)
        encoder.setValue(expertCount, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(topK * f) + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeRoutedPersistentPhase1SubsetU16Load(
        commandBuffer: any MTLCommandBuffer,
        routedArgBuffer: any MTLBuffer,
        routedBlobs: [any MTLBuffer],
        routedOffsets: MoEExpertOffsets,
        x: any MTLBuffer,
        acts: any MTLBuffer,
        activeSlots: any MTLBuffer,
        activeSlotsOffset: Int = 0,
        activeSlotIndices: [UInt32],
        activeCount: UInt32,
        d: UInt32,
        f: UInt32,
        topK: UInt32
    ) throws {
        guard activeCount > 0 else { return }
        validate(routedBlobs: routedBlobs, topK: topK)
        precondition(activeSlotIndices.count == Int(activeCount))
        let dimension = d
        let intermediate = f
        let expertCount = topK
        let active = activeCount
        let encoder = try commandBuffer.makeTracedComputeEncoder()
        encoder.setComputePipelineState(
            useRealDecodeConstants(d: d, f: f, topK: topK)
                ? phase1SubsetU16SpecializedPSO
                : phase1SubsetU16PSO)
        encoder.setBuffer(routedArgBuffer, offset: 0, index: 0)
        for slot in activeSlotIndices {
            encoder.useResource(routedBlobs[Int(slot)], usage: .read)
        }
        let offsets = routedOffsets
        encoder.setValue(offsets, index: 1)
        encoder.setBuffer(x, offset: 0, index: 2)
        encoder.setBuffer(acts, offset: 0, index: 3)
        encoder.setValue(dimension, index: 4)
        encoder.setValue(intermediate, index: 5)
        encoder.setValue(expertCount, index: 6)
        encoder.setBuffer(activeSlots, offset: activeSlotsOffset, index: 7)
        encoder.setValue(active, index: 8)
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(activeCount * f) + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeRoutedPersistentPhase2Reduce(
        commandBuffer: any MTLCommandBuffer,
        routedArgBuffer: any MTLBuffer,
        routedBlobs: [any MTLBuffer],
        routedOffsets: MoEExpertOffsets,
        acts: any MTLBuffer,
        routingWeights: any MTLBuffer,
        residual: any MTLBuffer,
        y: any MTLBuffer,
        d: UInt32,
        f: UInt32,
        topK: UInt32
    ) throws {
        validate(routedBlobs: routedBlobs, topK: topK)
        let dimension = d
        let intermediate = f
        let encoder = try commandBuffer.makeTracedComputeEncoder()
        encoder.setComputePipelineState(
            useRealDecodeConstants(d: d, f: f, topK: topK)
                ? phase2ReduceK8SpecializedPSO
                : phase2ReduceK8PSO)
        encoder.setBuffer(routedArgBuffer, offset: 0, index: 0)
        for buffer in routedBlobs { encoder.useResource(buffer, usage: .read) }
        let offsets = routedOffsets
        encoder.setValue(offsets, index: 1)
        encoder.setBuffer(acts, offset: 0, index: 2)
        encoder.setBuffer(routingWeights, offset: 0, index: 3)
        encoder.setBuffer(residual, offset: 0, index: 4)
        encoder.setBuffer(y, offset: 0, index: 5)
        encoder.setValue(dimension, index: 6)
        encoder.setValue(intermediate, index: 7)
        encoder.setValue(topK, index: 8)
        encoder.dispatchThreadgroups(
            MTLSize(width: Int(d), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// True when a routed argument buffer is a plain array of expert GPU
    /// addresses, which `encodeGatherRoutedSlots` writes on the GPU.
    var routedSlotsAreAddresses: Bool {
        routedArgEncoder.encodedLength == Self.maxStreamedExperts * MemoryLayout<UInt64>.stride
    }

    /// Fills `slots` (a routed argument buffer) on the GPU from the router's
    /// `indices` and an address table of `numExperts` entries, 0 where the
    /// expert is not bound; those slots get `fallback`. Copies `hidden` into
    /// `saved`.
    func encodeGatherRoutedSlots(commandBuffer: any MTLCommandBuffer,
                                 indices: any MTLBuffer,
                                 table: any MTLBuffer, tableOffset: Int,
                                 slots: any MTLBuffer,
                                 hidden: any MTLBuffer, saved: any MTLBuffer,
                                 topK: UInt32, numExperts: UInt32, d: UInt32,
                                 fallback: UInt64) throws {
        precondition(topK <= UInt32(Self.maxStreamedExperts))
        let k = topK, n = numExperts, dimension = d, zero = fallback
        let encoder = try commandBuffer.makeTracedComputeEncoder()
        encoder.setComputePipelineState(gatherRoutedSlotsPSO)
        encoder.setBuffer(indices, offset: 0, index: 0)
        encoder.setBuffer(table, offset: tableOffset, index: 1)
        encoder.setBuffer(slots, offset: 0, index: 2)
        encoder.setBuffer(hidden, offset: 0, index: 3)
        encoder.setBuffer(saved, offset: 0, index: 4)
        encoder.setValue(k, index: 5)
        encoder.setValue(n, index: 6)
        encoder.setValue(dimension, index: 7)
        encoder.setValue(zero, index: 8)
        encoder.dispatchThreads(MTLSize(width: 256, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Both routed phases over GPU-written `slots`, binding `resident`: every
    /// buffer a slot may point at.
    func encodeRoutedPersistentGPUSlots(commandBuffer: any MTLCommandBuffer,
                                        slots: any MTLBuffer,
                                        resident: [any MTLBuffer],
                                        routedOffsets: MoEExpertOffsets,
                                        x: any MTLBuffer,
                                        acts: any MTLBuffer,
                                        routingWeights: any MTLBuffer,
                                        residual: any MTLBuffer,
                                        y: any MTLBuffer,
                                        d: UInt32, f: UInt32, topK: UInt32) throws {
        precondition(topK > 0 && topK <= UInt32(Self.maxStreamedExperts))
        let specialized = useRealDecodeConstants(d: d, f: f, topK: topK)
        let offsets = routedOffsets
        let dimension = d, intermediate = f, expertCount = topK
        let phase1 = try commandBuffer.makeTracedComputeEncoder()
        phase1.setComputePipelineState(specialized ? phase1U16SpecializedPSO : phase1U16PSO)
        phase1.setBuffer(slots, offset: 0, index: 0)
        phase1.useResources(resident, usage: .read)
        phase1.setValue(offsets, index: 1)
        phase1.setBuffer(x, offset: 0, index: 2)
        phase1.setBuffer(acts, offset: 0, index: 3)
        phase1.setValue(dimension, index: 4)
        phase1.setValue(intermediate, index: 5)
        phase1.setValue(expertCount, index: 6)
        phase1.dispatchThreadgroups(
            MTLSize(width: (Int(topK * f) + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        phase1.endEncoding()
        let reduce = try commandBuffer.makeTracedComputeEncoder()
        reduce.setComputePipelineState(specialized ? phase2ReduceK8SpecializedPSO : phase2ReduceK8PSO)
        reduce.setBuffer(slots, offset: 0, index: 0)
        reduce.useResources(resident, usage: .read)
        reduce.setValue(offsets, index: 1)
        reduce.setBuffer(acts, offset: 0, index: 2)
        reduce.setBuffer(routingWeights, offset: 0, index: 3)
        reduce.setBuffer(residual, offset: 0, index: 4)
        reduce.setBuffer(y, offset: 0, index: 5)
        reduce.setValue(dimension, index: 6)
        reduce.setValue(intermediate, index: 7)
        reduce.setValue(expertCount, index: 8)
        reduce.dispatchThreadgroups(
            MTLSize(width: Int(d), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        reduce.endEncoding()
    }

    private func validate(routedBlobs: [any MTLBuffer], topK: UInt32) {
        precondition(topK > 0 && topK <= UInt32(Self.maxStreamedExperts))
        precondition(routedBlobs.count == Int(topK))
    }

    private func useRealDecodeConstants(d: UInt32, f: UInt32, topK: UInt32) -> Bool {
        d == Self.realDecodeD && f == Self.realDecodeF &&
            topK == UInt32(Self.maxStreamedExperts)
    }
}
