import Metal

struct PrefillChunkScratchLayout: Sendable, Equatable {
    let chunkTokens: Int
    let hiddenSize: Int
    let maxQElementsPerToken: Int
    let maxKVElementsPerToken: Int
    let sharedIntermediate: Int
    let routedIntermediate: Int
    let topK: Int
    /// Routed pairs one grouped-MoE microbatch processes.
    let routedPairMicrobatchRows = 32

    init(config: ArchConfig,
         chunkTokens: Int,
         chunkTokenLimit: Int = PrefillRuntimeConfig.maxChunkTokens) {
        self.chunkTokens = max(1, min(chunkTokens, chunkTokenLimit))
        self.hiddenSize = config.hiddenSize
        self.maxQElementsPerToken = config.numHeads * max(config.headDim, config.fullHeadDim)
        self.maxKVElementsPerToken = max(config.numKVHeads * config.headDim,
                                         config.numFullKVHeads * config.fullHeadDim)
        self.sharedIntermediate = config.intermediateSize
        self.routedIntermediate = config.moeIntermediateSize
        self.topK = config.topKExperts
    }

    var hiddenElements: Int { chunkTokens * hiddenSize }
    var qElements: Int { chunkTokens * maxQElementsPerToken }
    var kvElements: Int { chunkTokens * maxKVElementsPerToken }
    var routeIDElements: Int { chunkTokens * topK }
}

struct PrefillChunkScratchBuffers {
    let layout: PrefillChunkScratchLayout
    let hidden: any MTLBuffer
    let normed: any MTLBuffer
    let q: any MTLBuffer
    let kStage: any MTLBuffer
    let vStage: any MTLBuffer
    let attentionOutput: any MTLBuffer
    let denseX: any MTLBuffer
    let routedX: any MTLBuffer
    let routerX: any MTLBuffer
    let h1: any MTLBuffer
    let h2: any MTLBuffer
    let routePartials: any MTLBuffer
    let routeIDs: any MTLBuffer
    let routeWeights: any MTLBuffer
    let sharedActScratch: any MTLBuffer
    let routedGateUpActScratch: any MTLBuffer
    let routedDownScratch: any MTLBuffer

    static func allocate(device: any MTLDevice,
                         layout: PrefillChunkScratchLayout) throws -> PrefillChunkScratchBuffers {
        func privateBuffer(_ elements: Int, label: String) throws -> any MTLBuffer {
            guard let buffer = device.makeBuffer(
                length: max(elements, 1) * MemoryLayout<Float16>.stride,
                options: .storageModePrivate)
            else {
                throw ModelError.residentBufferWrapFailed
            }
            buffer.label = label
            return buffer
        }

        func sharedBuffer(_ bytes: Int, label: String) throws -> any MTLBuffer {
            guard let buffer = device.makeBuffer(length: max(bytes, 1),
                                                options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            buffer.label = label
            return buffer
        }

        let hidden = layout.hiddenElements
        let microbatch = layout.routedPairMicrobatchRows
        return PrefillChunkScratchBuffers(
            layout: layout,
            hidden: try privateBuffer(hidden, label: "prefill.hidden"),
            normed: try privateBuffer(hidden, label: "prefill.normed"),
            q: try privateBuffer(layout.qElements, label: "prefill.q"),
            kStage: try privateBuffer(layout.kvElements, label: "prefill.kStage"),
            vStage: try privateBuffer(layout.kvElements, label: "prefill.vStage"),
            attentionOutput: try privateBuffer(layout.qElements, label: "prefill.attnOut"),
            denseX: try privateBuffer(hidden, label: "prefill.denseX"),
            routedX: try privateBuffer(hidden, label: "prefill.routedX"),
            routerX: try privateBuffer(hidden, label: "prefill.routerX"),
            h1: try privateBuffer(hidden, label: "prefill.h1"),
            h2: try privateBuffer(hidden, label: "prefill.h2"),
            routePartials: try privateBuffer(hidden * layout.topK, label: "prefill.routePartials"),
            routeIDs: try sharedBuffer(layout.routeIDElements * MemoryLayout<UInt32>.stride,
                                       label: "prefill.routeIDs"),
            routeWeights: try sharedBuffer(layout.routeIDElements * MemoryLayout<Float16>.stride,
                                           label: "prefill.routeWeights"),
            sharedActScratch: try privateBuffer(layout.sharedIntermediate, label: "prefill.sharedActScratch"),
            routedGateUpActScratch: try privateBuffer(3 * microbatch * layout.routedIntermediate,
                                                      label: "prefill.routedGateUpActScratch"),
            routedDownScratch: try privateBuffer(microbatch * layout.hiddenSize,
                                                 label: "prefill.routedDownScratch"))
    }
}
