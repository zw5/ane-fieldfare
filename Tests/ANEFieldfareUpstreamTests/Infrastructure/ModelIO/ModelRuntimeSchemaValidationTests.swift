import Metal
import Testing

@testable import ANEFieldfareHotpath

@Suite struct ModelRuntimeSchemaValidationTests {
    @Test func rejectsMissingRequiredResidentTensor() throws {
        let model = try loadToy()
        var entries = model.residentIndex.entries
        entries["language_model.model.embed_tokens.weight"] = nil
        let index = ResidentIndex(header: model.residentIndex.header, entries: entries)

        #expect(throws: ModelError.self) {
            try Model.validateRuntimeSchema(
                residentIndex: index,
                layout: model.packedExpertsLayout,
                manifest: model.manifest,
                config: .gemma4Toy())
        }
    }

    @Test func rejectsRoutedTensorRangeOutsideUInt32() throws {
        let model = try loadToy()
        let layout = replacingExpert(model.packedExpertsLayout, layer: 0, expert: 0) { expert in
            var tensors = expert.subTensors
            guard let gate = tensors["gate"] else { return expert }
            tensors["gate"] = SubTensorEntry(
                offset: UInt64(UInt32.max),
                size: 2,
                dtype: gate.dtype,
                shape: gate.shape,
                bits: gate.bits)
            return ExpertEntry(
                expert: expert.expert,
                offset: expert.offset,
                size: expert.size,
                subTensors: tensors)
        }

        #expect(throws: ModelError.self) {
            try Model.validateRuntimeSchema(
                residentIndex: model.residentIndex,
                layout: layout,
                manifest: model.manifest,
                config: .gemma4Toy())
        }
    }

    @Test func rejectsOverflowingDerivedDimensions() throws {
        let model = try loadToy()
        let toy = ArchConfig.gemma4Toy()
        let config = ArchConfig(
            hiddenSize: toy.hiddenSize,
            intermediateSize: toy.intermediateSize,
            moeIntermediateSize: toy.moeIntermediateSize,
            numHeads: Int.max,
            numKVHeads: toy.numKVHeads,
            numFullKVHeads: toy.numFullKVHeads,
            headDim: 2,
            fullHeadDim: toy.fullHeadDim,
            vocabSize: toy.vocabSize,
            slidingWindow: toy.slidingWindow,
            finalLogitSoftcap: toy.finalLogitSoftcap,
            ropeTheta: toy.ropeTheta,
            fullRopeTheta: toy.fullRopeTheta,
            partialRotaryFactor: toy.partialRotaryFactor,
            numLayers: toy.numLayers,
            numExperts: toy.numExperts,
            topKExperts: toy.topKExperts,
            tieWordEmbeddings: toy.tieWordEmbeddings,
            attentionKEqV: toy.attentionKEqV,
            fullAttentionLayerMask: toy.fullAttentionLayerMask,
            hiddenActivation: toy.hiddenActivation)
        #expect(throws: ModelError.self) {
            try Model.validateRuntimeSchema(
                residentIndex: model.residentIndex,
                layout: model.packedExpertsLayout,
                manifest: model.manifest,
                config: config)
        }
    }

    private func loadToy() throws -> Model {
        let directory = try ModelLoaderTests.writeToySynthetic()
        let device = try #require(MTLCreateSystemDefaultDevice())
        return try Model.load(
            directoryURL: directory,
            device: device,
            expecting: .gemma4Toy())
    }

    private func replacingExpert(
        _ layout: PackedExpertsLayout,
        layer layerIndex: Int,
        expert expertIndex: Int,
        transform: (ExpertEntry) -> ExpertEntry
    ) -> PackedExpertsLayout {
        var layers = layout.layers
        let layer = layers[layerIndex]
        var experts = layer.experts
        experts[expertIndex] = transform(experts[expertIndex])
        layers[layerIndex] = LayerLayout(
            layer: layer.layer,
            file: layer.file,
            experts: experts)
        return PackedExpertsLayout(
            expertStride: layout.expertStride,
            numLayers: layout.numLayers,
            expertsPerLayer: layout.expertsPerLayer,
            layers: layers)
    }
}
