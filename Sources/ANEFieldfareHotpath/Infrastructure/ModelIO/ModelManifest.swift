import ANEFieldfareModelFormat
import Foundation

/// Reads `manifest.json` and checks it against the one architecture the
/// runtime's kernels implement.
enum ModelManifest {
    static func load(_ directory: ModelDirectory, path: String, expecting: ArchConfig) throws -> FieldfareManifest {
        let data: Data
        do {
            data = try directory.readMetadata(FieldfareFormat.manifestPath,
                                              maxBytes: FieldfareFormat.manifestMaxBytes)
        } catch ModelError.missingFile {
            throw ModelError.partialInstall(path: path)
        }
        let manifest: FieldfareManifest
        do { manifest = try FieldfareManifestCodec.decode(data) }
        catch { throw ModelError.invalidManifest(detail: "\(error)") }
        try validateArch(manifest.arch, expected: expecting)
        return manifest
    }

    static func validateArch(_ a: FieldfareArch, expected e: ArchConfig) throws {
        func check<T: Equatable & CustomStringConvertible>(
            _ field: String, _ actual: T, _ expected: T) throws {
            if actual != expected {
                throw ModelError.archMismatch(field: field, expected: "\(expected)", actual: "\(actual)")
            }
        }
        try check("hiddenSize",          a.hiddenSize,          e.hiddenSize)
        try check("intermediateSize",    a.intermediateSize,    e.intermediateSize)
        try check("moeIntermediateSize", a.moeIntermediateSize, e.moeIntermediateSize)
        try check("numHeads",            a.numHeads,            e.numHeads)
        try check("numKVHeads",          a.numKVHeads,          e.numKVHeads)
        try check("numFullKVHeads",      a.numFullKVHeads,      e.numFullKVHeads)
        try check("headDim",             a.headDim,             e.headDim)
        try check("fullHeadDim",         a.fullHeadDim,         e.fullHeadDim)
        try check("vocabSize",           a.vocabSize,           e.vocabSize)
        try check("slidingWindow",       a.slidingWindow,       e.slidingWindow)
        try check("finalLogitSoftcap",   a.finalLogitSoftcap,   e.finalLogitSoftcap)
        try check("ropeTheta",           a.ropeTheta,           e.ropeTheta)
        try check("fullRopeTheta",       a.fullRopeTheta,       e.fullRopeTheta)
        try check("partialRotaryFactor", a.partialRotaryFactor, e.partialRotaryFactor)
        try check("numLayers",           a.numLayers,           e.numLayers)
        try check("numExperts",          a.numExperts,          e.numExperts)
        try check("topKExperts",         a.topKExperts,         e.topKExperts)
        try check("tieWordEmbeddings",   a.tieWordEmbeddings,   e.tieWordEmbeddings)
        try check("attentionKEqV",       a.attentionKEqV,       e.attentionKEqV)
        try check("hiddenActivation",    a.hiddenActivation,    e.hiddenActivation)
        let fullLayers = e.fullAttentionLayerMask.indices.filter { e.fullAttentionLayerMask[$0] != 0 }
        try check("fullAttentionLayers", a.fullAttentionLayers.description, fullLayers.description)
    }
}
