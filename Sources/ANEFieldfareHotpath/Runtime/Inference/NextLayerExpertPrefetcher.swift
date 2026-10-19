// Adapted from Turbo Fieldfare PRs #187 and #188 by maxslamdunk (Apache-2.0).
// Changed here: the early read is a storage advisory into the page cache, so
// it never blocks decode and never needs staging buffers; a candidate must
// rank in the guess's top-k and be absent from memory.
import Foundation
import Metal

/// Counts for the next-layer early read, summed over decode forwards.
public struct EarlyExpertReadStats: Sendable, Equatable {
    /// Decode forwards that ran with the early read.
    public var forwards: UInt64 = 0
    /// Routed experts that were absent when their layer routed to them.
    public var misses: UInt64 = 0
    /// Early reads issued.
    public var reads: UInt64 = 0
    /// Early reads whose expert the next layer then routed to. `reads - used`
    /// were wasted bandwidth.
    public var used: UInt64 = 0

    public init() {}

    /// The share of early reads that were used.
    public var precision: Double? {
        reads > 0 ? Double(used) / Double(reads) : nil
    }
}

/// Decode control: ask storage for the next layer's likeliest absent experts
/// while it would otherwise sit idle.
///
/// Per layer L < last: cb1(L) also scores layer L+1's experts over L's router
/// input, with the fitted guess shipped for this model (`NextLayerGuessWeights`)
/// or else with L+1's own router. After L's own reads finish, the best-scoring
/// experts that are not in memory are advised into the page cache. Layer L+1
/// binds the same buffers either way, so routes and generated text never
/// depend on this; a wrong guess only spends bandwidth.
///
/// `ANE_FIELDFARE_EARLY_EXPERT_READS` (1–8) sets the reads per layer.
final class NextLayerExpertPrefetcher {
    static let readsEnvironmentKey = "ANE_FIELDFARE_EARLY_EXPERT_READS"
    static let defaultReadsPerLayer = 2
    static let maxReadsPerLayer = 8

    private var issued: (layer: Int, experts: Set<Int>)?
    private(set) var stats = EarlyExpertReadStats()
    /// The fitted guess, or nil to score with the next layer's own router.
    let guessWeights: NextLayerGuessWeights?
    /// Early reads to issue for the next layer.
    let readsPerLayer: Int

    init(model: Model, guess: RuntimeEarlyExpertRead, device: any MTLDevice,
         environment: [String: String] = ProcessInfo.processInfo.environment) throws {
        readsPerLayer = environment[Self.readsEnvironmentKey].flatMap(Int.init)
            .flatMap { (1...Self.maxReadsPerLayer).contains($0) ? $0 : nil } ?? Self.defaultReadsPerLayer
        let config = model.config
        guessWeights = guess == .fitted
            ? try NextLayerGuessWeights.bundled(
                sourceSnapshotHash: model.sourceSnapshotHash, numLayers: config.numLayers,
                numExperts: config.numExperts, hiddenSize: config.hiddenSize, device: device)
            : nil
    }

    /// Which guess scores the next layer, `fitted` or `router`.
    var guess: RuntimeEarlyExpertRead { guessWeights == nil ? .router : .fitted }

    /// Up to `limit` highest-scoring experts for which `isAbsent` holds, among
    /// the `topK` highest-scoring overall, best first; ties go to the lower ID.
    static func choose(scores: [Float],
                       limit: Int, topK: Int = 8, isAbsent: (Int) -> Bool) -> [Int] {
        let ranked = scores.indices.sorted {
            scores[$0] != scores[$1] ? scores[$0] > scores[$1] : $0 < $1
        }
        return Array(ranked.prefix(topK).filter(isAbsent).prefix(limit))
    }

    func beginForward() {
        issued = nil
        stats.forwards += 1
    }

    /// Records the layer's routed experts and how many of them were absent.
    func observe(layer: Int, experts: [Int], absent: Int) {
        stats.misses += UInt64(absent)
        if let issued, issued.layer == layer {
            stats.used += UInt64(experts.filter(issued.experts.contains).count)
        }
    }

    /// Advises `experts` of `layer` into the page cache.
    func issue(layer: Int, experts: [Int], model: Model) throws {
        try model.prefetchRoutedExperts(layer: layer, experts: experts)
        issued = (layer, Set(experts))
        stats.reads += UInt64(experts.count)
    }
}
