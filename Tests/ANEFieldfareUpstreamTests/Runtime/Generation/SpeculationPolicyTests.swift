import Testing
@testable import ANEFieldfareHotpath

/// The speculation scheduler: cost-priced choice between plain decode and a
/// drafted block, the caller's limit, and confidence-scheduled verify length.
@Suite struct SpeculationPolicyTests {
    /// Feeds `steps` verified blocks of `rows` rows accepting `accepted` drafts
    /// at `confidence` each.
    private static func train(_ policy: inout SpeculationPolicy, rows: Int, accepted: Int,
                              seconds: Double, confidence: Float = 0.95, steps: Int) {
        for _ in 0..<steps {
            policy.recordDraft(length: rows, seconds: 0.008)
            policy.recordVerify(rows: rows, seconds: seconds, accepted: accepted,
                                confidences: Array(repeating: confidence, count: rows - 1),
                                stepTokens: accepted + 1, stepSeconds: seconds + 0.008)
        }
    }

    @Test func aBlockIsChosenOnlyWhenAcceptancePaysForIt() {
        var paying = SpeculationPolicy(maxBlock: 16)
        Self.train(&paying, rows: 8, accepted: 7, seconds: 0.07, steps: 100)
        #expect(paying.next(limit: 1_000) > 1)

        var losing = SpeculationPolicy(maxBlock: 16)
        Self.train(&losing, rows: 8, accepted: 0, seconds: 0.07, confidence: 0.5, steps: 100)
        #expect(losing.next(limit: 1_000) == 1)
    }

    @Test func blockNeverExceedsTheLimit() {
        var policy = SpeculationPolicy(maxBlock: 16)
        Self.train(&policy, rows: 8, accepted: 7, seconds: 0.07, steps: 100)
        #expect(policy.next(limit: 3) <= 3)
        #expect(policy.next(limit: 1) == 1)
    }

    @Test func verifyLengthFollowsConfidence() {
        let policy = SpeculationPolicy(maxBlock: 16)
        #expect(policy.verifyLength(confidences: Array(repeating: 0.99, count: 7)) == 8)
        #expect(policy.verifyLength(confidences: [0.99, 0.99, 0.02, 0.99, 0.99]) == 3)
        #expect(policy.verifyLength(confidences: Array(repeating: 0.05, count: 7)) == 1)
    }
}
