import Testing
@testable import ANEFieldfareHotpath

/// The speculation scheduler: geometric expected yield, cost-priced choice
/// between plain decode and a drafted block, exploration back-off, confidence
/// calibration, and confidence-scheduled verify length.
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

    @Test func expectedTokensFollowGeometricAcceptance() {
        var policy = SpeculationPolicy(maxBlock: 16)
        Self.train(&policy, rows: 8, accepted: 7, seconds: 0.07, steps: 200)
        #expect(policy.acceptanceRate > 0.99)
        #expect(abs(policy.expectedTokens(length: 1) - 1) < 1e-9)
        #expect(policy.expectedTokens(length: 8) > 7.5)
    }

    @Test func highAcceptanceAndCheapVerifyChooseABlock() {
        var policy = SpeculationPolicy(maxBlock: 16)
        Self.train(&policy, rows: 8, accepted: 7, seconds: 0.07, steps: 100)
        #expect(policy.next(limit: 1_000) > 1)
    }

    @Test func lowAcceptanceFallsBackToPlainDecode() {
        var policy = SpeculationPolicy(maxBlock: 16)
        Self.train(&policy, rows: 8, accepted: 0, seconds: 0.07, confidence: 0.5, steps: 100)
        #expect(policy.next(limit: 1_000) == 1)
    }

    @Test func blockNeverExceedsTheLimit() {
        var policy = SpeculationPolicy(maxBlock: 16)
        Self.train(&policy, rows: 8, accepted: 7, seconds: 0.07, steps: 100)
        #expect(policy.next(limit: 3) <= 3)
        #expect(policy.next(limit: 1) == 1)
    }

    @Test func explorationBacksOffWhileSpeculationDoesNotPay() {
        var policy = SpeculationPolicy(maxBlock: 16)
        Self.train(&policy, rows: 8, accepted: 0, seconds: 0.07, confidence: 0.5, steps: 100)
        var gaps: [Int] = []
        var plain = 0
        for _ in 0..<400 {
            let length = policy.next(limit: 1_000)
            if length == 1 {
                plain += 1
                policy.recordPlain(seconds: 0.03)
            } else {
                gaps.append(plain)
                plain = 0
                Self.train(&policy, rows: length, accepted: 0, seconds: 0.05, confidence: 0.5, steps: 1)
            }
        }
        #expect(gaps.count >= 3)
        #expect(zip(gaps, gaps.dropFirst()).allSatisfy { $0.1 >= $0.0 })
        #expect(gaps.last! > gaps.first!)
    }

    @Test func calibrationLearnsFromVerifyOutcomes() {
        var policy = SpeculationPolicy(maxBlock: 16)
        // Drafts at confidence 0.95 are always rejected first.
        Self.train(&policy, rows: 4, accepted: 0, seconds: 0.04, confidence: 0.95, steps: 300)
        #expect(policy.calibrated(0.95) < 0.1)
        // Other deciles keep their prior.
        #expect(abs(policy.calibrated(0.35) - 0.35) < 0.05)
    }

    @Test func verifyLengthFollowsConfidence() {
        let policy = SpeculationPolicy(maxBlock: 16)
        #expect(policy.verifyLength(confidences: Array(repeating: 0.99, count: 7)) == 8)
        #expect(policy.verifyLength(confidences: [0.99, 0.99, 0.02, 0.99, 0.99]) == 3)
        #expect(policy.verifyLength(confidences: Array(repeating: 0.05, count: 7)) == 1)
    }
}
