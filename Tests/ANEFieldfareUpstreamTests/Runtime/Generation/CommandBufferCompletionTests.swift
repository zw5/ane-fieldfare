import Foundation
import Metal
import Testing

@testable import ANEFieldfareHotpath

@Suite struct CommandBufferCompletionTests {
    private static func nsError(description: String,
                                userInfo extra: [String: Any] = [:]) -> NSError {
        var info: [String: Any] = [NSLocalizedDescriptionKey: description]
        info.merge(extra) { current, _ in current }
        return NSError(domain: "MTLCommandBufferErrorDomain", code: 1, userInfo: info)
    }

    @Test func userInfoKeysAreNamedButValuesAreNotLeaked() throws {
        let detail = try #require(
            metalCommandBufferFailureDetail(
                label: "x",
                status: .error,
                error: Self.nsError(description: "boom",
                                    userInfo: ["TFSecretPayload": "do-not-leak-me"])))

        #expect(detail.contains("userInfoKeys="))
        #expect(detail.contains("TFSecretPayload"))
        #expect(!detail.contains("do-not-leak-me"))
    }

    @Test func completedStatusCarryingAnErrorIsAFailure() throws {
        let detail = try #require(
            metalCommandBufferFailureDetail(label: "x",
                                            status: .completed,
                                            error: Self.nsError(description: "boom")))
        #expect(detail.contains("status=completed"))
        #expect(detail.contains("description=boom"))
    }

    @Test func realNonCompletedCommandBufferThrows() throws {
        let context = try MetalContext()
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        commandBuffer.label = "test uncommitted"

        do {
            try checkCommandBufferError(commandBuffer)
            Issue.record("expected a failure for a buffer that never completed")
        } catch let error as MetalError {
            #expect("\(error)".contains("test uncommitted"))
        }
    }
}
