import Darwin
import Testing

@testable import ANEFieldfareHotpath

@Suite(.serialized) struct InteractivityWatchdogEnvironmentTests {
    private static let key = "AGX_RELAX_CDM_CTXSTORE_TIMEOUT"

    private static func read() -> String? {
        guard let raw = unsafe getenv(key) else { return nil }
        return unsafe String(cString: raw)
    }

    private static func withCleanEnvironment(_ body: () throws -> Void) rethrows {
        let saved = read()
        unsafe unsetenv(key)
        defer {
            if let saved {
                unsafe setenv(key, saved, 1)
            } else {
                unsafe unsetenv(key)
            }
        }
        try body()
    }

    #if os(macOS)
    @Test func constructingAMetalContextSetsTheWatchdogRelaxationOnMacOS() throws {
        try Self.withCleanEnvironment {
            #expect(Self.read() == nil)
            _ = try MetalContext()
            #expect(Self.read() == "1")
        }
    }

    @Test func anExplicitOperatorOverrideIsNotOverwritten() throws {
        try Self.withCleanEnvironment {
            unsafe setenv(Self.key, "0", 1)
            _ = try MetalContext()
            #expect(Self.read() == "0")
        }
    }
    #endif
}
