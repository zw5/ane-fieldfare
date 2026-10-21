// Adapted from Turbo Fieldfare's installer (drumih/turbo-fieldfare 2e9ea50, Apache-2.0).
import Foundation

public enum RemoteChunkPolicy {
    public static let defaultBytes = 64 * 1024 * 1024
    public static let maxBytes = defaultBytes
}
