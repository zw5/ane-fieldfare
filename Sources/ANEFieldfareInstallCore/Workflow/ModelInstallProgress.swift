// Adapted from Turbo Fieldfare's installer (drumih/turbo-fieldfare 2e9ea50, Apache-2.0).
import Foundation

public enum ModelInstallProgress: Equatable, Sendable {
    case downloadingMetadata
    case planning(downloadBytes: UInt64, outputBytes: UInt64)
    case checkingDisk(DiskSpaceRequirement)
    case reservingOutput(bytes: UInt64)
    case copyingPayload(
        reusedBytes: UInt64,
        downloadedThisRunBytes: UInt64,
        totalBytes: UInt64
    )
    case hashingOutput(String)
    case finalizing
}
