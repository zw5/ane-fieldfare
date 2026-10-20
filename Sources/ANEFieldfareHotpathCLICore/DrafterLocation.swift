public import Foundation

/// Where the DFlash drafter is looked for when `--drafter` is not given:
/// beside the model, or in a `dflash` folder next to the model's folder, the
/// layout the README's install steps use (`checkpoints/gturbo/gemma4.gturbo`
/// and `checkpoints/dflash/gemma-4-26B-A4B-it-DFlash`).
public enum DrafterLocation {
    public static let directoryName = "gemma-4-26B-A4B-it-DFlash"

    public static func candidates(model: String) -> [URL] {
        let folder = URL(fileURLWithPath: model).standardizedFileURL.deletingLastPathComponent()
        return [
            folder.appendingPathComponent(directoryName),
            folder.deletingLastPathComponent().appendingPathComponent("dflash")
                .appendingPathComponent(directoryName),
        ]
    }

    /// The first candidate holding a drafter checkpoint.
    public static func find(model: String) -> URL? {
        candidates(model: model).first {
            FileManager.default.fileExists(atPath: $0.appendingPathComponent("config.json").path)
        }
    }
}
