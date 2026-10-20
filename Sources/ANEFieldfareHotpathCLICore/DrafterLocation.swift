public import Foundation

/// Where the DFlash drafter is looked for when `--drafter` is not given:
/// beside the model, in a `dflash` folder beside the model (the README's
/// layout: `checkpoints/gemma-4-26b-a4b-it-qat.fieldfare` and
/// `checkpoints/dflash/gemma-4-26B-A4B-it-DFlash`), or in a `dflash` folder
/// next to the model's folder.
public enum DrafterLocation {
    public static let directoryName = "gemma-4-26B-A4B-it-DFlash"

    public static func candidates(model: String) -> [URL] {
        let folder = URL(fileURLWithPath: model).standardizedFileURL.deletingLastPathComponent()
        return [
            folder.appendingPathComponent(directoryName),
            folder.appendingPathComponent("dflash").appendingPathComponent(directoryName),
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
