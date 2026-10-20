import ANEFieldfareInstallCore
import ANEFieldfareModelFormat
import Foundation

let usage = """
Usage: ANEFieldfareInstall [options]
       ANEFieldfareInstall --verify <model.fieldfare>

Builds the model directory from Google's QAT Q4_0 release of Gemma 4
26B-A4B: the text model (14.4 GB) and the vision tower (1.2 GB). Each GGUF
is streamed from Hugging Face, checked against its pinned SHA-256 and
converted as it arrives; nothing but the output is written to disk. An
interrupted install resumes where it stopped.

Options:
  --output <dir>        Model directory to create
                        (default: checkpoints/\(SupportedModel.directoryName))
  --from-gguf <file>    Convert a local copy of the pinned text GGUF instead
  --from-mmproj <file>  Convert a local copy of the pinned vision GGUF instead
  --no-vision           Install the text model only
  --tokenizer <dir>     Take the pinned tokenizer files from a local directory
  --connections <n>     Parallel range requests (default 8)
  --overwrite           Replace an existing model directory
  --restart             Discard a partial install instead of resuming it
"""

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}

struct Arguments {
    var rest: [String]

    mutating func value(_ flag: String) -> String? {
        guard let i = rest.firstIndex(of: flag) else { return nil }
        guard i + 1 < rest.count else { fail("\(flag) needs a value") }
        let v = rest[i + 1]
        rest.removeSubrange(i...(i + 1))
        return v
    }

    mutating func flag(_ name: String) -> Bool {
        guard let i = rest.firstIndex(of: name) else { return false }
        rest.remove(at: i)
        return true
    }
}

var arguments = Arguments(rest: Array(CommandLine.arguments.dropFirst()))

if arguments.flag("--help") || arguments.flag("-h") {
    print(usage)
    exit(0)
}

let log: @Sendable (String) -> Void = { print($0) }

if let directory = arguments.value("--verify") {
    do {
        let manifest = try await Installer.verify(URL(fileURLWithPath: directory))
        print("verified \(manifest.files.count) files of \(manifest.modelID)")
        exit(0)
    } catch {
        fail("\(error)")
    }
}

let output = arguments.value("--output") ?? "checkpoints/\(SupportedModel.directoryName)"
let localGGUF = arguments.value("--from-gguf")
let localMMProj = arguments.value("--from-mmproj")
let noVision = arguments.flag("--no-vision")
let tokenizer = arguments.value("--tokenizer")
let connections = arguments.value("--connections").map { Int($0) ?? 0 } ?? 8
let overwrite = arguments.flag("--overwrite")
let restart = arguments.flag("--restart")
guard arguments.rest.isEmpty else { fail("unknown arguments: \(arguments.rest.joined(separator: " "))\n\n\(usage)") }
guard (1...32).contains(connections) else { fail("--connections must be 1...32") }

do {
    let source: any ByteSource = try localGGUF.map { try LocalFileSource(path: $0) }
        ?? HuggingFaceSource(file: SupportedModel.gguf, concurrency: connections)
    var options = InstallOptions(
        destination: URL(fileURLWithPath: output),
        source: source,
        sourceFile: SupportedModel.gguf,
        tokenizer: tokenizer.map { .directory(URL(fileURLWithPath: $0)) } ?? .huggingFace,
        tokenizerFiles: SupportedModel.tokenizer,
        modelID: SupportedModel.modelID)
    if !noVision {
        let visionSource: any ByteSource = try localMMProj.map { try LocalFileSource(path: $0) }
            ?? HuggingFaceSource(file: SupportedModel.mmproj, concurrency: connections)
        options.vision = (visionSource, SupportedModel.mmproj)
    }
    options.overwrite = overwrite
    options.restart = restart
    options.log = log
    try await Installer.install(options)
} catch {
    fail("install failed: \(error)")
}
