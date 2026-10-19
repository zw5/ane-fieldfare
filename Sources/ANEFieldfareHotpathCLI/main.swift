import Darwin
import Dispatch
import Foundation
import ANEFieldfareHotpathCLICore

let rawArgv = Array(CommandLine.arguments.dropFirst())
let parsedArgs: Args
do {
    parsedArgs = try Args.parse(rawArgv)
} catch ArgsError.helpRequested {
    print(Args.usage)
    exit(0)
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n\n".utf8))
    FileHandle.standardError.write(Data(Args.usage.utf8))
    FileHandle.standardError.write(Data("\n".utf8))
    exit(2)
}

// The process entry point owns the one unstructured task; SIGINT cancels it
// and its result is the exit code. Awaiting it leaves the main thread free.
let generation = Task { await run(args: parsedArgs) }

signal(SIGINT, SIG_IGN)
let sigintSource = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global())
sigintSource.setEventHandler { generation.cancel() }
sigintSource.resume()

exit(await generation.value.exitCode)
