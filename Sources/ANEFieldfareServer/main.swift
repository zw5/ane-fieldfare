import Darwin
import Dispatch
import Foundation
import ANEFieldfareServerCore

let options: ServerOptions
do {
    options = try ServerOptions.parse(Array(CommandLine.arguments.dropFirst()))
} catch ServerOptions.ParseError.helpRequested {
    print(ServerOptions.usage)
    exit(0)
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n\n\(ServerOptions.usage)\n".utf8))
    exit(2)
}

let log: @Sendable (String) -> Void = { line in
    FileHandle.standardError.write(Data((line + "\n").utf8))
}

// The process entry point owns the one unstructured task; SIGINT and SIGTERM
// cancel it, which stops the listener and every connection's task.
let server = Task { await runServer(options: options, log: log) }

/// A signal source that cancels `task`. Not top-level code: a closure written
/// there is main-actor isolated, and the handler runs on a global queue.
func cancelling(_ task: Task<Int32, Never>, on signalNumber: Int32) -> any DispatchSourceSignal {
    signal(signalNumber, SIG_IGN)
    let source = DispatchSource.makeSignalSource(signal: signalNumber, queue: .global())
    source.setEventHandler { @Sendable in task.cancel() }
    source.resume()
    return source
}

let signalSources = [cancelling(server, on: SIGINT), cancelling(server, on: SIGTERM)]

exit(await server.value)
