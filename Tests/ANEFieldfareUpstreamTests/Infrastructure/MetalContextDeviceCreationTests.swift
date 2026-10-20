import Foundation
import Testing

@Suite struct MetalContextDeviceCreationTests {
    @Test func onlyMetalContextCreatesTheSystemDevice() throws {
        let sources = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources")
        let sourcePrefix = sources.standardizedFileURL.path + "/"
        let walker = try #require(FileManager.default.enumerator(
            at: sources, includingPropertiesForKeys: nil),
            "cannot locate Sources/ from \(#filePath)")

        var scanned = 0
        var offenders: [String] = []
        while let url = walker.nextObject() as? URL {
            let relativePath = String(url.standardizedFileURL.path.dropFirst(sourcePrefix.count))
            // Native ANE probes have their own device contracts. This imported
            // rule covers the library and CLI whose device setup was merged.
            guard url.pathExtension == "swift",
                  ["ANEFieldfareHotpath/", "ANEFieldfareHotpathCLICore/", "ANEFieldfareHotpathCLI/"]
                    .contains(where: relativePath.hasPrefix),
                  relativePath != "ANEFieldfareHotpath/Infrastructure/Metal/MetalContext.swift"
            else { continue }
            let text = try String(contentsOf: url, encoding: .utf8)
            scanned += 1
            for (index, line) in text.split(separator: "\n", omittingEmptySubsequences: false)
                .enumerated() where line.contains("MTLCreateSystemDefaultDevice(") {
                offenders.append("\(relativePath):\(index + 1)")
            }
        }

        #expect(scanned > 0, "no Swift sources scanned under \(sources.path)")
        #expect(offenders.isEmpty,
                "Production device creation bypasses MetalContext: \(offenders.joined(separator: ", "))")
    }
}
