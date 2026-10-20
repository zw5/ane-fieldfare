// swift-tools-version: 6.2
import PackageDescription

// Gemma 4 MoE runtime on Metal, with routed prefill experts on the ANE
// through a small Objective-C carrier over the private ANE runtime.

// Every target builds in Swift 6 mode with complete concurrency checking, and
// every warning is an error: a diagnostic is fixed by changing the code, not
// by suppressing it. Strict memory safety makes each unsafe operation carry an
// explicit `unsafe`. `Scripts/lint_escape_hatches.py` covers what the
// compiler cannot reject on its own.
let strictSwiftSettings: [SwiftSetting] = [
    .enableUpcomingFeature("ExistentialAny"),
    .enableUpcomingFeature("InternalImportsByDefault"),
    .enableUpcomingFeature("MemberImportVisibility"),
    .strictMemorySafety(),
    .treatAllWarnings(as: .error),
]

let package = Package(
    name: "ANEFieldfareHotpath",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "ANEFieldfareHotpath", targets: ["ANEFieldfareHotpath"]),
        .executable(name: "ANEFieldfareHotpathCLI", targets: ["ANEFieldfareHotpathCLI"]),
        .executable(name: "ANEFieldfareDFlashCapture", targets: ["ANEFieldfareDFlashCapture"]),
        .executable(name: "ANEFieldfareBench", targets: ["ANEFieldfareBench"]),
        .executable(name: "ANEFieldfareServer", targets: ["ANEFieldfareServer"]),
        .executable(name: "ANEFieldfareInstall", targets: ["ANEFieldfareInstall"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "ANEFieldfareValidationSupport",
            dependencies: ["ANEFieldfareHotpath"],
            path: "Tests/Support/Validation",
            swiftSettings: strictSwiftSettings
        ),
        .testTarget(
            name: "ANEFieldfareUpstreamTests",
            dependencies: [
                "ANEFieldfareHotpath", "ANEFieldfareHotpathCLICore",
                "ANEFieldfareModelFormat", "ANEFieldfareValidationSupport",
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Tests/ANEFieldfareUpstreamTests",
            resources: [.copy("Runtime/Vision/Fixtures/images")],
            swiftSettings: strictSwiftSettings
        ),
        .testTarget(
            name: "ANEFieldfareInstallTests",
            dependencies: ["ANEFieldfareInstallCore", "ANEFieldfareModelFormat"],
            path: "Tests/ANEFieldfareInstallTests",
            swiftSettings: strictSwiftSettings
        ),
        .testTarget(
            name: "ANEFieldfareServerTests",
            dependencies: [
                "ANEFieldfareServerCore", "ANEFieldfareHotpath", "ANEFieldfareHotpathCLICore",
                "ANEFieldfareValidationSupport",
            ],
            path: "Tests/ANEFieldfareServerTests",
            swiftSettings: strictSwiftSettings
        ),
        .testTarget(
            name: "ANEFieldfareHotpathTests",
            dependencies: ["ANEFieldfareHotpath", "ANEFieldfareHotpathCLICore"],
            path: "Tests/ANEFieldfareHotpathTests",
            swiftSettings: strictSwiftSettings
        ),
        .target(
            name: "ANEFieldfareANECarrier",
            path: "Sources/ANEFieldfareANECarrier",
            publicHeadersPath: "include",
            linkerSettings: [
                .linkedFramework("Foundation"),
                .linkedFramework("IOSurface"),
                .linkedLibrary("dl"),
            ]
        ),
        // The `.fieldfare` model directory, shared by the runtime and the installer.
        .target(
            name: "ANEFieldfareModelFormat",
            path: "Sources/ANEFieldfareModelFormat",
            swiftSettings: strictSwiftSettings
        ),
        // Builds a `.fieldfare` directory from Google's QAT GGUF.
        .target(
            name: "ANEFieldfareInstallCore",
            dependencies: ["ANEFieldfareModelFormat"],
            path: "Sources/ANEFieldfareInstallCore",
            swiftSettings: strictSwiftSettings
        ),
        .executableTarget(
            name: "ANEFieldfareInstall",
            dependencies: ["ANEFieldfareInstallCore"],
            path: "Sources/ANEFieldfareInstall",
            swiftSettings: strictSwiftSettings
        ),
        .target(
            name: "ANEFieldfareHotpath",
            dependencies: [
                "ANEFieldfareANECarrier",
                "ANEFieldfareModelFormat",
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Sources/ANEFieldfareHotpath",
            resources: [.copy("Metal"), .copy("NextLayerGuess"), .copy("HotVocabulary")],
            swiftSettings: strictSwiftSettings
        ),
        .target(
            name: "ANEFieldfareHotpathCLICore",
            dependencies: ["ANEFieldfareHotpath"],
            path: "Sources/ANEFieldfareHotpathCLICore",
            swiftSettings: strictSwiftSettings
        ),
        .executableTarget(
            name: "ANEFieldfareHotpathCLI",
            dependencies: ["ANEFieldfareHotpathCLICore"],
            path: "Sources/ANEFieldfareHotpathCLI",
            swiftSettings: strictSwiftSettings
        ),
        // OpenAI-compatible HTTP API over the same generation loop as the CLI.
        .target(
            name: "ANEFieldfareServerCore",
            dependencies: ["ANEFieldfareHotpath", "ANEFieldfareHotpathCLICore"],
            path: "Sources/ANEFieldfareServerCore",
            swiftSettings: strictSwiftSettings
        ),
        .executableTarget(
            name: "ANEFieldfareServer",
            dependencies: ["ANEFieldfareServerCore"],
            path: "Sources/ANEFieldfareServer",
            swiftSettings: strictSwiftSettings
        ),
        .executableTarget(
            name: "ANEFieldfareDFlashCapture",
            dependencies: ["ANEFieldfareHotpath"],
            path: "Sources/ANEFieldfareDFlashCapture",
            swiftSettings: strictSwiftSettings
        ),
        .executableTarget(
            name: "ANEFieldfareBench",
            dependencies: ["ANEFieldfareHotpath"],
            path: "Sources/ANEFieldfareBench",
            swiftSettings: strictSwiftSettings
        ),
    ],
    swiftLanguageModes: [.v6]
)
