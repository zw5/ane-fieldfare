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
        .executable(name: "ANEFieldfareQualityGate", targets: ["ANEFieldfareQualityGate"]),
        .executable(name: "ANEFieldfareDFlashCapture", targets: ["ANEFieldfareDFlashCapture"]),
        .executable(name: "ANEFieldfareBench", targets: ["ANEFieldfareBench"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        // Upstream repack writers construct the synthetic loader fixtures.
        // This support target is reachable only from the regression tests.
        .target(
            name: "ANEFieldfareRepackTestSupport",
            dependencies: ["ANEFieldfareImportedFormat"],
            path: "Tests/Support/Repack",
            swiftSettings: strictSwiftSettings
        ),
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
                "ANEFieldfareImportedFormat", "ANEFieldfareValidationSupport",
                "ANEFieldfareRepackTestSupport",
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Tests/ANEFieldfareUpstreamTests",
            resources: [.copy("Runtime/Vision/Fixtures/images")],
            swiftSettings: strictSwiftSettings
        ),
        .testTarget(
            name: "ANEFieldfareFormatTests",
            dependencies: ["ANEFieldfareImportedFormat"],
            path: "Tests/ANEFieldfareFormatTests",
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
        .target(
            name: "ANEFieldfareImportedFormat",
            path: "Sources/ANEFieldfareImportedFormat",
            swiftSettings: strictSwiftSettings
        ),
        .target(
            name: "ANEFieldfareHotpath",
            dependencies: [
                "ANEFieldfareANECarrier",
                "ANEFieldfareImportedFormat",
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Sources/ANEFieldfareHotpath",
            resources: [.copy("Metal"), .copy("NextLayerGuess")],
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
            resources: [.copy("Fixtures")],
            swiftSettings: strictSwiftSettings
        ),
        .executableTarget(
            name: "ANEFieldfareQualityGate",
            dependencies: ["ANEFieldfareHotpath"],
            path: "Sources/ANEFieldfareQualityGate",
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
