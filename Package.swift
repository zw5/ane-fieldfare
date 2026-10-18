// swift-tools-version: 6.2
import PackageDescription

// Gemma 4 MoE runtime on Metal, with routed prefill experts on the ANE
// through a small Objective-C carrier over the private ANE runtime.
let package = Package(
    name: "ANEFieldfareHotpath",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "ANEFieldfareHotpath", targets: ["ANEFieldfareHotpath"]),
        .executable(name: "ANEFieldfareHotpathCLI", targets: ["ANEFieldfareHotpathCLI"]),
        .executable(name: "ANEFieldfareQualityGate", targets: ["ANEFieldfareQualityGate"]),
        .executable(name: "ANEFieldfareDFlashCapture", targets: ["ANEFieldfareDFlashCapture"]),
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
            path: "Tests/Support/Repack"
        ),
        .target(
            name: "ANEFieldfareValidationSupport",
            dependencies: ["ANEFieldfareHotpath"],
            path: "Tests/Support/Validation"
        ),
        .testTarget(
            name: "ANEFieldfareUpstreamTests",
            dependencies: [
                "ANEFieldfareHotpath", "ANEFieldfareHotpathCLICore",
                "ANEFieldfareImportedFormat", "ANEFieldfareValidationSupport",
                "ANEFieldfareRepackTestSupport",
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Tests/ANEFieldfareUpstreamTests",
            resources: [.copy("Runtime/Vision/Fixtures/images")]
        ),
        .testTarget(
            name: "ANEFieldfareFormatTests",
            dependencies: ["ANEFieldfareImportedFormat"],
            path: "Tests/ANEFieldfareFormatTests"
        ),
        .testTarget(
            name: "ANEFieldfareHotpathTests",
            dependencies: ["ANEFieldfareHotpath", "ANEFieldfareHotpathCLICore"],
            path: "Tests/ANEFieldfareHotpathTests"
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
            path: "Sources/ANEFieldfareImportedFormat"
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
            resources: [.copy("Metal"), .copy("NextLayerGuess")]
        ),
        .target(
            name: "ANEFieldfareHotpathCLICore",
            dependencies: ["ANEFieldfareHotpath"],
            path: "Sources/ANEFieldfareHotpathCLICore"
        ),
        .executableTarget(
            name: "ANEFieldfareHotpathCLI",
            dependencies: ["ANEFieldfareHotpathCLICore"],
            path: "Sources/ANEFieldfareHotpathCLI",
            resources: [.copy("Fixtures")]
        ),
        .executableTarget(
            name: "ANEFieldfareQualityGate",
            dependencies: ["ANEFieldfareHotpath"],
            path: "Sources/ANEFieldfareQualityGate"
        ),
        .executableTarget(
            name: "ANEFieldfareDFlashCapture",
            dependencies: ["ANEFieldfareHotpath"],
            path: "Sources/ANEFieldfareDFlashCapture"
        ),
    ]
)
