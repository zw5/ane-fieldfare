// swift-tools-version: 6.2
import PackageDescription

// The supported Swift decoder integrates the owned C/Objective-C ANE program
// and device-carrier ABI. Native probes retain Scripts/build.sh entry points.
let package = Package(
    name: "ANEFieldfareHotpath",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "ANEFieldfareHotpath", targets: ["ANEFieldfareHotpath"]),
        .library(name: "ANEFieldfareAOTExpertRuntime", targets: ["ANEFieldfareAOTExpertRuntime"]),
        .executable(name: "ANEFieldfareHotpathCLI", targets: ["ANEFieldfareHotpathCLI"]),
        .executable(name: "ANEFieldfarePrefill128Oracle", targets: ["ANEFieldfarePrefill128Oracle"]),
        .executable(name: "ANEFieldfareWorkstreamPGate", targets: ["ANEFieldfareWorkstreamPGate"]),
        .executable(name: "ANEFieldfareQualityGate", targets: ["ANEFieldfareQualityGate"]),
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
            path: "Sources/ANEFieldfareValidationSupport"
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
            name: "ANEFieldfareAOTExpertRuntime",
            path: "Sources/ANEFieldfareAOTExpertRuntime",
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
                "ANEFieldfareAOTExpertRuntime",
                "ANEFieldfareImportedFormat",
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Sources/ANEFieldfareHotpath",
            resources: [.copy("Metal")]
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
            name: "ANEFieldfarePrefill128Oracle",
            dependencies: ["ANEFieldfareHotpath"],
            path: "Sources/ANEFieldfarePrefill128Oracle"
        ),
        .executableTarget(
            name: "ANEFieldfareWorkstreamPGate",
            dependencies: ["ANEFieldfareHotpath"],
            path: "Sources/ANEFieldfareWorkstreamPGate"
        ),
        .executableTarget(
            name: "ANEFieldfareQualityGate",
            dependencies: ["ANEFieldfareHotpath"],
            path: "Sources/ANEFieldfareQualityGate"
        ),
    ]
)
