# Third-party notices

The device-resident Swift decoder components under
`Sources/ANEFieldfareHotpath`, `Sources/ANEFieldfareImportedFormat`, and
`Sources/ANEFieldfareHotpathCLICore` were adapted from Turbo Fieldfare by
drumih, licensed under Apache License 2.0. ANE Fieldfare retains that license
and credits the upstream implementation while owning its local contracts,
modifications, tests, and release format.

Upstream: https://github.com/drumih/turbo-fieldfare

The imported runtime, compact format reader, and command-line runner are
synchronized through upstream commit
`4c6db1e698ea861609109d4bc517410ff302af46` (2026-09-08). `UPSTREAM.json`
records the mapping and retained ANE Fieldfare contracts.

Upstream regression tests and numerical reference helpers are included under
`Tests/ANEFieldfareUpstreamTests`, `Tests/ANEFieldfareFormatTests`, and
`Tests/Support/Validation`, alongside this repository's own tests in the same
targets. The repack writers under
`Tests/Support/Repack` support synthetic loader fixtures only. These files
retain the same Apache 2.0 attribution and license. This import does not add
the upstream GUI, server, or installer executables.

The decode early expert read (`Runtime/Inference/NextLayerExpertPrefetcher.swift`,
`Runtime/Inference/NextLayerGuessWeights.swift`) and the fitted next-layer
guess file `Sources/ANEFieldfareHotpath/NextLayerGuess/gemma-4-26b-a4b-it-4bit.bin`
are adapted from Turbo Fieldfare pull requests #187 and #188 by maxslamdunk
(https://github.com/maxslamdunk/turbo-fieldfare, branch
`early-expert-read-fitted`), Apache License 2.0. The fitted file holds the
Gemma 4 router weights plus a ridge-fitted correction and is used unmodified
(SHA-256 `b18242445534d452b7efb9c7d1c358884ef56637ad6003c187e5bced50dc3dcd`).
