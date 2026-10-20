# Third-party notices

The device-resident Swift decoder components under
`Sources/ANEFieldfareHotpath` and `Sources/ANEFieldfareHotpathCLICore` were
adapted from Turbo Fieldfare by
drumih, licensed under Apache License 2.0. ANE Fieldfare retains that license
and credits the upstream implementation while owning its local contracts,
modifications, tests, and release format.

Upstream: https://github.com/drumih/turbo-fieldfare

The imported runtime and command-line runner are synchronized through
upstream commit
`4c6db1e698ea861609109d4bc517410ff302af46` (2026-09-08). `UPSTREAM.json`
records the mapping and retained ANE Fieldfare contracts.

Upstream regression tests and numerical reference helpers are included under
`Tests/ANEFieldfareUpstreamTests` and `Tests/Support/Validation`. These files
retain the same Apache 2.0 attribution and license. This import does not add
the upstream GUI, server, or installer executables.

The `.fieldfare` model format (`Sources/ANEFieldfareModelFormat`) and its
installer (`Sources/ANEFieldfareInstall`, `Sources/ANEFieldfareInstallCore`)
replace Turbo Fieldfare's `.gturbo` format and installer and are written for
this repository. The installer reads GGUF files and reproduces two ggml
routines from llama.cpp (MIT License, https://github.com/ggml-org/llama.cpp):
`dequantize_row_q6_K` and `quantize_row_q4_0_ref`.

The model weights are Google's Gemma 4 26B-A4B QAT release
(`google/gemma-4-26B-A4B-it-qat-q4_0-gguf`, Apache License 2.0), downloaded by
the installer and not included here.

The decode early expert read (`Runtime/Inference/NextLayerExpertPrefetcher.swift`,
`Runtime/Inference/NextLayerGuessWeights.swift`) and the fitted next-layer
guess file `Sources/ANEFieldfareHotpath/NextLayerGuess/gemma-4-26b-a4b-it-4bit.bin`
are adapted from Turbo Fieldfare pull requests #187 and #188 by maxslamdunk
(https://github.com/maxslamdunk/turbo-fieldfare, branch
`early-expert-read-fitted`), Apache License 2.0. The fitted file holds the
Gemma 4 router weights plus a ridge-fitted correction and is used unmodified
(SHA-256 `b18242445534d452b7efb9c7d1c358884ef56637ad6003c187e5bced50dc3dcd`).
