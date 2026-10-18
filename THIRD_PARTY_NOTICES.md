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
`Tests/Support/Validation`. The repack writers under
`Tests/Support/Repack` support synthetic loader fixtures only. These files
retain the same Apache 2.0 attribution and license. This import does not add
the upstream GUI, server, or installer executables.
