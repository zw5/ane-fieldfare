#!/usr/bin/env bash
# Release build of the installer, the CLI and the server into .build/swift-hotpath.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
for product in ANEFieldfareInstall ANEFieldfareHotpathCLI ANEFieldfareServer; do
  swift build -c release \
    --jobs "${ANE_FIELDFARE_BUILD_JOBS:-2}" \
    --product "$product" \
    --scratch-path "$project_root/.build/swift-hotpath" \
    --package-path "$project_root"
done
