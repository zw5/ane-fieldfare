#!/usr/bin/env bash
# Release build of the CLI and the logits oracle into .build/swift-hotpath.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
for product in ANEFieldfareHotpathCLI ANEFieldfareQualityGate; do
  swift build -c release \
    --jobs "${ANE_FIELDFARE_BUILD_JOBS:-2}" \
    --product "$product" \
    --scratch-path "$project_root/.build/swift-hotpath" \
    --package-path "$project_root"
done
