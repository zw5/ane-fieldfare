#!/usr/bin/env bash
# Build, then run every Swift Testing and XCTest suite. ANE tests need ANE
# hardware; checkpoint tests run when the installed checkpoint is present.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# XCTest is not shipped in the standalone Command Line Tools installation.
# Select a full installed Xcode for this process only; honor an explicit SDK.
if [[ -z "${DEVELOPER_DIR:-}" && "$(xcode-select -p)" == */CommandLineTools ]]; then
  for developer in /Applications/Xcode.app/Contents/Developer /Applications/Xcode-beta.app/Contents/Developer; do
    if [[ -d "$developer" ]]; then
      export DEVELOPER_DIR="$developer"
      break
    fi
  done
fi

"$project_root/Scripts/build.sh"
"$project_root/Scripts/with_benchmark_lock.sh" swift test -c release --jobs "${ANE_FIELDFARE_BUILD_JOBS:-2}" \
  --package-path "$project_root" --scratch-path "$project_root/.build/swift-hotpath"
