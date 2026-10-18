#!/usr/bin/env bash
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
"$project_root/Scripts/with_benchmark_lock.sh" swift test -c release --jobs "${ANE_FIELDFARE_BUILD_JOBS:-2}" --package-path "$project_root" \
  --scratch-path "$project_root/.build/swift-hotpath"
"$project_root/.build/swift-hotpath/release/ANEFieldfareWorkstreamPGate"

test_root="$project_root/.build/tests"
mkdir -p "$test_root"
xcrun clang -O3 -Wall -Wextra \
  "$project_root/Sources/ANEFieldfareRuntime/TensorIndex.c" \
  "$project_root/Sources/ANEFieldfareRuntime/Dequant.c" \
  "$project_root/Sources/ANEFieldfareDequantGate/main.c" \
  -o "$test_root/dequant"
"$test_root/dequant"

xcrun clang -O3 -Wall -Wextra -fobjc-arc -fblocks \
  -framework Foundation -framework IOSurface -ldl \
  -I "$project_root/Sources/ANEFieldfareRuntime" \
  "$project_root/Sources/ANEFieldfareRuntime/TensorIndex.c" \
  "$project_root/Sources/ANEFieldfareRuntime/ANEProgram.m" \
  "$project_root/Sources/ANEFieldfareRuntime/ANEExpertDispatch.m" \
  "$project_root/Tests/Native/ExpertPositions.m" \
  -o "$test_root/expert-positions"
"$project_root/Scripts/with_benchmark_lock.sh" "$test_root/expert-positions"

# The synthetic tests above need no checkpoint. Also validate the optional
# resident-FFN experiment against the installed compact weights when available.
shared_weights="$project_root/checkpoints/gturbo/gemma4.gturbo/model_weights.bin"
if [[ -f "$shared_weights" ]]; then
  xcrun clang -O3 -Wall -Wextra -fobjc-arc -fblocks \
    -framework Foundation -framework IOSurface -ldl \
    -I "$project_root/Sources/ANEFieldfareRuntime" \
    "$project_root/Sources/ANEFieldfareRuntime/TensorIndex.c" \
    "$project_root/Sources/ANEFieldfareRuntime/Dequant.c" \
    "$project_root/Sources/ANEFieldfareRuntime/ANEProgram.m" \
    "$project_root/Sources/ANEFieldfareRuntime/ANESharedFFN.m" \
    "$project_root/Tests/Native/SharedFFN.m" -o "$test_root/shared-ffn"
  "$project_root/Scripts/with_benchmark_lock.sh" "$test_root/shared-ffn" "$shared_weights"
fi

"$project_root/.build/ane-fieldfare-aot-expert-runtime-gate"
"$project_root/.build/ane-fieldfare-metal-attention-gate" \
  "$project_root/Sources/ANEFieldfareRuntime/Metal/attention.metal"
