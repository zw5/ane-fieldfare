#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$project_root/.build"

core_only=1
case "${1:-}" in
  "") ;;
  --core) core_only=1 ;;
  --historical) core_only=0 ;;
  *) echo "usage: $0 [--core|--historical]" >&2; exit 2 ;;
esac
(( $# <= 1 )) || { echo "usage: $0 [--core|--historical]" >&2; exit 2; }

build_hotpath() {
  swift build -c release \
    --jobs "${ANE_FIELDFARE_BUILD_JOBS:-2}" \
    --product ANEFieldfareHotpathCLI \
    --scratch-path "$project_root/.build/swift-hotpath" \
    --package-path "$project_root"
}

# These historical one-command products were replaced by the two multicall
# suites below. Remove only the exact derived outputs so a successful build
# cannot appear to preserve the old canonical executable surface.
obsolete_gate_outputs=(
  ane-fieldfare-compressed-constant-gate
  ane-fieldfare-compression-spelling-probe-gate
  ane-fieldfare-native-dtype-probe-gate
  ane-fieldfare-native-dtype-gate
  ane-fieldfare-program-residency-gate
  ane-fieldfare-resident-projection-gate
  ane-fieldfare-private-surface-gate
  ane-fieldfare-system-compressed-model-gate
  ane-fieldfare-compressed-compute-plan-gate
  ane-fieldfare-coreml-compile-gate
  ane-fieldfare-aot-compressed-projection-gate
  ane-fieldfare-dequant-gate
  ane-fieldfare-carrier-write-probe
  ane-fieldfare-prep-breakdown-gate
  ane-fieldfare-position-lane-gate
  ane-fieldfare-transport-ceiling-gate
  ane-fieldfare-expert-layout-gate
  ane-fieldfare-explicit-read-gate
)
for output in "${obsolete_gate_outputs[@]}"; do
  rm -f "$project_root/.build/$output"
done

xcrun clang -O2 -Wall -Wextra -fobjc-arc -fblocks \
  -framework Foundation \
  -ldl \
  "$project_root/Sources/ANEFieldfareCapabilityGate/main.m" \
  -o "$project_root/.build/ane-fieldfare-capability-gate"

xcrun clang -O2 -Wall -Wextra -fobjc-arc \
  -framework Foundation \
  -ldl \
  "$project_root/Sources/ANEFieldfareRuntimeMetadataProbe/main.m" \
  -o "$project_root/.build/ane-fieldfare-runtime-metadata-probe"

xcrun clang -O2 -Wall -Wextra -fobjc-arc -fblocks \
  -framework Foundation -framework IOSurface \
  -ldl \
  "$project_root/Sources/ANEFieldfareIntermediateTensorGate/main.m" \
  -o "$project_root/.build/ane-fieldfare-intermediate-tensor-gate"

xcrun clang -O2 -Wall -Wextra -fobjc-arc \
  -framework Foundation \
  -framework IOSurface \
  -ldl \
  "$project_root/Sources/ANEFieldfareChainingGate/main.m" \
  -o "$project_root/.build/ane-fieldfare-chaining-gate"

xcrun clang -O3 -Wall -Wextra -fobjc-arc -fblocks \
  -framework Foundation -framework IOSurface -ldl \
  "$project_root/Sources/ANEFieldfareRuntime/TensorIndex.c" \
  "$project_root/Sources/ANEFieldfareRuntime/ANEProgram.m" \
  "$project_root/Sources/ANEFieldfareRuntime/ANEDynamicProjection.m" \
  "$project_root/Sources/ANEFieldfareQueueGate/main.m" \
  -o "$project_root/.build/ane-fieldfare-queue-gate"

xcrun clang -O3 -Wall -Wextra -fobjc-arc -fblocks \
  -framework Foundation -framework IOSurface -ldl \
  "$project_root/Sources/ANEFieldfareRuntime/TensorIndex.c" \
  "$project_root/Sources/ANEFieldfareRuntime/ANEProgram.m" \
  "$project_root/Sources/ANEFieldfareRuntime/ANEDynamicProjection.m" \
  "$project_root/Sources/ANEFieldfareAsyncGate/main.m" \
  -o "$project_root/.build/ane-fieldfare-async-gate"

xcrun clang -O3 -Wall -Wextra -fobjc-arc -fblocks \
  -framework Foundation -framework IOSurface -framework Metal -ldl \
  "$project_root/Sources/ANEFieldfareRuntime/TensorIndex.c" \
  "$project_root/Sources/ANEFieldfareRuntime/ANEProgram.m" \
  "$project_root/Sources/ANEFieldfareRuntime/ANEDynamicProjection.m" \
  "$project_root/Sources/ANEFieldfareGPUANETransferGate/main.m" \
  -o "$project_root/.build/ane-fieldfare-gpu-ane-transfer-gate"

xcrun clang -O3 -Wall -Wextra -fobjc-arc -fblocks \
  -framework Foundation -framework IOSurface -ldl \
  "$project_root/Sources/ANEFieldfareRuntime/ANEProgram.m" \
  "$project_root/Sources/ANEFieldfarePerfStatsGate/main.m" \
  -o "$project_root/.build/ane-fieldfare-perf-stats-gate"

xcrun clang -O3 -Wall -Wextra -fobjc-arc -fblocks \
  -framework Foundation -framework IOSurface -framework Metal -ldl \
  "$project_root/Sources/ANEFieldfareRuntime/TensorIndex.c" \
  "$project_root/Sources/ANEFieldfareRuntime/ANEProgram.m" \
  "$project_root/Sources/ANEFieldfareRuntime/ANEDynamicProjection.m" \
  "$project_root/Sources/ANEFieldfareANEGPUTransferGate/main.m" \
  -o "$project_root/.build/ane-fieldfare-ane-gpu-transfer-gate"

xcrun clang -O3 -Wall -Wextra -fobjc-arc -fblocks \
  -framework Foundation -framework IOSurface -framework Metal -ldl \
  "$project_root/Sources/ANEFieldfareRuntime/TensorIndex.c" \
  "$project_root/Sources/ANEFieldfareRuntime/ANEProgram.m" \
  "$project_root/Sources/ANEFieldfareRuntime/ANEDynamicProjection.m" \
  "$project_root/Sources/ANEFieldfareEventGate/main.m" \
  -o "$project_root/.build/ane-fieldfare-event-gate"

xcrun clang -O2 -Wall -Wextra -fobjc-arc -fblocks \
  -framework Foundation -framework IOSurface -framework Metal -ldl \
  "$project_root/Sources/ANEFieldfareRuntime/ANEProgram.m" \
  "$project_root/Sources/ANEFieldfareMetalANEEventGate/main.m" \
  -o "$project_root/.build/ane-fieldfare-metal-ane-event-gate"

xcrun clang -O2 -Wall -Wextra -fobjc-arc \
  -framework Foundation \
  -ldl \
  "$project_root/Sources/ANEFieldfareAPIGate/main.m" \
  -o "$project_root/.build/ane-fieldfare-api-gate"

xcrun clang -O2 -Wall -Wextra -fobjc-arc -fblocks \
  -framework Foundation -framework CoreML \
  "$project_root/Sources/ANEFieldfareStateGate/main.m" \
  -o "$project_root/.build/ane-fieldfare-state-gate"

xcrun clang -O3 -Wall -Wextra -fobjc-arc -fblocks \
  -framework Foundation -framework IOSurface -framework Metal -ldl \
  "$project_root/Sources/ANEFieldfareRuntime/TensorIndex.c" \
  "$project_root/Sources/ANEFieldfareRuntime/Dequant.c" \
  "$project_root/Sources/ANEFieldfareRuntime/Attention.c" \
  "$project_root/Sources/ANEFieldfareRuntime/ANEProgram.m" \
  "$project_root/Sources/ANEFieldfareRuntime/ANEResidentProjection.m" \
  "$project_root/Sources/ANEFieldfareRuntime/MetalAttention.m" \
  "$project_root/Sources/ANEFieldfareRuntime/MetalExpertDispatch.m" \
  "$project_root/Sources/ANEFieldfarePlacementCalibrationGate/main.m" \
  -o "$project_root/.build/ane-fieldfare-placement-calibration-gate"

xcrun clang -O2 -Wall -Wextra -fobjc-arc \
  -framework Foundation -framework IOSurface -ldl \
  "$project_root/Sources/ANEFieldfareMultiProcedureGate/main.m" \
  -o "$project_root/.build/ane-fieldfare-multi-procedure-gate"

xcrun clang -O2 -Wall -Wextra -fobjc-arc \
  -framework Foundation -ldl \
  "$project_root/Sources/ANEFieldfareSystemMultiProcedureGate/main.m" \
  -o "$project_root/.build/ane-fieldfare-system-multi-procedure-gate"

xcrun clang -O2 -Wall -Wextra -fobjc-arc \
  -framework Foundation \
  -framework CoreML \
  -framework IOSurface \
  -ldl \
  "$project_root/Sources/ANEFieldfareGate/main.m" \
  -o "$project_root/.build/ane-fieldfare-gate"

xcrun clang -O2 -Wall -Wextra -fobjc-arc \
  -framework Foundation \
  -framework IOSurface \
  -ldl \
  "$project_root/Sources/ANEFieldfareMutableGate/main.m" \
  -o "$project_root/.build/ane-fieldfare-mutable-gate"

xcrun clang -O2 -Wall -Wextra -fobjc-arc \
  -framework Foundation \
  "$project_root/Sources/ANEFieldfareRuntime/MemoryBudget.c" \
  "$project_root/Sources/ANEFieldfareMemoryBudgetGate/main.m" \
  -o "$project_root/.build/ane-fieldfare-memory-budget-gate"

xcrun clang -O2 -Wall -Wextra -fobjc-arc \
  -framework Foundation \
  "$project_root/Sources/ANEFieldfareRuntime/TensorIndex.c" \
  "$project_root/Sources/ANEFieldfareRouterGate/main.m" \
  -o "$project_root/.build/ane-fieldfare-router-gate"

xcrun clang -O3 -Wall -Wextra \
  "$project_root/Sources/ANEFieldfareRuntime/TensorIndex.c" \
  "$project_root/Sources/ANEFieldfareRuntime/RoutedLayer.c" \
  "$project_root/Sources/ANEFieldfareRoutedLayerGate/main.c" \
  -o "$project_root/.build/ane-fieldfare-routed-layer-gate"

xcrun clang -O2 -Wall -Wextra -fobjc-arc \
  -framework Foundation \
  "$project_root/Sources/ANEFieldfareRuntime/TensorIndex.c" \
  "$project_root/Sources/ANEFieldfareRuntime/Attention.c" \
  "$project_root/Sources/ANEFieldfareAttentionGate/main.m" \
  -o "$project_root/.build/ane-fieldfare-attention-gate"

xcrun clang -O3 -Wall -Wextra -fobjc-arc \
  -framework Foundation \
  "$project_root/Sources/ANEFieldfareRuntime/TensorIndex.c" \
  "$project_root/Sources/ANEFieldfareRuntime/Decode.c" \
  "$project_root/Sources/ANEFieldfareRuntime/Tokenizer.m" \
  "$project_root/Sources/ANEFieldfareTokenDecodeGate/main.m" \
  -o "$project_root/.build/ane-fieldfare-token-decode-gate"

xcrun clang -O3 -Wall -Wextra \
  "$project_root/Sources/ANEFieldfareRuntime/TensorIndex.c" \
  "$project_root/Sources/ANEFieldfareRuntime/Attention.c" \
  "$project_root/Sources/ANEFieldfareRuntime/RoutedLayer.c" \
  "$project_root/Sources/ANEFieldfareRuntime/Decoder.c" \
  "$project_root/Sources/ANEFieldfareDecoderGate/main.c" \
  -o "$project_root/.build/ane-fieldfare-decoder-gate"

xcrun clang -O2 -Wall -Wextra -fobjc-arc \
  -framework Foundation \
  -framework IOSurface \
  -ldl \
  "$project_root/Sources/ANEFieldfareRuntime/TensorIndex.c" \
  "$project_root/Sources/ANEFieldfareRuntime/ANEProgram.m" \
  "$project_root/Sources/ANEFieldfareRuntime/ANEDynamicProjection.m" \
  "$project_root/Sources/ANEFieldfareProjectionGate/main.m" \
  -o "$project_root/.build/ane-fieldfare-projection-gate"

xcrun clang -O3 -Wall -Wextra -fobjc-arc \
  -framework Foundation \
  -framework IOSurface \
  -ldl \
  "$project_root/Sources/ANEFieldfareCompactProjectionGate/main.m" \
  -o "$project_root/.build/ane-fieldfare-compact-projection-gate"

xcrun clang -O3 -Wall -Wextra -fobjc-arc -fblocks \
  -framework Foundation \
  -framework IOSurface \
  -framework Metal \
  -ldl \
  "$project_root/Sources/ANEFieldfareRuntime/TensorIndex.c" \
  "$project_root/Sources/ANEFieldfareRuntime/RoutedLayer.c" \
  "$project_root/Sources/ANEFieldfareCompactExpertGate/main.m" \
  -o "$project_root/.build/ane-fieldfare-compact-expert-gate"

xcrun clang -O3 -Wall -Wextra -fobjc-arc \
  -framework Foundation \
  -framework IOSurface \
  -framework Metal \
  -ldl \
  "$project_root/Sources/ANEFieldfareRuntime/TensorIndex.c" \
  "$project_root/Sources/ANEFieldfareRuntime/RoutedLayer.c" \
  "$project_root/Sources/ANEFieldfareRuntime/ANEProgram.m" \
  "$project_root/Sources/ANEFieldfareRuntime/ANEExpertDispatch.m" \
  "$project_root/Sources/ANEFieldfareANEExpertDispatchGate/main.m" \
  -o "$project_root/.build/ane-fieldfare-ane-expert-dispatch-gate"

xcrun clang -O3 -Wall -Wextra -fobjc-arc -fblocks \
  -framework Foundation \
  -framework IOSurface \
  -ldl \
  "$project_root/Sources/ANEFieldfareRuntime/TensorIndex.c" \
  "$project_root/Sources/ANEFieldfareRuntime/Decode.c" \
  "$project_root/Sources/ANEFieldfareRuntime/ANEProgram.m" \
  "$project_root/Sources/ANEFieldfareRuntime/ANETiledHead.m" \
  "$project_root/Sources/ANEFieldfareANEHeadGate/main.m" \
  -o "$project_root/.build/ane-fieldfare-ane-head-gate"

xcrun clang -O3 -Wall -Wextra -fobjc-arc \
  -framework Foundation \
  -framework IOSurface \
  -framework Metal \
  -ldl \
  "$project_root/Sources/ANEFieldfareRuntime/TensorIndex.c" \
  "$project_root/Sources/ANEFieldfareRuntime/Attention.c" \
  "$project_root/Sources/ANEFieldfareRuntime/RoutedLayer.c" \
  "$project_root/Sources/ANEFieldfareRuntime/Decoder.c" \
  "$project_root/Sources/ANEFieldfareRuntime/Decode.c" \
  "$project_root/Sources/ANEFieldfareRuntime/Tokenizer.m" \
  "$project_root/Sources/ANEFieldfareRuntime/ANEProgram.m" \
  "$project_root/Sources/ANEFieldfareRuntime/ANEExpertDispatch.m" \
  "$project_root/Sources/ANEFieldfareRuntime/AOTExpertRuntime.c" \
  "$project_root/Sources/ANEFieldfareRuntime/AOTExpertArtifactProvider.m" \
  "$project_root/Sources/ANEFieldfareRuntime/MetalExpertDispatch.m" \
  "$project_root/Sources/ANEFieldfareRuntime/MetalAttention.m" \
  "$project_root/Sources/ANEFieldfareRuntime/ANEResidentProjection.m" \
  "$project_root/Sources/ANEFieldfareRuntime/InferenceOptimizer.m" \
  "$project_root/Sources/ANEFieldfareRuntime/Dequant.c" \
  "$project_root/Sources/ANEFieldfareGenerate/main.m" \
  -o "$project_root/.build/ane-fieldfare-generate"

# Production AOT expert integration seam. The gate uses a fake compiler-artifact
# provider to validate identity lookup and the 16-bank completion-generation
# lifecycle; it makes no ANE execution claim.
xcrun clang -O3 -Wall -Wextra \
  "$project_root/Sources/ANEFieldfareRuntime/AOTExpertRuntime.c" \
  "$project_root/Sources/ANEFieldfareAOTExpertRuntimeGate/main.c" \
  -o "$project_root/.build/ane-fieldfare-aot-expert-runtime-gate"

xcrun clang -O3 -Wall -Wextra -fobjc-arc -fblocks \
  -framework Foundation -framework IOSurface -ldl \
  "$project_root/Sources/ANEFieldfareRuntime/TensorIndex.c" \
  "$project_root/Sources/ANEFieldfareRuntime/ANEProgram.m" \
  "$project_root/Sources/ANEFieldfareRuntime/AOTExpertRuntime.c" \
  "$project_root/Sources/ANEFieldfareRuntime/AOTExpertArtifactProvider.m" \
  "$project_root/Sources/ANEFieldfareAOTExpertProviderGate/main.m" \
  -o "$project_root/.build/ane-fieldfare-aot-expert-provider-gate"

# One loaded many-procedure artifact, many compiler-derived request objects,
# and only the original eight carrier pairs. The live gate supplies the
# authentic artifact/checkpoint and retains exact poison-controlled oracles.
xcrun clang -O3 -Wall -Wextra -fobjc-arc -fblocks \
  -framework Foundation -framework IOSurface -ldl \
  "$project_root/Sources/ANEFieldfareRuntime/TensorIndex.c" \
  "$project_root/Sources/ANEFieldfareRuntime/ANEProgram.m" \
  "$project_root/Sources/ANEFieldfareAOTSharedSurfaceSelectionGate/main.m" \
  -o "$project_root/.build/ane-fieldfare-aot-shared-surface-selection-gate"

# Fixed-graph weight-state transport gate. It validates that one compiled
# eight-procedure ANE program can move between authentic compact expert states
# without recompilation while preserving an independent numerical oracle.
xcrun clang -O3 -Wall -Wextra -fobjc-arc -fblocks \
  -framework Foundation -framework IOSurface -ldl \
  "$project_root/Sources/ANEFieldfareRuntime/TensorIndex.c" \
  "$project_root/Sources/ANEFieldfareAOTLayerProcedureGate/weight_transport.m" \
  -o "$project_root/.build/ane-fieldfare-aot-weight-state-transport-gate"

xcrun clang -O2 -Wall -Wextra -fobjc-arc \
  -framework Foundation \
  "$project_root/Sources/ANEFieldfareRuntime/InferenceOptimizer.m" \
  "$project_root/Sources/ANEFieldfareInferenceOptimizerGate/main.m" \
  -o "$project_root/.build/ane-fieldfare-inference-optimizer-gate"

xcrun clang -O3 -Wall -Wextra -fobjc-arc \
  -framework Foundation -framework Metal -framework IOSurface \
  "$project_root/Sources/ANEFieldfareRuntime/TensorIndex.c" \
  "$project_root/Sources/ANEFieldfareRuntime/Attention.c" \
  "$project_root/Sources/ANEFieldfareRuntime/MetalAttention.m" \
  "$project_root/Sources/ANEFieldfareMetalAttentionGate/main.m" \
  -o "$project_root/.build/ane-fieldfare-metal-attention-gate"

xcrun clang -O3 -Wall -Wextra -fobjc-arc \
  -framework Foundation -framework IOSurface -ldl \
  "$project_root/Sources/ANEFieldfareRuntime/TensorIndex.c" \
  "$project_root/Sources/ANEFieldfareRuntime/Attention.c" \
  "$project_root/Sources/ANEFieldfareRuntime/RoutedLayer.c" \
  "$project_root/Sources/ANEFieldfareRuntime/Decoder.c" \
  "$project_root/Sources/ANEFieldfareRuntime/Decode.c" \
  "$project_root/Sources/ANEFieldfareRuntime/Tokenizer.m" \
  "$project_root/Sources/ANEFieldfareRuntime/ANEProgram.m" \
  "$project_root/Sources/ANEFieldfareRuntime/ANEExpertDispatch.m" \
  "$project_root/Sources/ANEFieldfarePrefillBatchedGate/main.m" \
  -o "$project_root/.build/ane-fieldfare-prefill-batched-gate"

xcrun clang -O2 -Wall -Wextra -fobjc-arc \
  -framework Foundation \
  -framework IOSurface \
  -ldl \
  "$project_root/Sources/ANEFieldfareRuntime/MemoryBudget.c" \
  "$project_root/Sources/ANEFieldfareExpertGate/main.m" \
  -o "$project_root/.build/ane-fieldfare-expert-gate"

# Explicitly scoped production build. The default still requires the complete
# historical suite and fails if its recovered sources are missing.
if (( core_only )); then
  build_hotpath
  echo "Core build complete; historical experiment suites were not requested."
  exit 0
fi

xcrun clang -O3 -Wall -Wextra -fobjc-arc -fblocks \
  -framework Foundation \
  -framework IOSurface \
  -ldl \
  "$project_root/Sources/ANEFieldfareRuntime/MemoryBudget.c" \
  "$project_root/Sources/ANEFieldfareRuntime/ANEProgram.m" \
  "$project_root/Sources/ANEFieldfareConstantWeightGate/main.m" \
  -o "$project_root/.build/ane-fieldfare-constant-weight-gate"

# Hybrid companion rail: compact INT4 routed experts stay compact in unified
# memory and execute on Metal; the complete layer gate retains the CPU oracle.
xcrun clang -O3 -Wall -Wextra -fobjc-arc -fblocks \
  -framework Foundation -framework Metal \
  "$project_root/Sources/ANEFieldfareRuntime/TensorIndex.c" \
  "$project_root/Sources/ANEFieldfareRuntime/Dequant.c" \
  "$project_root/Sources/ANEFieldfareRuntime/RoutedLayer.c" \
  "$project_root/Sources/ANEFieldfareRuntime/MetalExpertDispatch.m" \
  "$project_root/Sources/ANEFieldfareMetalExpertGate/main.m" \
  -o "$project_root/.build/ane-fieldfare-metal-expert-gate"

# Position-batched greedy head used by training-free speculative verification.
xcrun clang -O3 -Wall -Wextra -fobjc-arc -fblocks \
  -framework Foundation -framework Metal \
  "$project_root/Sources/ANEFieldfareRuntime/TensorIndex.c" \
  "$project_root/Sources/ANEFieldfareRuntime/Dequant.c" \
  "$project_root/Sources/ANEFieldfareRuntime/RoutedLayer.c" \
  "$project_root/Sources/ANEFieldfareRuntime/MetalExpertDispatch.m" \
  "$project_root/Sources/ANEFieldfareBatchedHeadGate/main.m" \
  -o "$project_root/.build/ane-fieldfare-batched-head-gate"

# Canonical benchmark suites. Gate mains are compiled as renamed functions and
# linked into two real multicall binaries; no suite subcommand shells out to a
# legacy executable. Historical receipts retain their original exact commands.
suite_object_root="$project_root/.build/benchmark-suite-objects"
rm -rf "$suite_object_root"
mkdir -p "$suite_object_root/private" "$suite_object_root/transport"

compile_objc_gate() {
  local suite="$1" source_dir="$2" symbol="$3"
  xcrun clang -O3 -Wall -Wextra -fobjc-arc -fblocks -Dmain="$symbol" -c \
    "$project_root/Sources/$source_dir/main.m" \
    -o "$suite_object_root/$suite/$symbol.o"
}

compile_c_gate() {
  local suite="$1" source_dir="$2" symbol="$3"
  xcrun clang -O3 -Wall -Wextra -Dmain="$symbol" -c \
    "$project_root/Sources/$source_dir/main.c" \
    -o "$suite_object_root/$suite/$symbol.o"
}

compile_objc_gate private ANEFieldfareCompressedConstantGate ANEFieldfareCompressedConstantMain
compile_objc_gate private ANEFieldfareCompressionSpellingProbe ANEFieldfareCompressionSpellingMain
compile_objc_gate private ANEFieldfareNativeDtypeProbe ANEFieldfareNativeDtypeProbeMain
compile_objc_gate private ANEFieldfareNativeDtypeGate ANEFieldfareNativeDtypeMain
compile_objc_gate private ANEFieldfareProgramResidencyGate ANEFieldfareProgramResidencyMain
compile_objc_gate private ANEFieldfareResidentProjectionGate ANEFieldfareResidentProjectionMain
compile_objc_gate private ANEFieldfarePrivateSurfaceGate ANEFieldfarePrivateSurfaceMain
compile_objc_gate private ANEFieldfareSystemCompressedModelGate ANEFieldfareSystemCompressedModelMain
compile_objc_gate private ANEFieldfareCompressedComputePlanGate ANEFieldfareCompressedComputePlanMain
compile_objc_gate private ANEFieldfareCoreMLCompileGate ANEFieldfareCoreMLCompileMain
compile_objc_gate private ANEFieldfareAOTCompressedProjectionGate ANEFieldfareAOTCompressedProjectionMain
compile_objc_gate private ANEFieldfareAOTResidencyContentionGate ANEFieldfareAOTResidencyContentionMain

xcrun clang -O3 -Wall -Wextra -fobjc-arc -fblocks \
  -DANE_FIELDFARE_PRIVATE_SUITE \
  -framework Foundation -framework CoreML -framework IOSurface -ldl \
  "$project_root/Sources/ANEFieldfareRuntime/TensorIndex.c" \
  "$project_root/Sources/ANEFieldfareRuntime/Dequant.c" \
  "$project_root/Sources/ANEFieldfareRuntime/ANEProgram.m" \
  "$project_root/Sources/ANEFieldfareBenchmarkSuite/main.c" \
  "$suite_object_root/private/"*.o \
  -o "$project_root/.build/ane-fieldfare-private-benchmarks"

compile_c_gate transport ANEFieldfareDequantGate ANEFieldfareDequantMain
compile_objc_gate transport ANEFieldfareCarrierWriteProbe ANEFieldfareCarrierWriteMain
compile_objc_gate transport ANEFieldfarePrepBreakdownGate ANEFieldfarePrepBreakdownMain
compile_objc_gate transport ANEFieldfarePositionLaneGate ANEFieldfarePositionLaneMain
compile_objc_gate transport ANEFieldfareTransportCeilingGate ANEFieldfareTransportCeilingMain
compile_objc_gate transport ANEFieldfareExpertLayoutGate ANEFieldfareExpertLayoutMain
compile_objc_gate transport ANEFieldfareExplicitReadGate ANEFieldfareExplicitReadMain
compile_objc_gate transport ANEFieldfareExpertResidencyGate ANEFieldfareExpertResidencyMain
compile_objc_gate transport ANEFieldfareSustainedRampProbe ANEFieldfareSustainedRampMain

xcrun clang -O3 -Wall -Wextra -fobjc-arc -fblocks \
  -DANE_FIELDFARE_TRANSPORT_SUITE \
  -framework Foundation -framework IOSurface -ldl \
  "$project_root/Sources/ANEFieldfareRuntime/TensorIndex.c" \
  "$project_root/Sources/ANEFieldfareRuntime/Dequant.c" \
  "$project_root/Sources/ANEFieldfareRuntime/RoutedLayer.c" \
  "$project_root/Sources/ANEFieldfareRuntime/ANEProgram.m" \
  "$project_root/Sources/ANEFieldfareRuntime/ANEExpertDispatch.m" \
  "$project_root/Sources/ANEFieldfareBenchmarkSuite/main.c" \
  "$suite_object_root/transport/"*.o \
  -o "$project_root/.build/ane-fieldfare-transport-benchmarks"

xcrun clang -O3 -Wall -Wextra -fobjc-arc \
  -framework Foundation \
  -framework IOSurface \
  -ldl \
  "$project_root/Sources/ANEFieldfareRuntime/TensorIndex.c" \
  "$project_root/Sources/ANEFieldfareRuntime/Attention.c" \
  "$project_root/Sources/ANEFieldfareRuntime/RoutedLayer.c" \
  "$project_root/Sources/ANEFieldfareRuntime/Decoder.c" \
  "$project_root/Sources/ANEFieldfareRuntime/Decode.c" \
  "$project_root/Sources/ANEFieldfareRuntime/Tokenizer.m" \
  "$project_root/Sources/ANEFieldfareRuntime/Dequant.c" \
  "$project_root/Sources/ANEFieldfareDeltaCapture/main.m" \
  -o "$project_root/.build/ane-fieldfare-delta-capture"

# Owned compact expert-streaming runtime.  This is a first-class product, not
# a benchmark invocation of the read-only Turbo reference.
build_hotpath
