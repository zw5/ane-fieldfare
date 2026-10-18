#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
"$project_root/Scripts/build.sh" --historical

aot_runtime_receipt="$($project_root/.build/ane-fieldfare-aot-expert-runtime-gate)"
jq -e '
  .schema == "ane-fieldfare.aot-expert-runtime.v1" and
  .status == "validated" and .request_banks == 16 and .top_k == 8 and
  .bank_bases == [0,8,0] and .completion_generations == [1,1,2] and
  .procedure_loads == 8 and .procedure_cache_hits == 16 and
  .submissions == 24 and .completions == 24 and
  .duplicate_top8_rejected_before_submit and .eight_bank_shape_rejected and
  .missing_provider_rejected and .unloads == 8 and .context_destroys == 1
' <<<"$aot_runtime_receipt" >/dev/null

aot_fixture_artifact="$project_root/.build/aot-block-slice/eight-experts.mlmodelc"
aot_fixture_checkpoint="$project_root/checkpoints/gturbo/gemma4.gturbo"
if [[ -f "$aot_fixture_artifact/model.mil" &&
      -f "$aot_fixture_checkpoint/packed_experts/layout.json" ]]; then
  aot_fixture_root="$project_root/.build/aot-expert-provider-fixture"
  mkdir -p "$aot_fixture_root"
  ln -sfn "$aot_fixture_artifact" "$aot_fixture_root/layer-00.mlmodelc"
  aot_provider_receipt="$($project_root/.build/ane-fieldfare-aot-expert-provider-gate \
    "$aot_fixture_root" "$aot_fixture_checkpoint")"
  jq -e '
    .schema == "ane-fieldfare.aot-expert-provider.v1" and
    .status == "validated" and .compile_count == 1 and .load_count == 1 and
    .loaded_layer_handles == 1 and .compiler_named_procedures == 8 and
    .logical_request_banks == 16 and .physical_ane_requests == 8 and
    .a_b_bank_bases == [0,8] and (.production_double_buffering | not) and
    .submissions == 16 and .completions == 16 and
    .comparisons == 22528 and .finite == 22528 and
    .maximum_absolute_error <= .tolerance and
    .unavailable_expert_rejected_before_submit
  ' <<<"$aot_provider_receipt" >/dev/null

  aot_transport_state_b="$project_root/.build/aot-layer-procedures/experts-8-15.mlmodelc"
  if [[ -f "$aot_transport_state_b/weights/weight.bin" ]]; then
    aot_transport_receipt="$($project_root/Scripts/with_benchmark_lock.sh \
      "$project_root/.build/ane-fieldfare-aot-weight-state-transport-gate" \
      "$aot_fixture_artifact" "$aot_transport_state_b" "$aot_fixture_checkpoint")"
    jq -e '
      .schema == "ane-fieldfare.aot-weight-state-transport.v1" and
      .status == "validated" and .exit_status == 0 and
      .host_chip != "unknown" and .os_build != "unknown" and
      .ane_architecture == "H17G" and
      .lifecycle.compile_count == 1 and
      .lifecycle.initial_load_count == 1 and
      .lifecycle.weight_state_transports == 2 and
      .lifecycle.recompile_count_during_transport == 0 and
      (.lifecycle.states | map(.state)) == ["A", "B", "A"] and
      (.lifecycle.states | all(.oracle)) and
      .numerical_result.comparisons == 4325376 and
      .numerical_result.finite == .numerical_result.comparisons and
      .numerical_result.max_absolute_error <= .numerical_result.tolerance
    ' <<<"$aot_transport_receipt" >/dev/null
  fi
fi
PYTHONPYCACHEPREFIX="$project_root/.build/pycache" \
  python3 -m py_compile "$project_root/Scripts/build_aot_compressed_projection.py" \
    "$project_root/Scripts/build_multifunction_probe.py" \
    "$project_root/Scripts/evaluate_prompt_lookup.py" \
    "$project_root/Analysis/execution_field_continuation.py" \
    "$project_root/Sources/ANEFieldfareStateGate/build_state_model.py"

state_python="${ANE_FIELDFARE_COREML_PYTHON:-$project_root/.build/coremltools-venv/bin/python}"
[[ -x "$state_python" ]] || state_python=python3
"$state_python" -c 'import coremltools' || {
  echo "Core ML state gate requires coremltools; set ANE_FIELDFARE_COREML_PYTHON" >&2
  exit 1
}
"$state_python" "$project_root/Sources/ANEFieldfareStateGate/build_state_model.py" >/dev/null
state_receipt="$($project_root/.build/ane-fieldfare-state-gate \
  "$project_root/.build/state-recurrence/state-recurrence.mlpackage" || true)"
jq -e '
  .schema == "ane-fieldfare.state-recurrence.v1" and
  .state_lifecycle.status == "validated" and
  .state_readback_matches_oracle == true and
  .max_abs_error == 0 and
  .compute_plan.loaded == 1 and
  (.ane_placement.supported_by_compute_plan == true or
   .ane_placement.supported_by_compute_plan == 1) and
  ((.status == "validated" and .ane_placement.status == "validated") or
   (.status == "rejected" and .stage == "compute_plan_ane_assignment" and
    .ane_placement.status == "rejected"))
' <<<"$state_receipt" >/dev/null
jq -e '
  .schema == "ane-fieldfare.ane-decode-admission.v1" and
  .status == "rejected" and
  .exact_token_oracle.preserved_across_all_runs and
  (.paired_result.enabled_over_disabled < .paired_result.promotion_threshold) and
  (.paired_result.threshold_cleared | not) and
  (.paired_result.cache_hit_miss_regression | not) and
  (.paired_result.disk_regression | not) and
  (.paired_result.physical_footprint_regression | not)
' "$project_root/results/ane-decode-admission-m5-macos27.json" >/dev/null

metal_ane_event="$($project_root/.build/ane-fieldfare-metal-ane-event-gate)"
jq -e '
  .schema == "ane-fieldfare.metal-ane-event.v1" and
  .status == "validated" and
  .stage == "metal_signal_to_ane_wait_oracle" and
  (.abi_checks | all(.matches)) and
  .objects.iosurface_shared_event_same_identity and
  .request.before_shared_events_nil and
  .request.after_contains_runtime_wait_event and
  .guards.input_4k_aligned_and_length_multiple and
  .guards.metal_buffer_contents_matches_ane_input and
  .lifecycle.ane_compile_count == 1 and .lifecycle.ane_load_count == 1 and
  (.ordering.host_semaphore_between_metal_and_ane | not) and
  (.ordering.host_wait_before_release | not) and
  (.runs | length == 3) and
  (.runs | all(.request_configured and .ane_begin_accepted and
               .negative_control.passed and
               .negative_control.ane_in_flight_before_release == 1 and
               .negative_control.output_poison_values_before_release == 4096 and
               .host_released_only_metal_blocker and .metal_completed and
               .ane_completed_after_event and .oracle.valid and
               .oracle.input_mismatches == 0 and
               .oracle.output_mismatches == 0 and
               .oracle.finite_outputs == 4096 and
               .oracle.max_absolute_error == 0))
' <<<"$metal_ane_event" >/dev/null

# A nested acquisition must fail and must not alter the outer owner's token.
# The full suite may itself be run under the canonical helper; otherwise create
# a temporary outer owner solely for this control.
check_nested_lock_rejection='set -e
  before="$(<"$1/.build/ane-fieldfare-benchmark.lock/owner")"
  if "$1/Scripts/with_benchmark_lock.sh" true >/dev/null 2>&1; then
    echo "nested benchmark lock unexpectedly succeeded" >&2
    exit 1
  fi
  after="$(<"$1/.build/ane-fieldfare-benchmark.lock/owner")"
  [[ "$before" == "$after" ]]'
if [[ -r "$project_root/.build/ane-fieldfare-benchmark.lock/owner" ]]; then
  bash -c "$check_nested_lock_rejection" benchmark-lock-test "$project_root"
else
  "$project_root/Scripts/with_benchmark_lock.sh" \
    bash -c "$check_nested_lock_rejection" benchmark-lock-test "$project_root"
fi
"$project_root/.build/ane-fieldfare-api-gate"
optimizer_receipt="$($project_root/.build/ane-fieldfare-inference-optimizer-gate \
  "$project_root/checkpoints/gturbo/gemma4.gturbo")"
jq -e '
  .schema == "ane-fieldfare.inference-optimizer-gate.v1" and
  .status == "validated" and
  .architecture.hiddenSize == 2816 and
  .architecture.numExperts == 128 and
  .architecture.topKExperts == 8 and
  .candidate.generation.training_required == false and
  .candidate.generation.model_compilation_required == false and
  .candidate.schedule.group_routes_by_weight_identity == true and
  .candidate.schedule.batch_shared_ffn == true and
  .candidate.schedule.device_attention == true and
  .synthetic_architectures_checked == 2 and
  .cross_shape_rules_validated == true
' <<<"$optimizer_receipt" >/dev/null
metal_attention="$($project_root/.build/ane-fieldfare-metal-attention-gate \
  "$project_root/Sources/ANEFieldfareRuntime/Metal/attention.metal")"
jq -e '
  .schema == "ane-fieldfare.metal-attention.v1" and
  .status == "validated" and
  (.tiles | length == 2) and
  [.tiles[].positions] == [64,32] and
  (.tiles | all(.valid and .combined_oracle.tolerance_failures == 0 and
                .key_cache_oracle.tolerance_failures == 0 and
                .value_cache_oracle.tolerance_failures == 0)) and
  .metal.dispatches == 2 and .metal.positions == 96
' <<<"$metal_attention" >/dev/null
runtime_metadata="$($project_root/.build/ane-fieldfare-runtime-metadata-probe)"
jq -e '
  .schema == "ane-fieldfare.runtime-metadata.v2" and
  .status == "present" and
  (.framework_loaded == true or .framework_loaded == 1) and
  .framework_class_count >= 38 and
  ([.framework_class_names[]] | index("_ANESharedEvents") != null) and
  ([.framework_class_names[]] | index("_ANEIntermediateTensor") != null) and
  ([.framework_class_names[]] | index("_ANEPerformanceStatsIOSurface") != null) and
  ([.discovered_relevant_classes[].class] | index("_ANEVirtualClient") != null)
' <<<"$runtime_metadata" >/dev/null
intermediate_tensor_receipt="$($project_root/.build/ane-fieldfare-intermediate-tensor-gate)"
jq -e '
  def truthy: . == true or . == 1;
  def falsey: . == false or . == 0;
  .schema == "ane-fieldfare.intermediate-tensor.v1" and
  .status == "validated_execution_capture_rejected" and
  (.framework_loaded | truthy) and
  .compile_count == 1 and .load_count == 1 and
  (.host_descriptor_control.constructed | truthy) and
  (.host_descriptor_control.round_trip | truthy) and
  .negative_controls.nil_model_query.result == "nil" and
  .negative_controls.missing_model_query.result == "nil" and
  .negative_controls.nil_surface_normalization.accepted == false and
  .negative_controls.host_descriptor_normalization.accepted == true and
  (.negative_controls.host_descriptor_normalization.bytes_changed | falsey) and
  (.compiler_program.compiled | truthy) and (.compiler_program.loaded | truthy) and
  (.compiler_program.compiler_artifacts.contains_internal_name | truthy) and
  (.compiler_program.compiler_artifacts.anedebug_files | length) == 0 and
  (.compiler_program.any_descriptors | falsey) and
  (.compiler_program.exact_named_descriptor | falsey) and
  (.execution_oracle.passed | truthy) and
  .execution_oracle.final_mismatches == 0 and
  .execution_oracle.poison_remaining == 0 and
  .execution_oracle.final_values_differing_from_expected_intermediate == 63 and
  (.execution_oracle.intermediate_observed | falsey)
' <<<"$intermediate_tensor_receipt" >/dev/null
"$project_root/.build/ane-fieldfare-multi-procedure-gate"
offset_surface_receipt="$(mktemp -t ane-fieldfare-offset-surface.XXXXXX)"
if ANE_FIELDFARE_OFFSET_ARENA=1 \
    "$project_root/.build/ane-fieldfare-multi-procedure-gate" >"$offset_surface_receipt"; then
  echo "nonzero startOffset unexpectedly passed its numerical oracle" >&2
  rm -f "$offset_surface_receipt"
  exit 1
fi
jq -e '
  .status == "rejected" and
  .offset_arena_enabled == true and
  .map_count == 3 and .unmap_count == 3 and
  ([.runs[].request_valid] | all) and
  ([.runs[].mapper_valid] | all) and
  ([.runs[].mapped] | all) and
  ([.runs[].evaluated] | all) and
  ([.runs[].mismatches] == [63, 63, 63]) and
  ([.runs[].base_mismatches_for_alternate_input] == [0, 0, 0])
' "$offset_surface_receipt" >/dev/null
rm -f "$offset_surface_receipt"
"$project_root/.build/ane-fieldfare-gate"
"$project_root/Scripts/test_mutable_constructions.sh"
"$project_root/.build/ane-fieldfare-memory-budget-gate"
"$project_root/.build/ane-fieldfare-compact-projection-gate"
"$project_root/.build/ane-fieldfare-constant-weight-gate"
"$project_root/.build/ane-fieldfare-transport-benchmarks" dequant
"$project_root/.build/ane-fieldfare-transport-benchmarks" carrier-write

# The suites expose a stable registry even when a normal test run does not
# mutate ANE residency, evict UBC pages, or compile hundreds of MB of constants.
canonical_benchmark_binaries=(
  ane-fieldfare-private-benchmarks
  ane-fieldfare-transport-benchmarks
  ane-fieldfare-prefill-batched-gate
)
for binary in "${canonical_benchmark_binaries[@]}"; do
  [[ -x "$project_root/.build/$binary" ]] || {
    echo "missing canonical benchmark binary: $binary" >&2
    exit 1
  }
done

obsolete_benchmark_binaries=(
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
for binary in "${obsolete_benchmark_binaries[@]}"; do
  [[ ! -e "$project_root/.build/$binary" ]] || {
    echo "obsolete one-command benchmark survived canonical build: $binary" >&2
    exit 1
  }
done

expected_private=$'compressed-constant\topt-in\ncompression-spelling\topt-in\nnative-dtype-probe\topt-in\nnative-dtype\topt-in\nprogram-residency\topt-in\nresident-projection\topt-in\nprivate-surface\tsafe\nsystem-compressed-model\tsafe\ncompressed-compute-plan\tsafe\ncoreml-compile\topt-in\naot-compressed-projection\topt-in\naot-residency-contention\topt-in'
expected_transport=$'dequant\tsafe\ncarrier-write\tsafe\nprep-breakdown\tsafe\nposition-lane\tsafe\ntransport-ceiling\topt-in\nexpert-layout\tsafe\nexplicit-read\topt-in\nexpert-residency\topt-in\nsustained-ramp\tsafe'
[[ "$("$project_root/.build/ane-fieldfare-private-benchmarks" --list)" == "$expected_private" ]] || {
  echo "private benchmark registry mismatch" >&2
  exit 1
}
[[ "$("$project_root/.build/ane-fieldfare-transport-benchmarks" --list)" == "$expected_transport" ]] || {
  echo "transport benchmark registry mismatch" >&2
  exit 1
}
if "$project_root/.build/ane-fieldfare-transport-benchmarks" not-a-command >/dev/null 2>&1; then
  echo "unknown benchmark subcommand negative control unexpectedly succeeded" >&2
  exit 1
fi

model_dir="${ANE_FIELDFARE_GTURBO_MODEL:-$project_root/checkpoints/gturbo/gemma4.gturbo}"
if [[ -f "$model_dir/manifest.json" && -f "$model_dir/model_weights.bin" ]]; then
  hotpath_bin="$project_root/.build/swift-hotpath/release/ANEFieldfareHotpathCLI"
  hotpath_fixture="$project_root/Sources/ANEFieldfareHotpathCLI/Fixtures/capital-chat.json"
  hotpath_control="$($hotpath_bin --model "$model_dir" \
    --messages-file "$hotpath_fixture" --max-new 16 --temperature 0 --top-k 0 2>/dev/null)"
  [[ "$hotpath_control" == 'The capital of France is **Paris**.' ]]
  jacobi_control="$(ANE_FIELDFARE_JACOBI_WIDTH=4 ANE_FIELDFARE_JACOBI_ITERATIONS=4 \
    "$hotpath_bin" --model "$model_dir" --messages-file "$hotpath_fixture" \
      --max-new 16 --temperature 0 --top-k 0 2>/dev/null)"
  [[ "$jacobi_control" == "$hotpath_control" ]]
  self_draft_control="$(ANE_FIELDFARE_SELF_DRAFT_LAYERS=15 ANE_FIELDFARE_SELF_DRAFT_WIDTH=4 \
    "$hotpath_bin" --model "$model_dir" --messages-file "$hotpath_fixture" \
      --max-new 16 --temperature 0 --top-k 0 2>/dev/null)"
  [[ "$self_draft_control" == "$hotpath_control" ]]

  perf_stats_gate="$($project_root/.build/ane-fieldfare-perf-stats-gate)"
  jq -e '
    .status == "validated_execution_stats_schema_unavailable" and
    (.abi_inventory | all(.present)) and
    (.shapes | length == 2) and
    (.shapes | all(
      .status == "validated_execution" and
      .requested_mask == 1 and
      .mask_readback_after_load == 1 and
      .repeated_work.evaluation_count == 17 and
      ([.repeated_work.runs[].oracle.valid] | all) and
      ([.repeated_work.runs[].perfStatsArray.items[0].raw_surface.nonzero_bytes] | all(. == 0)) and
      .reset_control.set_mask_zero_accepted and
      .reset_control.mask_readback == 0 and
      .negative_control.invalid_bank_returns_null
    ))
  ' <<<"$perf_stats_gate" >/dev/null
  placement_calibration="$($project_root/.build/ane-fieldfare-placement-calibration-gate "$model_dir")"
  # Batch-1 is intentionally not an admission requirement: repeated live runs
  # cross the 1.05 boundary in both directions, and the end-to-end decode gate
  # rejects promotion. Prefill widths 8/18/64 must retain a material win.
  jq -e '
    def truthy: . == true or . == 1;
    .status == "validated" and
    .placement_changed == false and
    [.arms[].positions] == [1,8,18,64] and
    (.arms | all(.max_abs_cross_backend_error < 0.02)) and
    ([.arms[] | select(.positions >= 8)] |
      all((.ane_wins | truthy) and .ane_speedup_over_metal > 1.05))
  ' <<<"$placement_calibration" >/dev/null
  gpu_ane_transfer_gate="$($project_root/.build/ane-fieldfare-gpu-ane-transfer-gate "$model_dir")"
  jq -e '
    .status == "validated" and
    .lifecycle.ane_compile_count == 1 and
    .lifecycle.ane_load_count == 1 and
    .lifecycle.metal_library_compile_count == 1 and
    .lifecycle.metal_pipeline_load_count == 1 and
    .direct_iosurface_backed_metal_buffer_supported and
    (.arms | length == 3) and
    (.arms | all(.repetitions == 8 and .warm_repetitions == 7 and .all_runs_valid)) and
    (.arms[1].explicit_transfer_bytes == .shape.carrier_bytes) and
    (.arms[2].explicit_transfer_bytes == 0) and
    (.arms | all([.runs[].oracle.poison_values_remaining] | add == 0))
  ' <<<"$gpu_ane_transfer_gate" >/dev/null
  ane_gpu_transfer_gate="$($project_root/.build/ane-fieldfare-ane-gpu-transfer-gate "$model_dir")"
  jq -e '
    .status == "validated" and
    .lifecycle.ane_compile_count == 1 and
    .lifecycle.ane_load_count == 1 and
    .lifecycle.metal_library_compile_count == 1 and
    .lifecycle.metal_pipeline_load_count == 1 and
    .guards.base_nonnull and
    .guards.page_alignment and
    .guards.buffer_contents_identity and
    .guards.surface_and_program_outlive_metal_buffers and
    (.arms | length == 3) and
    (.arms | all(.warm_repetitions == 7 and .all_runs_valid)) and
    .arms[0].cpu_memcpy_bytes == .shape.output_bytes and
    .arms[1].explicit_blit_bytes == .shape.output_bytes and
    .arms[2].direct_shared_bytes == .shape.output_bytes and
    .arms[2].cpu_memcpy_bytes == 0 and
    .arms[2].explicit_blit_bytes == 0 and
    (.arms | all([.runs[].oracle.poison_values_remaining] | add == 0)) and
    (.synchronization.private_shared_event_transaction.executed | not)
  ' <<<"$ane_gpu_transfer_gate" >/dev/null
  event_gate="$($project_root/.build/ane-fieldfare-event-gate "$model_dir")"
  jq -e '
    .schema == "ane-fieldfare.event-transaction.v1" and
    .status == "validated" and
    .stage == "ane_signal_to_metal_wait_oracle" and
    (.abi_checks | all(.matches)) and
    .objects.same_object_identity and
    .guards.output_base_nonnull and
    .guards.output_page_aligned and
    .guards.metal_buffer_contents_identity and
    .negative_control.blocked_and_preserved_poison and
    .shared_event.request_attachment == "validated" and
    .shared_event.ane_signal_to_metal_wait_ordering == "validated" and
    (.shared_event.host_semaphore_between_ane_and_metal | not) and
    .transaction.status == "present" and
    (.transaction.nonzero_handle_constructed | not) and
    (.runs | length == 8) and
    (.runs | all(.negative_control_blocked and .ane_begin_accepted and
                 .metal_completed_after_ane_signal and .valid and
                 .oracle.valid and .oracle.poison_values_remaining == 0))
  ' <<<"$event_gate" >/dev/null
  async_gate="$($project_root/.build/ane-fieldfare-async-gate "$model_dir")"
  jq -e '
    .status == "validated" and
    .lifecycle.compile_count == 1 and
    .lifecycle.load_count == 1 and
    .lifecycle.mapped_request_banks == 8 and
    .async.maximum_current_async_requests_in_flight >= 2 and
    (.async.completions_by_bank | all(.completed and .oracle.valid)) and
    .negative_control.duplicate_rejected and
    .timeout.zero_nanosecond_timeout_observed and
    .timeout.recovered_with_five_second_wait and
    (.shared_event_transaction.construction_attempted | not)
  ' <<<"$async_gate" >/dev/null
  "$project_root/.build/ane-fieldfare-router-gate" "$model_dir" 2
  "$project_root/.build/ane-fieldfare-projection-gate" "$model_dir"
  "$project_root/.build/ane-fieldfare-routed-layer-gate" "$model_dir"
  "$project_root/.build/ane-fieldfare-attention-gate" "$model_dir"
  "$project_root/.build/ane-fieldfare-token-decode-gate" "$model_dir"
  "$project_root/.build/ane-fieldfare-ane-expert-dispatch-gate" "$model_dir"
  "$project_root/.build/ane-fieldfare-metal-expert-gate" "$model_dir" 6
  batched_head="$($project_root/.build/ane-fieldfare-batched-head-gate "$model_dir")"
  jq -e '
    .schema == "ane-fieldfare.batched-greedy-head.v1" and
    .status == "validated" and .positions == 8 and
    .token_mismatches == 0 and .maximum_raw_logit_error == 0 and
    .compile_count == 1 and .pipeline_load_count == 16
  ' <<<"$batched_head" >/dev/null
  generation="$(
    ANE_FIELDFARE_RAW_PROMPT=1 \
    "$project_root/.build/ane-fieldfare-generate" "$model_dir" \
      "The capital of France is" 8
  )"
  jq -e '
    .status == "accepted" and
    .expert_backend == "metal_compact_int4" and
    .generated_tokens == [1623,1133,506,45518,1852,2591,497,236761] and
    .metal_compact.mapped_layers == 30 and
    .metal_compact.library_compile_count == 1 and
    .metal_compact.pipeline_load_count == 16 and
    .metal_compact.exact_cache_slots_per_layer == 20 and
    .metal_compact.fp16_weight_carrier_bytes == 0 and
    .metal_compact.dense_breakdown.attention_input.dispatches == 390 and
    .metal_compact.dense_breakdown.attention_output.dispatches == 390 and
    .metal_compact.dense_breakdown.shared_ffn.dispatches == 390 and
    .metal_compact.dense_breakdown.head.dispatches == 8 and
    .private_ane.expert_dispatches == 0
  ' <<<"$generation" >/dev/null
  templated_generation="$(
    "$project_root/.build/ane-fieldfare-generate" "$model_dir" \
      "The capital of France is" 16
  )"
  jq -e '
    .status == "accepted" and
    .prompt_mode == "gemma_user_chat_template" and
    .text == "The capital of France is **Paris**." and
    .stop_reason == "stop_token" and
    .stop_token == 106 and
    (.generated_tokens | index(106) | not)
  ' <<<"$templated_generation" >/dev/null
  ane_prefill_generation="$(
    ANE_FIELDFARE_ANE_PROJECTION_MB=80 \
    ANE_FIELDFARE_ANE_PREFILL=1 \
      "$project_root/.build/ane-fieldfare-generate" "$model_dir" \
        "The capital of France is" 1
  )"
  jq -e '
    .status == "accepted" and
    .generated_tokens == [818] and
    .prefill_mode == "ane_tiled64_projection_metal_routed" and
    .prefill_tiles == 1 and
    .largest_prefill_tile == 18 and
    .ane_resident_projections.promoted_layers == 1 and
    .ane_resident_projections.programs == 1 and
    .ane_resident_projections.compile_count == 1 and
    .ane_resident_projections.load_count == 1 and
    .ane_resident_projections.attention_input_evaluations == 1 and
    .ane_resident_projections.attention_input_positions == 18 and
    .ane_resident_projections.attention_output_evaluations == 1 and
    .ane_resident_projections.attention_output_positions == 18 and
    .ane_resident_projections.blocking_ane_host_waits == 2 and
    .ane_resident_projections.metal_completion_host_waits == 0 and
    (.ane_resident_projections.shared_event_requested | not) and
    (.ane_resident_projections.decode_enabled | not) and
    .private_ane.expert_dispatches == 0
  ' <<<"$ane_prefill_generation" >/dev/null
  device_prefill_generation="$(
    ANE_FIELDFARE_ANE_PROJECTION_MB=80 \
    ANE_FIELDFARE_ANE_PREFILL=1 \
    ANE_FIELDFARE_DEVICE_PREFILL=1 \
      "$project_root/.build/ane-fieldfare-generate" "$model_dir" \
        "The capital of France is" 1
  )"
  jq -e '
    .status == "accepted" and
    .generated_tokens == [818] and
    .device_prefill_requested and
    .ane_resident_projections.promoted_layers == 1 and
    .ane_resident_projections.shared_event_enabled and
    .ane_resident_projections.device_pipeline_evaluations == 1 and
    .ane_resident_projections.device_pipeline_positions == 18 and
    .ane_resident_projections.device_pipeline_blocking_host_waits == 1 and
    .ane_resident_projections.device_pipeline_intermediate_host_copies == 0 and
    .metal_attention.direct_event_dispatches == 1 and
    .metal_attention.direct_event_positions == 18 and
    .metal_attention.host_waits == 29 and
    .prefill_shared_ffn == "metal_batched_grouped_routed" and
    .metal_compact.dense_breakdown.shared_ffn.dispatches == 30 and
    .metal_compact.dispatches < 200
  ' <<<"$device_prefill_generation" >/dev/null
  ane_event_prefill_generation="$(
    ANE_FIELDFARE_ANE_PROJECTION_MB=80 \
    ANE_FIELDFARE_ANE_PREFILL=1 \
    ANE_FIELDFARE_ANE_METAL_EVENT=1 \
      "$project_root/.build/ane-fieldfare-generate" "$model_dir" \
        "The capital of France is" 1
  )"
  jq -e '
    .status == "accepted" and
    .generated_tokens == [818] and
    .ane_resident_projections.shared_event_requested and
    .ane_resident_projections.shared_event_enabled and
    .ane_resident_projections.shared_event_evaluations == 2 and
    .ane_resident_projections.shared_event_fallbacks == 0 and
    .ane_resident_projections.blocking_ane_host_waits == 0 and
    .ane_resident_projections.metal_completion_host_waits == 2 and
    .ane_resident_projections.zero_timeout_ane_completion_drains == 2 and
    .ane_resident_projections.event_fallback_reason == "none"
  ' <<<"$ane_event_prefill_generation" >/dev/null
  ane_event_fallback_generation="$(
    ANE_FIELDFARE_ANE_PROJECTION_MB=80 \
    ANE_FIELDFARE_ANE_PREFILL=1 \
    ANE_FIELDFARE_ANE_METAL_EVENT=1 \
    ANE_FIELDFARE_ANE_METAL_EVENT_FORCE_FALLBACK=1 \
      "$project_root/.build/ane-fieldfare-generate" "$model_dir" \
        "The capital of France is" 1
  )"
  jq -e '
    .status == "accepted" and
    .generated_tokens == [818] and
    .ane_resident_projections.shared_event_requested and
    (.ane_resident_projections.shared_event_enabled | not) and
    .ane_resident_projections.shared_event_evaluations == 0 and
    .ane_resident_projections.shared_event_fallbacks == 2 and
    .ane_resident_projections.blocking_ane_host_waits == 2 and
    .ane_resident_projections.metal_completion_host_waits == 0 and
    .ane_resident_projections.event_fallback_reason ==
      "forced shared-event fail-closed control"
  ' <<<"$ane_event_fallback_generation" >/dev/null
  ane_decode_opt_in="$(
    ANE_FIELDFARE_ANE_PROJECTION_MB=80 \
    ANE_FIELDFARE_ANE_PREFILL=1 \
    ANE_FIELDFARE_ANE_PROJECTION_DECODE=1 \
      "$project_root/.build/ane-fieldfare-generate" "$model_dir" \
        "The capital of France is" 1
  )"
  jq -e '
    .status == "accepted" and
    .ane_resident_projections.promoted_layers == 1 and
    .ane_resident_projections.decode_enabled == true
  ' <<<"$ane_decode_opt_in" >/dev/null
  long_prefill_prompt='The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Portugal is Lisbon. The capital of Japan is Tokyo. The capital of Canada is Ottawa. The capital of Brazil is Brasilia. The capital of Colombia is Bogota. The capital of Australia is Canberra. The capital of India is New Delhi. The capital of Egypt is Cairo. Now answer briefly: the capital of France is'
  tiled_prefill_generation="$(
    ANE_FIELDFARE_ANE_PROJECTION_MB=80 \
    ANE_FIELDFARE_ANE_PREFILL=1 \
      "$project_root/.build/ane-fieldfare-generate" "$model_dir" \
        "$long_prefill_prompt" 1
  )"
  jq -e '
    .status == "accepted" and
    .prompt_tokens == 109 and
    .generated_tokens == [50429] and
    .text == "Paris" and
    .prefill_tiles == 2 and
    .largest_prefill_tile == 64 and
    .prefill_shared_ffn == "metal_per_position" and
    .ane_resident_projections.programs == 1 and
    .ane_resident_projections.compile_count == 1 and
    .ane_resident_projections.load_count == 1 and
    .ane_resident_projections.attention_input_evaluations == 2 and
    .ane_resident_projections.attention_input_positions == 109 and
    .ane_resident_projections.attention_output_evaluations == 2 and
    .ane_resident_projections.attention_output_positions == 109 and
    .metal_compact.dense_breakdown.attention_input.dispatches == 58 and
    .metal_compact.dense_breakdown.attention_output.dispatches == 58
  ' <<<"$tiled_prefill_generation" >/dev/null
  grouped_prefill_generation="$(
    ANE_FIELDFARE_ANE_PROJECTION_MB=80 \
    ANE_FIELDFARE_ANE_PREFILL=1 \
    ANE_FIELDFARE_DEVICE_PREFILL=1 \
      "$project_root/.build/ane-fieldfare-generate" "$model_dir" \
        "$long_prefill_prompt" 1
  )"
  jq -e '
    .status == "accepted" and
    .prompt_tokens == 109 and
    .generated_tokens == [50429] and
    .text == "Paris" and
    .prefill_tiles == 2 and
    .prefill_shared_ffn == "metal_batched_grouped_routed" and
    .metal_compact.pipeline_load_count == 16 and
    .metal_compact.dense_breakdown.shared_ffn.dispatches == 60 and
    .metal_compact.dispatches < 300 and
    .ane_resident_projections.device_pipeline_intermediate_host_copies == 0
  ' <<<"$grouped_prefill_generation" >/dev/null
  "$project_root/.build/ane-fieldfare-compact-expert-gate" "$model_dir"
  "$project_root/.build/ane-fieldfare-ane-head-gate" "$model_dir"
  "$project_root/.build/ane-fieldfare-transport-benchmarks" position-lane "$model_dir" 16 7
  "$project_root/.build/ane-fieldfare-transport-benchmarks" expert-layout \
    "$model_dir/packed_experts/layer_00.bin" 57 31
  ANE_FIELDFARE_PREP_BREAKDOWN=1 \
    "$project_root/.build/ane-fieldfare-transport-benchmarks" prep-breakdown "$model_dir" 64
  ANE_FIELDFARE_EXPERT_CONSTRUCTION=fused_fp16 \
  ANE_FIELDFARE_EXPERT_SEQUENCE=0,1,0 \
    "$project_root/.build/ane-fieldfare-expert-gate" "$model_dir" 0 0
fi

# Residency and large-constant probes deliberately remain opt-in: their
# lifecycle effects and runtime are the subject of the experiment itself.
if [[ "${ANE_FIELDFARE_RUN_EXPENSIVE_PRIVATE_GATES:-0}" == 1 ]]; then
  "$project_root/.build/ane-fieldfare-private-benchmarks" compression-spelling
  "$project_root/.build/ane-fieldfare-private-benchmarks" native-dtype-probe
  "$project_root/.build/ane-fieldfare-private-benchmarks" native-dtype
  "$project_root/.build/ane-fieldfare-private-benchmarks" compressed-constant
  "$project_root/.build/ane-fieldfare-private-benchmarks" program-residency
fi
