#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
turbo_root="$(cd -- "$project_root/../reference/turbo-fieldfare" && pwd)"
model_dir="${ANE_FIELDFARE_GTURBO_MODEL:-$project_root/checkpoints/gturbo/gemma4.gturbo}"
result_path="${ANE_FIELDFARE_COMPARISON_RECEIPT:-$project_root/results/inference-comparison-m5-macos27.json}"
prompt="${ANE_FIELDFARE_COMPARISON_PROMPT:-The capital of France is}"
max_new="${ANE_FIELDFARE_COMPARISON_MAX_NEW:-8}"
scratch_path="$project_root/.build/turbo-reference"
binary="$scratch_path/release/TurboFieldfareCLI"

fail() {
  printf 'comparison preflight rejected: %s\n' "$1" >&2
  exit 3
}

[[ -f "$model_dir/manifest.json" ]] || fail "missing manifest.json"
[[ -f "$model_dir/model_weights.bin" ]] || fail "missing model_weights.bin"
[[ "$max_new" =~ ^[1-9][0-9]*$ ]] || fail "ANE_FIELDFARE_COMPARISON_MAX_NEW must be positive"

if pgrep -f 'TurboFieldfareServer|TurboFieldfareMac|TurboFieldfareDecodeService|TurboFieldfareCLI|TurboFieldfarePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm' >/dev/null; then
  fail "another model process is active"
fi
competing_io=false
if pgrep -f 'hf download|huggingface.*download|snapshot_download' >/dev/null; then
  if [[ "${ANE_FIELDFARE_ALLOW_COMPETING_IO:-0}" != 1 ]]; then
    fail "a model download is active; refusing a contaminated comparison"
  fi
  competing_io=true
fi

mkdir -p "$project_root/.build" "$project_root/results"
swift build -c release --product TurboFieldfareCLI \
  --package-path "$turbo_root" \
  --scratch-path "$scratch_path"

run_dir="$(mktemp -d "$project_root/.build/inference-comparison.XXXXXX")"
trap 'rm -rf "$run_dir"' EXIT

command=(
  "$binary"
  --model "$model_dir"
  --prompt "$prompt"
  --max-new "$max_new"
  --max-context 4096
  --temperature 0
  --top-k 0
  --top-p 1
)

set +e
/usr/bin/time -l "${command[@]}" >"$run_dir/stdout" 2>"$run_dir/stderr"
exit_code=$?
set -e

footer="$(rg '^\[stop=' "$run_dir/stderr" | tail -1 || true)"
peak_rss="$(awk '/maximum resident set size/{print $1}' "$run_dir/stderr" | tail -1)"
peak_footprint="$(awk '/peak memory footprint/{print $1}' "$run_dir/stderr" | tail -1)"
commit="$(git -C "$turbo_root" rev-parse HEAD)"
stdout_text="$(<"$run_dir/stdout")"
stderr_text="$(<"$run_dir/stderr")"
generation_receipt="$project_root/results/generation-raw8-m5-macos27.json"
[[ -f "$generation_receipt" ]] || fail "missing end-to-end ANE generation receipt"

jq -n \
  --arg schema "ane-fieldfare.inference-comparison.v1" \
  --arg status "$([[ $exit_code -eq 0 && -n "$footer" ]] && printf validated || printf failed)" \
  --arg host "$(scutil --get LocalHostName 2>/dev/null || hostname)" \
  --arg os "$(sw_vers -productVersion) ($(sw_vers -buildVersion))" \
  --arg turboCommit "$commit" \
  --arg prompt "$prompt" \
  --argjson maxNew "$max_new" \
  --arg command "${command[*]}" \
  --argjson exitCode "$exit_code" \
  --arg footer "$footer" \
  --arg output "$stdout_text" \
  --arg stderr "$stderr_text" \
  --argjson competingIO "$competing_io" \
  --argjson peakRSS "${peak_rss:-0}" \
  --argjson peakFootprint "${peak_footprint:-0}" \
  --slurpfile ane "$generation_receipt" \
  '{
    schema: $schema,
    status: $status,
    host: $host,
    os: $os,
    comparison_kind: "capability_and_resource_axes_only",
    conditions: {
      competing_model_download_active: $competingIO,
      performance_eligible: ($competingIO | not)
    },
    regular_turbo_fieldfare: {
      commit: $turboCommit,
      prompt: $prompt,
      max_new_tokens: $maxNew,
      sampling: {temperature: 0, top_k: null, top_p: 1},
      command: $command,
      exit_status: $exitCode,
      timing_footer: $footer,
      output: $output,
      stderr: $stderr,
      maximum_rss_bytes: $peakRSS,
      peak_physical_footprint_bytes: $peakFootprint,
      claim_status: (if $exitCode == 0 and ($footer | length) > 0 then "validated_streaming_generation" else "failed" end)
    },
    ane_fieldfare: {
      executable_token_inference_present: true,
      retained_run: $ane[0],
      claim_status: "accepted_end_to_end_generation"
    },
    comparable_metrics: [],
    target_constraints_observed: {
      turbo_process_peak_physical_footprint_bytes: $peakFootprint,
      ane_fieldfare_target_bytes: 2147483648,
      ane_fieldfare_hard_safety_limit_bytes: 2415919104,
      turbo_memory_band: (if $peakFootprint <= 2147483648 then "within_target"
                          elif $peakFootprint <= 2415919104 then "within_tolerance"
                          else "over_safety_limit" end)
    },
    non_comparability_reason: "Both runtimes generate the same retained eight-token raw-prompt sequence, but the ANE receipt is a separate clean run and this script has not yet rerun both implementations under one uncontended timing window.",
    claim_boundary: "A regular Turbo Fieldfare generation run plus the retained end-to-end ANE Fieldfare generation receipt. Exact output equality is a correctness control; throughput is not comparable until both arms are run-matched in one clean window."
  }' >"$result_path"

cat "$result_path"
exit "$exit_code"
