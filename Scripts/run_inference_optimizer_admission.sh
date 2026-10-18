#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model="${1:-$project_root/checkpoints/gturbo/gemma4.gturbo}"
prompt="${ANE_FIELDFARE_OPTIMIZER_PROMPT:-The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Portugal is Lisbon. The capital of Japan is Tokyo. The capital of Canada is Ottawa. The capital of Brazil is Brasilia. The capital of Colombia is Bogota. The capital of Australia is Canberra. The capital of India is New Delhi. The capital of Egypt is Cairo. Now answer briefly: the capital of France is}"
threshold="${ANE_FIELDFARE_OPTIMIZER_THRESHOLD:-1.03}"
scratch="$(mktemp -d "${TMPDIR:-/tmp}/ane-fieldfare-optimizer.XXXXXX")"
trap 'rm -rf -- "$scratch"' EXIT

baseline="$scratch/baseline.json"
candidate="$scratch/candidate.json"
final="$scratch/final.json"
capabilities="$scratch/backend-capabilities.json"
observations="$scratch/cost-observations.json"
mkdir -p "$project_root/.build/inference-plan-cache"

# These are an explicit, replaceable backend contract and measured-cost input,
# not optimizer policy. Production admission must point provenance at retained
# machine receipts for the exact host/build under test.
jq -n '{
  max_position_carrier: 64,
  resident_expert_slots: 15,
  expert_weight_state_width: 8,
  resident_projection_workspace_bytes: 83886080,
  device_attention_prefill: true,
  weight_state_transport: true,
  native_route_verification: true,
  provenance: "local retained ANE gates; candidate pending admission"
}' >"$capabilities"
jq -n '{
  weight_state_transport_ms: 18.701458333333335,
  receiver_lead_time_ms: 0
}' >"$observations"

"$project_root/Scripts/with_benchmark_lock.sh" env \
  ANE_FIELDFARE_RAW_PROMPT=1 \
  "$project_root/.build/ane-fieldfare-generate" "$model" "$prompt" 1 >"$baseline"

"$project_root/Scripts/with_benchmark_lock.sh" env \
  ANE_FIELDFARE_RAW_PROMPT=1 \
  ANE_FIELDFARE_TRACE_CANDIDATE=1 \
  ANE_FIELDFARE_BACKEND_CAPABILITIES="$capabilities" \
  ANE_FIELDFARE_COST_OBSERVATIONS="$observations" \
  "$project_root/.build/ane-fieldfare-generate" "$model" "$prompt" 1 >"$candidate"

jq -e --argjson threshold "$threshold" --slurpfile baseline "$baseline" '
  .status == "accepted" and
  .inference_optimizer.plan_source == "unadmitted_candidate" and
  .prompt_token_ids == $baseline[0].prompt_token_ids and
  .generated_tokens == $baseline[0].generated_tokens and
  .text == $baseline[0].text and
  ($baseline[0].elapsed_ms / .elapsed_ms) >= $threshold
' "$candidate" >/dev/null

signature="$(jq -r '.inference_optimizer.architecture_signature' "$candidate")"
safe_signature="$(tr -cd 'A-Za-z0-9._-' <<<"$signature")"
plan_cache="$project_root/.build/inference-plan-cache/$safe_signature.json"
jq -n --argjson threshold "$threshold" \
  --slurpfile baseline "$baseline" --slurpfile candidate "$candidate" '
  {
    schema: "ane-fieldfare.execution-plan-admission.v1",
    status: "validated",
    plan: $candidate[0].inference_optimizer.plan,
    admission: {
      oracle: "exact prompt token IDs, generated token IDs, and decoded text",
      exact: (($baseline[0].prompt_token_ids == $candidate[0].prompt_token_ids) and
              ($baseline[0].generated_tokens == $candidate[0].generated_tokens) and
              ($baseline[0].text == $candidate[0].text)),
      baseline_elapsed_ms: $baseline[0].elapsed_ms,
      candidate_elapsed_ms: $candidate[0].elapsed_ms,
      speedup: ($baseline[0].elapsed_ms / $candidate[0].elapsed_ms),
      threshold: $threshold,
      threshold_cleared: (($baseline[0].elapsed_ms / $candidate[0].elapsed_ms) >= $threshold)
    },
    claim_boundary: "Training-free execution-plan admission for one architecture signature and declared prefill-width regime; not a cross-model throughput claim."
  }
' >"$plan_cache"

"$project_root/Scripts/with_benchmark_lock.sh" env \
  ANE_FIELDFARE_RAW_PROMPT=1 \
  ANE_FIELDFARE_TRACE_PLAN="$plan_cache" \
  "$project_root/.build/ane-fieldfare-generate" "$model" "$prompt" 1 >"$final"

jq -e --slurpfile candidate "$candidate" '
  .status == "accepted" and
  .inference_optimizer.plan_source == "validated_cache" and
  .generated_tokens == $candidate[0].generated_tokens and
  .text == $candidate[0].text
' "$final" >/dev/null

host="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m)"
os="$(sw_vers -productVersion) ($(sw_vers -buildVersion))"
jq -n --arg host "$host" --arg os "$os" --arg plan_cache "$plan_cache" \
  --slurpfile baseline "$baseline" --slurpfile candidate "$candidate" --slurpfile final "$final" '
  {
    schema: "ane-fieldfare.inference-trace-optimizer.v1",
    status: "validated",
    host: $host,
    os: $os,
    training_required: false,
    optimizer: {
      architecture_signature: $candidate[0].inference_optimizer.architecture_signature,
      architecture: $candidate[0].inference_optimizer.architecture,
      candidate_generation: $candidate[0].inference_optimizer.plan.generation,
      admitted_plan_cache: $plan_cache
    },
    admission: {
      baseline_elapsed_ms: $baseline[0].elapsed_ms,
      candidate_elapsed_ms: $candidate[0].elapsed_ms,
      speedup: ($baseline[0].elapsed_ms / $candidate[0].elapsed_ms),
      exact_prompt_tokens: ($baseline[0].prompt_token_ids == $candidate[0].prompt_token_ids),
      exact_generated_tokens: ($baseline[0].generated_tokens == $candidate[0].generated_tokens),
      exact_text: ($baseline[0].text == $candidate[0].text)
    },
    promoted_execution: {
      plan_source: $final[0].inference_optimizer.plan_source,
      elapsed_ms: $final[0].elapsed_ms,
      prompt_tokens: $final[0].prompt_tokens,
      generated_tokens: $final[0].generated_tokens,
      text: $final[0].text,
      prefill_mode: $final[0].prefill_mode,
      routed_command_buffers: $final[0].metal_compact.dispatches,
      expert_cache_hits: $final[0].metal_compact.exact_cache_hits,
      expert_cache_misses: $final[0].metal_compact.exact_cache_misses,
      ane_promoted_layers: $final[0].ane_resident_projections.promoted_layers
    },
    claim_boundary: "A shape-derived candidate was executed against the generic reference path, admitted by an exact end-to-end token/text oracle and a wall-time threshold, persisted, reloaded, and executed through the production generator. The first model adapter remains Gemma; cross-architecture rediscovery is not yet claimed."
  }
' | tee "$project_root/results/inference-trace-optimizer-m5-macos27.json"
