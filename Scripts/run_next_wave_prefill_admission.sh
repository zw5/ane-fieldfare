#!/usr/bin/env bash
# Serial, fail-closed P1/P3 admission harness for the exact next-wave prefill
# transport. This script deliberately does not build: the binary under test is
# frozen before the benchmark lock is acquired.
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
protected_scratch="$(cd -- "$project_root/../reference/turbo-fieldfare" 2>/dev/null && pwd)/scratch"
lock_wrapper="$project_root/Scripts/with_benchmark_lock.sh"

# Hold one lock for the complete window. Per-arm locking would permit an
# unrelated benchmark to enter between paired observations.
if [[ "${ANE_NEXT_WAVE_LOCK_HELD:-0}" != 1 ]]; then
  exec "$lock_wrapper" env ANE_NEXT_WAVE_LOCK_HELD=1 "$0" "$@"
fi

model="${ANE_NEXT_WAVE_MODEL:-$project_root/checkpoints/gturbo/gemma4.gturbo}"
binary="${ANE_NEXT_WAVE_BINARY:-$project_root/.build/swift-hotpath/release/ANEFieldfareHotpathCLI}"
receipt="${ANE_NEXT_WAVE_RECEIPT:-$project_root/results/next-wave-prefill-admission-m5-macos27.json}"
prompt_file="${ANE_NEXT_WAVE_PROMPT_FILE:-}"
prompt="${ANE_NEXT_WAVE_PROMPT:-The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Portugal is Lisbon. The capital of Japan is Tokyo. The capital of Canada is Ottawa. The capital of Brazil is Brasilia. The capital of Colombia is Bogota. The capital of Australia is Canberra. The capital of India is New Delhi. The capital of Egypt is Cairo. Now answer briefly: the capital of France is}"
max_context="${ANE_NEXT_WAVE_MAX_CONTEXT:-4096}"
max_new="${ANE_NEXT_WAVE_MAX_NEW:-1}"
expert_top_k="${ANE_NEXT_WAVE_EXPERT_TOP_K:-8}"
prefill_expert_cap="${ANE_NEXT_WAVE_PREFILL_EXPERT_CAP:-128}"
poll_seconds="${ANE_NEXT_WAVE_WATCHDOG_POLL_SECONDS:-0.25}"
disk_device="${ANE_NEXT_WAVE_DISK_DEVICE:-disk0}"

# Literal preregistered limits from NEXT_WAVE_STRICT_IMPROVEMENT_PLAN.md.
preflight_free_min=30
runtime_free_min=20
footprint_ceiling=2415919104
# Loading the 12.9 GB mapped control can legitimately evict cold host pages on
# a 16 GB machine. The hard free-memory and process-footprint watchdogs remain
# authoritative; this limit rejects only pathological swap churn.
swap_growth_ceiling=1073741824
prefill_speedup_min=1.10
process_speedup_min=1.08
retained_prefill_ceiling_s=3.812
retained_routed_ffn_ceiling_s=3.278
decode_regression_max=0.02
first_token_regression_max=0.02
measured_pair_min=4

fail() { printf 'next-wave prefill admission rejected: %s\n' "$1" >&2; exit 3; }
command -v jq >/dev/null || fail "jq is required"
command -v memory_pressure >/dev/null || fail "memory_pressure is required"
command -v vm_stat >/dev/null || fail "vm_stat is required"
command -v pmset >/dev/null || fail "pmset is required"
command -v pgrep >/dev/null || fail "pgrep is required"
command -v footprint >/dev/null || fail "footprint is required for the byte-exact process watchdog"
[[ -x "$binary" ]] || fail "missing frozen binary at $binary; build before admission"
[[ -f "$model/manifest.json" && -f "$model/model_weights.bin" ]] || fail "incomplete model directory"
[[ "$max_context" =~ ^[1-9][0-9]*$ && "$max_new" =~ ^[1-9][0-9]*$ &&
   "$expert_top_k" =~ ^[1-9][0-9]*$ && "$prefill_expert_cap" =~ ^[1-9][0-9]*$ ]] || fail "token limits and expert controls must be positive integers"
[[ "$poll_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail "invalid watchdog poll interval"
case "$(realpath "$model")/" in "$protected_scratch/"*) fail "model points into protected Turbo installation scratch" ;; esac
case "$(realpath "$binary")" in "$protected_scratch/"*) fail "binary points into protected Turbo installation scratch" ;; esac
case "$receipt" in "$project_root/results/"*) ;; *) fail "machine receipt must stay under results/" ;; esac

mkdir -p "$project_root/.build" "$project_root/results"
run_dir="$(mktemp -d "$project_root/.build/next-wave-prefill-admission.XXXXXX")"
case "$run_dir/" in
  "$project_root/.build/"*) ;;
  *) fail "scratch escaped the repository .build boundary" ;;
esac
case "$run_dir/" in
  "$protected_scratch/"*) fail "refusing protected Turbo installation scratch" ;;
esac

active_pid=""
cleanup() {
  if [[ -n "$active_pid" ]] && kill -0 "$active_pid" 2>/dev/null; then
    kill -TERM "$active_pid" 2>/dev/null || true
    wait "$active_pid" 2>/dev/null || true
  fi
  rm -rf -- "$run_dir"
}
trap cleanup EXIT HUP INT TERM

# This check names processes only. It does not inspect files or payloads below
# the protected installation scratch directory.
conflict_pattern='ANEFieldfareHotpathCLI|ane-fieldfare-generate|TurboFieldfare(Server|Mac|DecodeService|CLI|PackageTests)|mlx[_-]lm|swift-build|swift-frontend|xcodebuild|hf[[:space:]]+download|huggingface.*download|snapshot_download|turbo-fieldfare/scratch'
conflicting_process_active() { pgrep -f "$conflict_pattern" >/dev/null; }
if conflicting_process_active; then
  fail "a model, compiler, installer, or competing model-I/O process is active"
fi

memory_free_percent() {
  memory_pressure -Q 2>/dev/null | awk -F': *' '/System-wide memory free percentage/{gsub(/%/,"",$2); value=$2} END {if (value ~ /^[0-9]+([.][0-9]+)?$/) print value; else exit 1}'
}

throttled_pages() {
  vm_stat 2>/dev/null | awk -F': *' '/Pages throttled/{gsub(/[.[:space:]]/,"",$2); value=$2} END {if (value ~ /^[0-9]+$/) print value; else exit 1}'
}

swap_used_bytes() {
  sysctl -n vm.swapusage 2>/dev/null | awk '
    function bytes(v, n,u) {
      n=v; sub(/[MGT]$/, "", n); u=substr(v,length(v),1)
      if (u=="G") return n*1073741824
      if (u=="T") return n*1099511627776
      return n*1048576
    }
    {for(i=1;i<=NF;i++) if($i=="used" && $(i+1)=="=") {printf "%.0f\n",bytes($(i+2)); found=1}}
    END {if(!found) exit 1}'
}

disk_bytes() {
  iostat -Id "$disk_device" 2>/dev/null | awk 'NF >= 3 && $1 ~ /^[0-9.]+$/ {mb=$3} END {if (mb ~ /^[0-9.]+$/) printf "%.0f\n", mb*1048576; else exit 1}'
}

wall_nanoseconds() {
  perl -MTime::HiRes=time -e 'printf "%.0f\n", time() * 1000000000'
}

process_phys_footprint_bytes() {
  local pid="$1"
  footprint -p "$pid" -f bytes --noCategories 2>/dev/null | awk '
    /phys_footprint:/ {value=$2}
    END {if (value ~ /^[0-9]+$/) print value; else exit 1}'
}

thermal_snapshot() {
  local destination="$1"
  pmset -g therm >"$destination.raw" 2>&1 || return 1
  local cpu scheduler level warning
  cpu="$(awk -F'= *' '/CPU_Speed_Limit/{gsub(/[^0-9]/,"",$2); print $2}' "$destination.raw" | tail -1)"
  scheduler="$(awk -F'= *' '/CPU_Scheduler_Limit/{gsub(/[^0-9]/,"",$2); print $2}' "$destination.raw" | tail -1)"
  level="$(awk -F'= *' '/Thermal_Level/{gsub(/[^0-9]/,"",$2); print $2}' "$destination.raw" | tail -1)"
  [[ "$cpu" =~ ^[0-9]+$ ]] || cpu=100
  [[ "$scheduler" =~ ^[0-9]+$ ]] || scheduler=100
  [[ "$level" =~ ^[0-9]+$ ]] || level=0
  warning=false
  if (( cpu < 100 || scheduler < 100 || level > 0 )); then warning=true; fi
  jq -n --arg raw "$(<"$destination.raw")" --argjson cpu "$cpu" \
    --argjson scheduler "$scheduler" --argjson level "$level" --argjson warning "$warning" \
    '{raw:$raw,cpu_speed_limit_percent:$cpu,cpu_scheduler_limit_percent:$scheduler,thermal_level:$level,warning:$warning}' >"$destination"
}

system_snapshot() {
  local destination="$1" free throttled swap load1 load5 load15 disk
  free="$(memory_free_percent)" || return 1
  throttled="$(throttled_pages)" || return 1
  swap="$(swap_used_bytes)" || return 1
  read -r load1 load5 load15 < <(sysctl -n vm.loadavg | tr -d '{}') || return 1
  disk="$(disk_bytes)" || disk=""
  thermal_snapshot "$destination.thermal" || return 1
  jq -n --argjson epoch "$(date +%s)" --argjson free "$free" \
    --argjson throttled "$throttled" --argjson swap "$swap" \
    --argjson load1 "$load1" --argjson load5 "$load5" --argjson load15 "$load15" \
    --arg disk "$disk" --slurpfile thermal "$destination.thermal" '
      {epoch_seconds:$epoch,free_memory_percent:$free,throttled_pages:$throttled,
       swap_used_bytes:$swap,load_average:[$load1,$load5,$load15],
       disk_bytes:(if $disk == "" then null else ($disk|tonumber) end),thermal:$thermal[0]}' >"$destination"
}

preflight_snapshot="$run_dir/preflight.json"
system_snapshot "$preflight_snapshot" || fail "unable to obtain complete host resource preflight"
jq -e --argjson free "$preflight_free_min" '
  .free_memory_percent >= $free and .throttled_pages == 0 and (.thermal.warning|not)
' "$preflight_snapshot" >/dev/null || fail "host resource preflight did not meet 30% free, zero-throttle, no-warning gate"

prompt_args=(--prompt "$prompt")
if [[ -n "$prompt_file" ]]; then
  [[ -f "$prompt_file" ]] || fail "prompt file not found"
  case "$(realpath "$prompt_file")" in "$protected_scratch/"*) fail "prompt points into protected Turbo installation scratch" ;; esac
  prompt_args=(--prompt-file "$prompt_file")
fi
base_command=("$binary" --model "$model" "${prompt_args[@]}" --max-new "$max_new" \
  --max-context "$max_context" --prefill-chunk 64 --expert-top-k "$expert_top_k" \
  --prefill-expert-cap "$prefill_expert_cap" \
  --temperature 0 --top-k 0 --top-p 1 --repetition-penalty 1 --seed 0 --quiet)
printf -v command_string '%q ' "${base_command[@]}"

arms_jsonl="$run_dir/arms.jsonl"
: >"$arms_jsonl"

admission_abort() {
  local reason="$1" run_exit="${2:-1}"
  local retained_arms="$run_dir/rejected-arms.json"
  jq -s '.' "$arms_jsonl" >"$retained_arms"
  jq -n --arg reason "$reason" --argjson exit "$run_exit" --arg command "$command_string" \
    --arg model "$model" --arg binary "$binary" --slurpfile preflight "$preflight_snapshot" \
    --slurpfile arms "$retained_arms" --argjson footprint "$footprint_ceiling" \
    --argjson preflight_free "$preflight_free_min" --argjson runtime_free "$runtime_free_min" \
    --argjson swap_growth "$swap_growth_ceiling" --argjson prefill_speedup "$prefill_speedup_min" \
    --argjson process_speedup "$process_speedup_min" '
    {schema:"ane-fieldfare.next-wave-prefill-admission.v1",status:"rejected",
     rejection_reason:$reason,exact_command:$command,exit_status:$exit,
     artifact:{model_directory:$model,binary:$binary},preflight:$preflight[0],arms:$arms[0],
     thresholds:{preflight_free_memory_percent_min:$preflight_free,
       runtime_free_memory_percent_min:$runtime_free,process_phys_footprint_bytes_max:$footprint,
       swap_growth_bytes_max:$swap_growth,throttled_pages_max:0,measured_pairs_min:4,
       median_complete_prefill_speedup_min:$prefill_speedup,median_whole_process_speedup_min:$process_speedup},
     p3_verdict:"FAIL",wave_verdict:"FAIL",
     claim_boundary:"The harness stopped at the first failed or resource-rejected arm. No later arm was launched; no performance or wave claim is admitted."}' >"$receipt"
  printf 'next-wave prefill admission rejected: %s\nreceipt: %s\n' "$reason" "$receipt" >&2
  exit "$run_exit"
}

extract_profile() {
  local stderr_file="$1" output="$2"
  awk 'index($0,"{")==1 {print}' "$stderr_file" | while IFS= read -r line; do
    jq -c 'select(type=="object" and .schema=="ane-fieldfare.hotpath-profile.v1")' <<<"$line" 2>/dev/null || true
  done | tail -1 >"$output"
  [[ -s "$output" ]]
}

run_arm() { # arm-name phase pair-index order sequence-index
  local arm="$1" phase="$2" pair_index="$3" order="$4" sequence_index="$5"
  local stem="$run_dir/${sequence_index}-${phase}-${arm}"
  local before="$stem.before.json" after="$stem.after.json"
  local stdout_file="$stem.stdout" stderr_file="$stem.stderr" watchdog_file="$stem.watchdog"
  local start_ns end_ns process_wall exit_code peak=0 breach="" sample_free sample_swap sample_throttled sample_footprint arm_command
  local watchdog_samples=0 thermal_probe="$stem.thermal-probe.json"

  conflicting_process_active && fail "competing process appeared before $phase/$arm"
  system_snapshot "$before" || fail "unable to snapshot before $phase/$arm"
  if ! jq -e --argjson free "$preflight_free_min" '.free_memory_percent >= $free and .throttled_pages == 0 and (.thermal.warning|not)' "$before" >/dev/null; then
    fail "arm preflight failed for $phase/$arm"
  fi
  local swap_before throttled_before min_free disk_before
  swap_before="$(jq -r '.swap_used_bytes' "$before")"
  throttled_before="$(jq -r '.throttled_pages' "$before")"
  min_free="$(jq -r '.free_memory_percent' "$before")"
  disk_before="$(jq -r '.disk_bytes // empty' "$before")"

  start_ns="$(wall_nanoseconds)"
  if [[ "$arm" == control ]]; then
    arm_command="ANE_FIELDFARE_HOTPATH_PROFILE=1 ANE_FIELDFARE_PREFILL_BOUNDARY_ORACLE=1 $command_string"
    env ANE_FIELDFARE_HOTPATH_PROFILE=1 ANE_FIELDFARE_PREFILL_BOUNDARY_ORACLE=1 \
      "${base_command[@]}" >"$stdout_file" 2>"$stderr_file" &
  else
    arm_command="ANE_FIELDFARE_HOTPATH_PROFILE=1 ANE_FIELDFARE_PREFILL_BOUNDARY_ORACLE=1 ANE_FIELDFARE_PREAD_WIDE_PREFILL_CANDIDATE=1 $command_string"
    env ANE_FIELDFARE_HOTPATH_PROFILE=1 ANE_FIELDFARE_PREFILL_BOUNDARY_ORACLE=1 ANE_FIELDFARE_PREAD_WIDE_PREFILL_CANDIDATE=1 \
      "${base_command[@]}" >"$stdout_file" 2>"$stderr_file" &
  fi
  active_pid=$!

  while kill -0 "$active_pid" 2>/dev/null; do
    sample_footprint="$(process_phys_footprint_bytes "$active_pid" || true)"
    if [[ ! "$sample_footprint" =~ ^[0-9]+$ ]]; then breach="phys_footprint_unobservable"
    elif (( sample_footprint > footprint_ceiling )); then
      watchdog_samples=$((watchdog_samples + 1))
      breach="phys_footprint_ceiling"
    elif ! sample_free="$(memory_free_percent)"; then breach="memory_pressure_unobservable"
    elif awk -v x="$sample_free" -v floor="$runtime_free_min" 'BEGIN{exit !(x<floor)}'; then breach="free_memory_floor"
    elif ! sample_swap="$(swap_used_bytes)"; then breach="swap_unobservable"
    elif (( sample_swap - swap_before > swap_growth_ceiling )); then breach="swap_growth_ceiling"
    elif ! sample_throttled="$(throttled_pages)"; then breach="throttled_pages_unobservable"
    elif (( sample_throttled != 0 || sample_throttled > throttled_before )); then breach="throttled_pages"
    elif ! thermal_snapshot "$thermal_probe"; then breach="thermal_state_unobservable"
    elif [[ "$(jq -r '.warning' "$thermal_probe")" == true ]]; then breach="thermal_or_performance_warning"
    else
      watchdog_samples=$((watchdog_samples + 1))
      (( sample_footprint > peak )) && peak="$sample_footprint"
      min_free="$(awk -v a="$min_free" -v b="$sample_free" 'BEGIN{print (a<b?a:b)}')"
    fi
    if [[ "$breach" == phys_footprint_unobservable ]] && ! kill -0 "$active_pid" 2>/dev/null; then
      breach=""
      break
    fi
    if [[ -n "$breach" ]]; then
      printf '%s\n' "$breach" >"$watchdog_file"
      kill -TERM "$active_pid" 2>/dev/null || true
      sleep 1
      kill -KILL "$active_pid" 2>/dev/null || true
      break
    fi
    sleep "$poll_seconds"
  done
  set +e
  wait "$active_pid"
  exit_code=$?
  set -e
  active_pid=""
  end_ns="$(wall_nanoseconds)"
  process_wall="$(awk -v a="$start_ns" -v b="$end_ns" 'BEGIN{printf "%.9f",(b-a)/1000000000}')"
  if ! system_snapshot "$after"; then
    breach="${breach:-post_snapshot_unobservable}"
    jq '.thermal.warning=true' "$before" >"$after"
  fi
  if (( watchdog_samples == 0 )) && [[ -z "$breach" ]]; then breach="phys_footprint_unobserved"; fi

  local profile="$stem.profile.json" profile_ok=false status="rejected"
  if extract_profile "$stderr_file" "$profile"; then profile_ok=true; else printf '{}\n' >"$profile"; fi
  local swap_after throttled_after disk_after physical_bytes thermal_warning
  swap_after="$(jq -r '.swap_used_bytes // 0' "$after" 2>/dev/null || echo 0)"
  throttled_after="$(jq -r '.throttled_pages // -1' "$after" 2>/dev/null || echo -1)"
  disk_after="$(jq -r '.disk_bytes // empty' "$after" 2>/dev/null || true)"
  thermal_warning="$(jq -r '.thermal.warning // true' "$after" 2>/dev/null || echo true)"
  if [[ "$thermal_warning" == true && -z "$breach" ]]; then breach="thermal_or_performance_warning"; fi
  if (( swap_after - swap_before > swap_growth_ceiling )) && [[ -z "$breach" ]]; then breach="swap_growth_ceiling"; fi
  if (( throttled_after != 0 || throttled_after > throttled_before )) && [[ -z "$breach" ]]; then breach="throttled_pages"; fi
  physical_bytes=""
  if [[ -n "$disk_before" && -n "$disk_after" ]]; then physical_bytes=$((disk_after - disk_before)); fi
  if [[ -n "$breach" ]]; then status="resource_rejected"
  elif (( exit_code == 0 )) && [[ "$profile_ok" == true ]]; then status="accepted"
  fi

  jq -n --arg arm "$arm" --arg phase "$phase" --arg order "$order" \
    --argjson pair "$pair_index" --argjson sequence "$sequence_index" --arg status "$status" \
    --arg breach "$breach" --argjson exit "$exit_code" --arg command "$arm_command" \
    --arg text "$(<"$stdout_file")" --arg stderr "$(<"$stderr_file")" \
    --argjson wall "$process_wall" --argjson peak "$peak" --argjson min_free "$min_free" \
    --argjson watchdog_samples "$watchdog_samples" \
    --arg physical "$physical_bytes" --slurpfile before "$before" --slurpfile after "$after" \
    --slurpfile profile "$profile" '
      {arm:$arm,phase:$phase,pair_index:$pair,comparison_order:$order,sequence_index:$sequence,
       status:$status,watchdog_action:(if $breach=="" then null else ("killed:"+$breach) end),
       exact_command:$command,exit_status:$exit,text:$text,stderr:$stderr,profile:$profile[0],
       metrics:{process_wall_seconds:$wall,
         prefill_wall_seconds:($profile[0].prefill_wall_seconds // null),
         first_token_latency_seconds:($profile[0].first_token_latency_seconds // null),
         recurrent_decode_tokens_per_second:($profile[0].recurrent_decode_tokens_per_second // (if ($profile[0].decode_seconds // 0)>0 then (($profile[0].new_tokens // 0)/$profile[0].decode_seconds) else null end)),
         routed_ffn_wall_seconds:($profile[0].prefill_routed_ffn_seconds // $profile[0].routed_ffn_wall_seconds // null),
         expert_io_wall_seconds:($profile[0].prefill_transport_pread_seconds // $profile[0].io_seconds // null),
         logical_requested_bytes:($profile[0].prefill_transport_requested_bytes // null),
         explicit_pread_bytes:($profile[0].prefill_transport_pread_bytes // null),
         explicit_pread_calls:($profile[0].prefill_transport_pread_calls // null),
         physical_disk_bytes:($profile[0].prefill_physical_disk_bytes // null),
         system_disk_bytes_delta:(if $physical=="" then null else ($physical|tonumber) end),
         cache_hits:($profile[0].prefill_transport_cache_hits // null),
         cache_misses:($profile[0].prefill_transport_cache_misses // null),
         queue_wait_seconds:($profile[0].prefill_transport_queue_wait_seconds // null),
         completion_wait_seconds:($profile[0].prefill_transport_completion_wait_seconds // null),
         forced_drains:($profile[0].prefill_transport_forced_drains // null),
         maximum_pending_depth:($profile[0].prefill_transport_max_pending_depth // null),
         slot_high_water_mark:($profile[0].prefill_transport_slot_high_water // null),
         peak_phys_footprint_bytes:$peak,watchdog_samples:$watchdog_samples,
         swap_delta_bytes:($after[0].swap_used_bytes-$before[0].swap_used_bytes),
         memory_pressure_floor_percent:$min_free,throttled_pages_before:$before[0].throttled_pages,
         throttled_pages_after:$after[0].throttled_pages,thermal_performance_warning:$after[0].thermal.warning},
       resource_snapshots:{before:$before[0],after:$after[0]}}' >>"$arms_jsonl"

  [[ "$status" == accepted ]]
}

sequence=0
run_one() { sequence=$((sequence + 1)); run_arm "$1" "$2" "$3" "$4" "$sequence"; }

# Fast path for directly establishing whether the bounded candidate can run
# safely and produce a complete profile on this host. It deliberately makes no
# comparative or promotion claim.
if [[ "${ANE_NEXT_WAVE_CANDIDATE_ONLY:-0}" == 1 ]]; then
  run_one candidate candidate_probe 0 C || admission_abort "bounded candidate probe failed"
  jq -s --arg command "$command_string" --slurpfile preflight "$preflight_snapshot" '
    {schema:"ane-fieldfare.next-wave-candidate-probe.v1",status:"accepted",
     exact_command:$command,preflight:$preflight[0],arm:.[0],
     claim_boundary:"One bounded pread candidate run completed under the hard process-footprint and free-memory watchdogs. This is an execution and measurement receipt, not a speedup claim."}' \
    "$arms_jsonl" >"$receipt"
  jq '{schema,status,metrics:.arm.metrics,profile:.arm.profile}' "$receipt"
  printf 'receipt: %s\n' "$receipt" >&2
  exit 0
fi

# P1 order is literal mapped-control / pread-candidate / mapped-control.
run_one control p1_oracle 0 ACA || admission_abort "P1 opening mapped control failed"
run_one candidate p1_oracle 0 ACA || admission_abort "P1 pread candidate failed"
run_one control p1_oracle 0 ACA || admission_abort "P1 closing mapped control failed"

# Counterbalanced storage/thermal warmups; never included in measured medians.
run_one control warmup 0 AB || admission_abort "AB control warmup failed"
run_one candidate warmup 0 AB || admission_abort "AB candidate warmup failed"
run_one candidate warmup 0 BA || admission_abort "BA candidate warmup failed"
run_one control warmup 0 BA || admission_abort "BA control warmup failed"

# Four registered matched pairs, evenly split between AB and BA.
for registered in '1 AB' '2 BA' '3 AB' '4 BA'; do
  read -r pair order <<<"$registered"
  if [[ "$order" == AB ]]; then
    run_one control measured "$pair" "$order" || admission_abort "measured pair $pair control failed"
    run_one candidate measured "$pair" "$order" || admission_abort "measured pair $pair candidate failed"
  else
    run_one candidate measured "$pair" "$order" || admission_abort "measured pair $pair candidate failed"
    run_one control measured "$pair" "$order" || admission_abort "measured pair $pair control failed"
  fi
done

host="$(scutil --get LocalHostName 2>/dev/null || hostname)"
host_chip="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m)"
os_version="$(sw_vers -productVersion)"
os_build="$(sw_vers -buildVersion)"
branch="$(git -C "$project_root" branch --show-current)"
dirty_summary="$(git -C "$project_root" status --short)"
binary_sha="$(shasum -a 256 "$binary" | awk '{print $1}')"
manifest_sha="$(shasum -a 256 "$model/manifest.json" | awk '{print $1}')"
ane_arch="$(ioreg -l 2>/dev/null | awk -F'= ' '/ane.*arch|ANE.*arch/{print $2; exit}' || true)"

jq -s --arg host "$host" --arg chip "$host_chip" --arg os "$os_version" --arg os_build "$os_build" \
  --arg ane_arch "$ane_arch" --arg branch "$branch" --arg dirty "$dirty_summary" \
  --arg model "$model" --arg manifest_sha "$manifest_sha" --arg binary "$binary" \
  --arg binary_sha "$binary_sha" --arg command "$command_string" \
  --argjson footprint "$footprint_ceiling" --argjson preflight_free "$preflight_free_min" \
  --argjson runtime_free "$runtime_free_min" --argjson swap_growth "$swap_growth_ceiling" \
  --argjson prefill_speedup "$prefill_speedup_min" --argjson process_speedup "$process_speedup_min" \
  --argjson prefill_ceiling "$retained_prefill_ceiling_s" --argjson routed_ceiling "$retained_routed_ffn_ceiling_s" \
  --argjson decode_regression "$decode_regression_max" --argjson first_token_regression "$first_token_regression_max" '
  def median: sort | if length%2==1 then .[length/2|floor] else ((.[length/2-1]+.[length/2])/2) end;
  def metric($arm;$name): [.[]|select(.phase=="measured" and .arm==$arm)|.metrics[$name]] | if any(.==null) then null else median end;
  def ratio($numerator;$denominator): if $numerator==null or $denominator==null or $denominator==0 then null else $numerator/$denominator end;
  . as $arms |
  (metric("control";"prefill_wall_seconds")) as $control_prefill |
  (metric("candidate";"prefill_wall_seconds")) as $candidate_prefill |
  (metric("control";"process_wall_seconds")) as $control_process |
  (metric("candidate";"process_wall_seconds")) as $candidate_process |
  (metric("control";"recurrent_decode_tokens_per_second")) as $control_decode |
  (metric("candidate";"recurrent_decode_tokens_per_second")) as $candidate_decode |
  (metric("control";"first_token_latency_seconds")) as $control_first |
  (metric("candidate";"first_token_latency_seconds")) as $candidate_first |
  (ratio($control_prefill;$candidate_prefill)) as $prefill_ratio |
  (ratio($control_process;$candidate_process)) as $process_ratio |
  ([range(1;5) as $p | {pair:$p,
      order:([$arms[]|select(.phase=="measured" and .pair_index==$p)|.comparison_order][0]),
      control:([$arms[]|select(.phase=="measured" and .pair_index==$p and .arm=="control")|.metrics.process_wall_seconds][0]),
      candidate:([$arms[]|select(.phase=="measured" and .pair_index==$p and .arm=="candidate")|.metrics.process_wall_seconds][0])} |
      . + {candidate_favored:(.candidate < .control)}]) as $pairs |
  ([.[]|select(.phase=="p1_oracle")]) as $oracle |
  (($oracle|length)==3 and $oracle[0].arm=="control" and $oracle[1].arm=="candidate" and $oracle[2].arm=="control" and
    $oracle[0].profile.wide_prefill_transport=="mapped-control" and
    $oracle[1].profile.wide_prefill_transport=="pread-double-buffered" and
    $oracle[2].profile.wide_prefill_transport=="mapped-control" and
    ($oracle|map(.profile.prompt_token_ids)|all(.!=null)) and
    ($oracle|map(.profile.prompt_token_ids|length)|all(.==97)) and
    ($oracle|map(.profile.prompt_token_ids)|unique|length)==1 and
    ($oracle|map(.profile.generated_token_ids)|unique|length)==1 and
    ($oracle|map(.profile.generated_token_ids[-1])|all(.==236881)) and
    ($oracle|map(.text)|unique|length)==1 and
    ($oracle|map(.profile.stop_reason)|all(.!=null)) and ($oracle|map(.profile.stop_reason)|unique|length)==1 and
    ($oracle|map(.profile.prefill_boundary_oracle.passed)|all(.==true)) and
    ($oracle|map(.profile.prefill_boundary_oracle.digests)|unique|length)==1 and
    ($oracle|map(.profile.exact_route_oracle.passed)|all(.==true)) and
    ($oracle|map(.profile.exact_route_oracle.digest)|unique|length)==1 and
    ($oracle|map(.profile.all_values_finite)|all(.==true))) as $exact_oracle |
  {
    schema:"ane-fieldfare.next-wave-prefill-admission.v1",status:"completed",
    host:{name:$host,chip:$chip,os_version:$os,os_build:$os_build,observable_ane_architecture:(if $ane_arch=="" then null else $ane_arch end)},
    repository:{branch:$branch,dirty_state_summary:$dirty},
    artifact:{model_directory:$model,model_manifest_sha256:$manifest_sha,binary:$binary,binary_sha256:$binary_sha},
    execution:{exact_command:$command,exit_status:0,compile_count:0,load_count:length,
      serial_one_model_process:true,benchmark_lock:true,protected_scratch_untouched:true,
      comparison_sequence:[.[]|{sequence_index,phase,arm,pair_index,comparison_order}]},
    mode:{exact_top_k:8,lossy:false,mapped_control_switch:"production default",pread_candidate_switch:"ANE_FIELDFARE_PREAD_WIDE_PREFILL_CANDIDATE=1"},
    thresholds:{preflight_free_memory_percent_min:$preflight_free,runtime_free_memory_percent_min:$runtime_free,
      process_phys_footprint_bytes_max:$footprint,swap_growth_bytes_max:$swap_growth,throttled_pages_max:0,
      measured_pairs_min:4,median_complete_prefill_speedup_min:$prefill_speedup,
      median_whole_process_speedup_min:$process_speedup,retained_64_plus_17_prefill_seconds_max:$prefill_ceiling,
      retained_64_plus_17_routed_ffn_seconds_strict_max:$routed_ceiling,
      recurrent_decode_regression_max_fraction:$decode_regression,first_token_latency_regression_max_fraction:$first_token_regression},
    arms:$arms,
    oracle:{mapped_candidate_mapped_order:true,exact:$exact_oracle,retained_fixture_token_236881:
      (($oracle|map(.profile.generated_token_ids[-1])|all(.==236881)))},
    medians:{control_prefill_seconds:$control_prefill,candidate_prefill_seconds:$candidate_prefill,
      complete_prefill_speedup:$prefill_ratio,control_process_seconds:$control_process,
      candidate_process_seconds:$candidate_process,whole_process_speedup:$process_ratio,
      control_decode_tokens_per_second:$control_decode,candidate_decode_tokens_per_second:$candidate_decode,
      control_first_token_latency_seconds:$control_first,candidate_first_token_latency_seconds:$candidate_first},
    matched_pairs:$pairs,
    gates:{exact_numerical_oracle:$exact_oracle,
      explicit_transport_accounting:($arms|[.[]|select(.arm=="candidate")]|all(
        .profile.wide_prefill_transport=="pread-double-buffered" and
        .metrics.logical_requested_bytes!=null and .metrics.explicit_pread_bytes!=null and
        .metrics.logical_requested_bytes==.metrics.explicit_pread_bytes and
        .metrics.explicit_pread_calls>0 and
        .metrics.system_disk_bytes_delta!=null and
        (.profile.prefill_transport_speculative_bytes // -1)==0 and
        .metrics.maximum_pending_depth==1 and .metrics.slot_high_water_mark!=null and
        .metrics.slot_high_water_mark<=16)),
      mapped_control_explicit:($arms|[.[]|select(.arm=="control")]|all(.profile.wide_prefill_transport=="mapped-control")),
      complete_prefill_speedup:(($prefill_ratio//0)>=$prefill_speedup),
      whole_process_speedup:(($process_ratio//0)>=$process_speedup),
      every_matched_order_favors_candidate:($pairs|all(.candidate_favored)),
      retained_prefill_ceiling:(($candidate_prefill//1e99)<=$prefill_ceiling),
      retained_routed_ffn_ceiling:((metric("candidate";"routed_ffn_wall_seconds")//1e99)<$routed_ceiling),
      recurrent_decode_no_regression:(($candidate_decode//0) >= (($control_decode//1e99)*(1-$decode_regression))),
      first_token_no_regression:(($candidate_first//1e99) <= (($control_first//0)*(1+$first_token_regression))),
      resource_gate:($arms|all(.status=="accepted" and .metrics.peak_phys_footprint_bytes<=$footprint and
        .metrics.memory_pressure_floor_percent >= $runtime_free and .metrics.swap_delta_bytes <= $swap_growth and
        .metrics.throttled_pages_after==0 and (.metrics.thermal_performance_warning|not)))},
    p3_verdict:null,
    wave_verdict:"PENDING_P4",
    claim_boundary:"This receipt can validate or reject the exact mapped-control versus route-authoritative pread candidate through P3. It does not execute P4 actual 16k/32k plus 1,000-token admission and therefore cannot declare the next wave PASS. A resource rejection or missing oracle/metric is fail-closed."
  } | .p3_verdict=(if ([.gates[]]|all) then "PASS" else "FAIL" end)
' "$arms_jsonl" >"$receipt"

jq '{schema,p3_verdict,wave_verdict,gates,medians,receipt_claim:.claim_boundary}' "$receipt"
printf 'receipt: %s\n' "$receipt" >&2
jq -e '.p3_verdict=="PASS"' "$receipt" >/dev/null
