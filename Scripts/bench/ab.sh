#!/usr/bin/env bash
# A/B two bench snapshots: ABBA-ordered rounds in separate processes, so drift
# (thermal state, page cache) lands on both sides, then compare.py's
# statistics and HTML report.
#
# usage: Scripts/bench/ab.sh <snapshot-A> <snapshot-B> [rounds-per-side=2] [bench flags...]
#   e.g. Scripts/bench/snapshot.sh base   # before the change
#        Scripts/bench/snapshot.sh try    # after it
#        Scripts/bench/ab.sh base try 2 --drafter checkpoints/dflash/gemma-4-26B-A4B-it-DFlash
# Bench flags default to the quick suite (3 prompts, 192 tokens, 2 reps).
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
a="${1:?}"; b="${2:?}"; rounds="${3:-2}"; shift $(( $# >= 3 ? 3 : $# ))
out="$root/.build/bench-runs/$(date +%Y%m%d-%H%M%S)-$a-vs-$b"
mkdir -p "$out"
run() {  # snapshot round [bench flags...]
  local name="$1" round="$2"; shift 2
  local bin="$root/.build/bench-bins/$name"
  ANE_FIELDFARE_BENCH_GIT="$(cat "$bin/GIT")" "$bin/ANEFieldfareBench" \
    --model "$root/checkpoints/gturbo/gemma4.gturbo" --label "$name" --out "$out/$name-r$round.json" "$@"
}
for ((i = 0; i < rounds; i++)); do
  if (( i % 2 == 0 )); then run "$a" "$i" "$@"; run "$b" "$i" "$@"
  else run "$b" "$i" "$@"; run "$a" "$i" "$@"; fi
done
python3 "$root/Scripts/bench/compare.py" --a "$out/$a"-r*.json --b "$out/$b"-r*.json --html "$out/report.html"
echo "$out/report.html"
