#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
lock_dir="$project_root/.build/ane-fieldfare-benchmark.lock"

if (( $# == 0 )); then
  echo "usage: $0 <command> [arguments...]" >&2
  exit 2
fi

mkdir -p "$project_root/.build"
owner_token="pid=$$;random=$RANDOM;started=$(date +%s)"

if ! mkdir "$lock_dir" 2>/dev/null; then
  if [[ -r "$lock_dir/owner" ]]; then
    echo "benchmark lock occupied: $(<"$lock_dir/owner")" >&2
  else
    echo "benchmark lock occupied: owner record unavailable" >&2
  fi
  exit 75
fi

printf '%s\n' "$owner_token" > "$lock_dir/owner"

release_lock() {
  local recorded=""
  [[ -r "$lock_dir/owner" ]] && recorded="$(<"$lock_dir/owner")"
  if [[ "$recorded" == "$owner_token" ]]; then
    rm -f "$lock_dir/owner"
    rmdir "$lock_dir" 2>/dev/null || true
  else
    echo "benchmark lock ownership changed; refusing cleanup" >&2
  fi
}
trap release_lock EXIT HUP INT TERM

"$@"
