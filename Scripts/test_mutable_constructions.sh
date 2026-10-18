#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
"$project_root/Scripts/build.sh"

for construction in \
  baseline buildinfo_plain buildinfo_hash \
  buildinfo_symbol buildinfo_hash_symbol buildinfo_symbol_array buildinfo_hash_symbol_array \
  option_plain option_hash option_symbol option_hash_symbol \
  compile_option_plain compile_option_hash compile_option_symbol compile_option_hash_symbol \
  top_comment; do
  set +e
  ANE_FIELDFARE_MUTABLE_CONSTRUCTION="$construction" \
    "$project_root/.build/ane-fieldfare-mutable-gate"
  construction_status=$?
  set -e
  if [[ $construction_status -ne 0 && $construction_status -ne 1 ]]; then
    exit "$construction_status"
  fi
done
