#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
case "${1:-}" in
  "") (( $# == 0 )) || exit 2; exec "$project_root/Scripts/test_core.sh" ;;
  --historical) (( $# == 1 )) || exit 2; exec "$project_root/Scripts/test_historical.sh" ;;
  *) echo "usage: $0 [--historical]" >&2; exit 2 ;;
esac
