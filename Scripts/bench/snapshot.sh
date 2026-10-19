#!/usr/bin/env bash
# Builds the bench and freezes it, with its resource bundles, as a named
# snapshot under .build/bench-bins/<name>, so a later build can be compared
# against it.  usage: Scripts/bench/snapshot.sh <name>
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
name="${1:?usage: snapshot.sh <name>}"
swift build -c release --jobs "${ANE_FIELDFARE_BUILD_JOBS:-8}" --product ANEFieldfareBench \
  --scratch-path "$root/.build/swift-hotpath" --package-path "$root" >"${TMPDIR:-/tmp}/bench-snapshot.log" 2>&1 \
  || { tail -40 "${TMPDIR:-/tmp}/bench-snapshot.log"; exit 1; }
dest="$root/.build/bench-bins/$name"
rm -rf "$dest" && mkdir -p "$dest"
cp "$root/.build/swift-hotpath/release/ANEFieldfareBench" "$dest/"
cp -R "$root/.build/swift-hotpath/release/"*.bundle "$dest/"
sha="$(git -C "$root" rev-parse --short HEAD)"
git -C "$root" diff --quiet HEAD || sha="$sha+dirty($(git -C "$root" diff HEAD | shasum | cut -c1-8))"
echo "$sha" > "$dest/GIT"
echo "snapshot $name: $dest ($sha)"
