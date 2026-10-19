#!/usr/bin/env bash
# End-to-end check of ANEFieldfareServer against the installed model.
#
# Runs the CLI and the server on the same prompts and requires the server's
# output to equal the CLI's: greedy chat (non-streaming and streaming),
# seeded sampled chat, and a raw /v1/completions prompt. With SMOKE_DRAFTER
# set it restarts the server with --drafter and compares a greedy request with
# the CLI's --drafter run (reported, not required: speculative output may
# differ from run to run at near-ties). Prints CLI and server tok/s.
#
# Loads the model several times; run it under the benchmark lock:
#   Scripts/with_benchmark_lock.sh Scripts/server/smoke.sh
#
# Environment: ANE_FIELDFARE_TEST_CHECKPOINT (model, default
# checkpoints/gturbo/gemma4.gturbo), SMOKE_DRAFTER (drafter directory),
# SMOKE_MAX_NEW (default 128), SMOKE_SKIP_BUILD=1.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
model="${ANE_FIELDFARE_TEST_CHECKPOINT:-$project_root/checkpoints/gturbo/gemma4.gturbo}"
drafter="${SMOKE_DRAFTER:-}"
max_new="${SMOKE_MAX_NEW:-128}"
context=4096
bin="$project_root/.build/swift-hotpath/release"
work="$(mktemp -d "${TMPDIR:-/tmp}/ane-fieldfare-smoke.XXXXXX")"
server_pid=""

cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -INT "$server_pid" 2>/dev/null || true
    for _ in $(seq 1 50); do kill -0 "$server_pid" 2>/dev/null || break; sleep 0.2; done
    kill -KILL "$server_pid" 2>/dev/null || true
  fi
  rm -rf "$work"
}
trap cleanup EXIT

[[ -f "$model/model_weights.bin" ]] || { echo "model not installed at $model" >&2; exit 2; }
if [[ "${SMOKE_SKIP_BUILD:-0}" != 1 ]]; then "$project_root/Scripts/build.sh"; fi

chat_prompt="Explain in one paragraph how a refrigerator keeps food cold."
raw_prompt="The three largest planets in the solar system are"
failures=0
check() { # name expected-file actual-file
  if cmp -s "$2" "$3"; then
    echo "PASS $1"
  else
    echo "FAIL $1"; echo "  expected: $(head -c 300 "$2" | tr '\n' '|')"; echo "  actual:   $(head -c 300 "$3" | tr '\n' '|')"
    failures=$((failures + 1))
  fi
}
footer_tps() { sed -n 's/.*tok\/s=\([0-9.]*\).*/\1/p' "$1" | tail -1; }

echo "== CLI reference runs"
"$bin/ANEFieldfareHotpathCLI" --model "$model" --max-context "$context" --chat-prompt "$chat_prompt" \
  --temperature 0 --max-new "$max_new" > "$work/cli-greedy.txt" 2> "$work/cli-greedy.err"
"$bin/ANEFieldfareHotpathCLI" --model "$model" --max-context "$context" --chat-prompt "$chat_prompt" \
  --temperature 0.7 --top-k 40 --top-p 0.9 --seed 7 --max-new "$max_new" --quiet > "$work/cli-sampled.txt"
"$bin/ANEFieldfareHotpathCLI" --model "$model" --max-context "$context" --prompt "$raw_prompt" \
  --temperature 0 --max-new "$max_new" --quiet > "$work/cli-raw.txt"
cli_tps="$(footer_tps "$work/cli-greedy.err")"
if [[ -n "$drafter" ]]; then
  "$bin/ANEFieldfareHotpathCLI" --model "$model" --max-context "$context" --chat-prompt "$chat_prompt" \
    --temperature 0 --max-new "$max_new" --drafter "$drafter" > "$work/cli-drafter.txt" 2> "$work/cli-drafter.err"
  cli_drafter_tps="$(footer_tps "$work/cli-drafter.err")"
fi

start_server() { # extra flags...
  port="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')"
  "$bin/ANEFieldfareServer" --model "$model" --max-context "$context" --port "$port" "$@" 2> "$work/server.err" &
  server_pid=$!
  for _ in $(seq 1 600); do
    curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1 && return 0
    kill -0 "$server_pid" 2>/dev/null || { cat "$work/server.err" >&2; exit 1; }
    sleep 0.5
  done
  echo "server did not become ready" >&2; exit 1
}
stop_server() {
  kill -INT "$server_pid"
  wait "$server_pid" || true
  server_pid=""
}
content() { python3 -c 'import json,sys; d=json.load(sys.stdin); c=d["choices"][0]; sys.stdout.write(c["message"]["content"] if "message" in c else c["text"])'; }

echo "== server"
start_server
base="http://127.0.0.1:$port/v1"
greedy=$(python3 -c 'import json,sys; print(json.dumps({"model":"x","messages":[{"role":"user","content":sys.argv[1]}],"temperature":0,"max_tokens":int(sys.argv[2])}))' "$chat_prompt" "$max_new")
curl -sS "$base/chat/completions" -H 'Content-Type: application/json' -d "$greedy" > "$work/greedy.json"
content < "$work/greedy.json" > "$work/server-greedy.txt"
check "greedy chat (non-streaming) == CLI" "$work/cli-greedy.txt" "$work/server-greedy.txt"
server_tps="$(python3 -c 'import json,sys; print("%.3f" % json.load(open(sys.argv[1]))["timings"]["predicted_per_second"])' "$work/greedy.json")"

stream=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d["stream"]=True; d["stream_options"]={"include_usage":True}; print(json.dumps(d))' "$greedy")
curl -sS -N "$base/chat/completions" -H 'Content-Type: application/json' -d "$stream" > "$work/stream.sse"
python3 - "$work/stream.sse" "$work/server-stream.txt" <<'EOF'
import json, sys
lines = open(sys.argv[1]).read().split("\n\n")
events = [l[len("data: "):] for l in lines if l.startswith("data: ")]
assert events[-1] == "[DONE]", "stream does not end with [DONE]"
chunks = [json.loads(e) for e in events[:-1]]
assert chunks[0]["choices"][0]["delta"].get("role") == "assistant", "first chunk carries no role"
finish = [c["choices"][0]["finish_reason"] for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
assert finish in (["stop"], ["length"]), f"finish reasons {finish}"
assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] > 0, "no usage chunk"
text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"])
open(sys.argv[2], "w").write(text)
EOF
check "greedy chat (streaming) == CLI" "$work/cli-greedy.txt" "$work/server-stream.txt"

sampled=$(python3 -c 'import json,sys; print(json.dumps({"messages":[{"role":"user","content":sys.argv[1]}],"temperature":0.7,"top_k":40,"top_p":0.9,"seed":7,"max_tokens":int(sys.argv[2])}))' "$chat_prompt" "$max_new")
curl -sS "$base/chat/completions" -H 'Content-Type: application/json' -d "$sampled" | content > "$work/server-sampled.txt"
check "seeded sampled chat == CLI" "$work/cli-sampled.txt" "$work/server-sampled.txt"

raw=$(python3 -c 'import json,sys; print(json.dumps({"prompt":sys.argv[1],"temperature":0,"max_tokens":int(sys.argv[2])}))' "$raw_prompt" "$max_new")
curl -sS "$base/completions" -H 'Content-Type: application/json' -d "$raw" | content > "$work/server-raw.txt"
check "raw completion == CLI --prompt" "$work/cli-raw.txt" "$work/server-raw.txt"
grep '^\[chatcmpl\|^\[cmpl' "$work/server.err" || true
stop_server

if [[ -n "$drafter" ]]; then
  echo "== server --drafter"
  start_server --drafter "$drafter"
  base="http://127.0.0.1:$port/v1"
  curl -sS "$base/chat/completions" -H 'Content-Type: application/json' -d "$greedy" > "$work/drafter.json"
  content < "$work/drafter.json" > "$work/server-drafter.txt"
  if cmp -s "$work/cli-drafter.txt" "$work/server-drafter.txt"; then
    echo "SAME greedy chat with drafter == CLI --drafter"
  else
    echo "DIFFERS greedy chat with drafter vs CLI --drafter (near-tie variation is possible)"
  fi
  server_drafter_tps="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1]))["timings"]; print("%.3f (%s)" % (d["predicted_per_second"], d["decode_path"]))' "$work/drafter.json")"
  grep '^\[chatcmpl' "$work/server.err" || true
  stop_server
fi

echo "== tok/s, greedy chat, $max_new new tokens max"
echo "CLI:    $cli_tps"
echo "server: $server_tps"
if [[ -n "$drafter" ]]; then
  echo "CLI --drafter:    $cli_drafter_tps"
  echo "server --drafter: $server_drafter_tps"
fi
if (( failures > 0 )); then echo "$failures check(s) failed"; exit 1; fi
echo "all checks passed"
