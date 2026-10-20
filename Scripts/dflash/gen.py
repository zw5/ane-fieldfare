#!/usr/bin/env python3
"""Greedy target continuations with token IDs, for the offline DFlash acceptance gate."""
import json, os, subprocess, sys
root = "<repo>"
here = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else ".")
prompts = {
    "code": "Write a Python function that merges overlapping intervals in a list, with a docstring and a short example.",
    "math": "A train travels 180 km at 60 km/h and then 120 km at 40 km/h. What is its average speed for the whole trip? Show your reasoning step by step.",
    "explain": "Explain how a hash map works, including collisions and resizing.",
    "chat": "I'm planning a weekend trip to Lisbon. What should I see and eat?",
}
env = {k: v for k, v in os.environ.items() if not k.startswith("ANE_FIELDFARE_")} | {"ANE_FIELDFARE_HOTPATH_PROFILE": "1"}
for name, text in prompts.items():
    cmd = [f"{root}/Scripts/with_benchmark_lock.sh", f"{root}/.build/swift-hotpath/release/ANEFieldfareHotpathCLI",
           "--model", f"{root}/checkpoints/gemma-4-26b-a4b-it-qat.fieldfare", "--chat-prompt", text, "--backend", "metal",
           "--max-new", sys.argv[1] if len(sys.argv) > 1 else "256", "--max-context", "4096",
           "--temperature", "0", "--top-k", "0", "--top-p", "1"]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    prof = next(json.loads(l) for l in p.stderr.splitlines() if l.startswith("{") and '"schema"' in l)
    rec = {"name": name, "prompt": text, "exit": p.returncode, "text": p.stdout,
           "prompt_token_ids": prof["prompt_token_ids"], "generated_token_ids": prof["generated_token_ids"],
           "decode_tps": prof["recurrent_decode_tokens_per_second"],
           "cache_hits": prof["routed_expert_cache_hits"], "cache_misses": prof["routed_expert_cache_misses"],
           "io_seconds": prof["io_seconds"], "decode_seconds": prof["decode_seconds"]}
    json.dump(rec, open(f"{here}/{name}.gen.json", "w"), indent=1)
    print(name, p.returncode, len(rec["prompt_token_ids"]), len(rec["generated_token_ids"]), round(rec["decode_tps"], 2), flush=True)
