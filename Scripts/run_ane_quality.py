#!/usr/bin/env python3
"""Fixed-input logits oracle. Run through with_benchmark_lock.sh on an ANE host."""
import argparse
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
p = argparse.ArgumentParser()
p.add_argument("fixture", type=Path, help="JSON prompt_token_ids and continuation_token_ids")
p.add_argument("output", type=Path, help="new results directory")
p.add_argument("--chunk", type=int, choices=(512, 1024), default=1024)
p.add_argument("--ane-attention", choices=("causal-simd", "full-tensorops-2d-preferred"), default="causal-simd")
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
receipt = {"schema": "ane-fieldfare.quality-run.v1", "host": platform.node(),
           "os_build": subprocess.check_output(["sw_vers", "-buildVersion"], text=True).strip(),
           "ane_architecture": "not exposed by this runner", "runs": []}
env = {k: v for k, v in os.environ.items() if not k.startswith("ANE_FIELDFARE_")}
for name, chunk in (("metal128", 128), (f"metal{a.chunk}", a.chunk), (f"ane{a.chunk}", a.chunk)):
    prefix = a.output.resolve() / name
    run_env = env | ({"ANE_FIELDFARE_ANE_ROUTED_PREFILL_LAYERS": "30", "ANE_FIELDFARE_PREFILL_ATTENTION": a.ane_attention} if name.startswith("ane") else {})
    command = [str(ROOT / ".build/swift-hotpath/release/ANEFieldfareQualityGate"),
               str(ROOT / "checkpoints/gturbo/gemma4.gturbo"), str(a.fixture.resolve()), str(chunk), str(prefix)]
    print("RUN", name, flush=True)
    started = time.monotonic()
    with Path(str(prefix) + ".log").open("wb") as output:
        process = subprocess.Popen(command, env=run_env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            status = process.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            status = process.wait()
    run = {"command": command, "environment": {k: v for k, v in run_env.items() if k.startswith("ANE_FIELDFARE_")},
           "exit_status": status, "wall_seconds": time.monotonic() - started}
    metadata = Path(str(prefix) + ".json")
    if metadata.exists():
        data = json.loads(metadata.read_text())
        run.update(ane_routed_prefill=data["ane_routed_prefill"], ane_shared_ffn=data["ane_shared_ffn"],
                   numerical_result={"all_values_finite": data["all_values_finite"], "rows": data["rows"]})
    receipt["runs"].append(run)
    (a.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if status:
        raise SystemExit(f"{name} failed: see {prefix}.log")
for name in (f"metal{a.chunk}", f"ane{a.chunk}"):
    subprocess.run([sys.executable, str(ROOT / "Scripts/compare_logits.py"),
                    str(a.output / "metal128"), str(a.output / name), str(a.output / (name + "-comparison.json"))], check=True)
receipt["claim_boundary"] = "Independent full-logit comparison on identical fixed token inputs. Diagnostic fixtures, not a held-out language-quality benchmark."
(a.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
