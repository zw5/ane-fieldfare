#!/usr/bin/env python3
"""Compare retained full logits under identical fixed token inputs, without sampling."""
import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("reference", type=Path)
p.add_argument("candidate", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
r = json.loads(Path(str(a.reference) + ".json").read_text())
c = json.loads(Path(str(a.candidate) + ".json").read_text())
for key in ("prompt_token_ids", "continuation_token_ids", "rows", "vocabulary"):
    if r[key] != c[key]:
        raise SystemExit("incomparable inputs: " + key)
shape = (r["rows"], r["vocabulary"])
x = np.fromfile(r["logits_path"], dtype="<f2").astype(np.float64).reshape(shape)
y = np.fromfile(c["logits_path"], dtype="<f2").astype(np.float64).reshape(shape)
finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
if not finite:
    raise SystemExit("non-finite logits")

def logprobs(z):
    shifted = z - z.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))

lx, ly = logprobs(x), logprobs(y)
px, py = np.exp(lx), np.exp(ly)
kl = (px * (lx - ly)).sum(axis=1)
tv = np.abs(px - py).sum(axis=1) / 2
targets = r["continuation_token_ids"]
rx = -lx[np.arange(len(targets)), targets]
cy = -ly[np.arange(len(targets)), targets]
report = {
    "schema": "ane-fieldfare.fixed-continuation-comparison.v1",
    "status": "validated" if np.array_equal(x, y) else "measured_numerical_difference",
    "reference": str(a.reference), "candidate": str(a.candidate),
    "rows": shape[0], "vocabulary": shape[1], "all_values_finite": finite,
    "exact_fp16_logits": bool(np.array_equal(x, y)),
    "maximum_absolute_logit_error": float(np.abs(x-y).max()),
    "rms_logit_error": float(np.sqrt(np.square(x-y).mean())),
    "relative_l2_logit_error": float(np.linalg.norm(x-y) / np.linalg.norm(x)),
    "greedy_agreement_count": int((x.argmax(axis=1) == y.argmax(axis=1)).sum()),
    "mean_kl_reference_to_candidate_nats": float(kl.mean()),
    "maximum_kl_reference_to_candidate_nats": float(kl.max()),
    "mean_total_variation": float(tv.mean()), "maximum_total_variation": float(tv.max()),
    "forced_tokens": len(targets), "reference_mean_nll_nats": float(rx.mean()),
    "candidate_mean_nll_nats": float(cy.mean()), "mean_nll_delta_nats": float((cy-rx).mean()),
    "claim_boundary": "Numerical comparison of full next-token distributions under the same retained inputs, including continuation reads of prefill KV. These inputs are a diagnostic, not a held-out language-quality benchmark."
}
a.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
