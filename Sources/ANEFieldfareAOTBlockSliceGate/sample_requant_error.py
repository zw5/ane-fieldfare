#!/usr/bin/env python3
"""CPU-only distribution screen for authentic expert requantization."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

from build_requant_sweep import decoded_matrix, requantize


FIXTURES = ((0, 0, "baseline"), (0, 79, "retained_route"),
            (15, 42, "mid_layer"), (29, 127, "high_id_final_layer"))
BLOCKS = (128, 256, 512, 2816)
POSITIONS, HIDDEN = 8, 2816


def reconstruct(slices, rows: int, columns: int) -> np.ndarray:
    result = np.empty((rows, columns), dtype=np.float32)
    for begin, codes, scale, bias in slices:
        result[:, begin:begin + codes.shape[1]] = (
            codes.astype(np.float32) * scale.astype(np.float32)[:, None] +
            bias.astype(np.float32)[:, None]
        )
    return result


def expert(weight_gate, weight_up, weight_down, x):
    gate = (x @ weight_gate.T).astype(np.float16).astype(np.float32)
    up = (x @ weight_up.T).astype(np.float16).astype(np.float32)
    gelu = 0.5 * gate * (1.0 + np.tanh(0.7978845608028654 *
        (gate + 0.044715 * gate * gate * gate)))
    activated = (gelu * up).astype(np.float16).astype(np.float32)
    return (activated @ weight_down.T).astype(np.float16).astype(np.float32)


def metrics(actual, exact):
    delta = actual.astype(np.float64) - exact.astype(np.float64)
    exact64, actual64 = exact.astype(np.float64), actual.astype(np.float64)
    exact_square = float(np.square(exact64).sum())
    actual_square = float(np.square(actual64).sum())
    dot = float((exact64 * actual64).sum())
    return {
        "values": int(exact.size),
        "max_absolute_error": float(np.abs(delta).max()),
        "rmse": float(np.sqrt(np.square(delta).mean())),
        "exact_rms": float(np.sqrt(np.square(exact64).mean())),
        "normalized_rmse": float(np.sqrt(np.square(delta).sum() / exact_square)),
        "cosine_similarity": dot / np.sqrt(exact_square * actual_square),
    }


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: sample_requant_error.py <gturbo-root>")
    root = Path(sys.argv[1]).resolve()
    layout = json.loads((root / "packed_experts/layout.json").read_text())
    column = np.arange(HIDDEN, dtype=np.float32)[None, :]
    position = np.arange(POSITIONS, dtype=np.float32)[:, None]
    inputs = (0.83 * np.sin(column * 0.013 + position * 0.37) +
              0.41 * np.cos(column * 0.031 - position * 0.19)).astype(np.float16).astype(np.float32)
    results = []
    for layer_index, expert_index, role in FIXTURES:
        layer = next(x for x in layout["layers"] if int(x["layer"]) == layer_index)
        record = next(x for x in layer["experts"] if int(x["expert"]) == expert_index)
        blob = (root / "packed_experts" / layer["file"]).read_bytes()
        base, tensors = int(record["offset"]), record["tensors"]
        exact_weights = {name: decoded_matrix(blob, base, tensors, name)
                         for name in ("gate", "up", "down")}
        exact_output = expert(exact_weights["gate"], exact_weights["up"],
                              exact_weights["down"], inputs)
        arms = []
        for block in BLOCKS:
            quantized = {}
            weight_squared = 0.0; weight_values = 0; weight_max = 0.0
            for name, weight in exact_weights.items():
                slices, error = requantize(weight, block)
                quantized[name] = reconstruct(slices, *weight.shape)
                weight_squared += error["rmse"] ** 2 * error["values"]
                weight_values += error["values"]
                weight_max = max(weight_max, error["max_absolute_error"])
            arms.append({
                "block": block,
                "weight_error": {"rmse": (weight_squared / weight_values) ** 0.5,
                                 "max_absolute_error": weight_max},
                "output_error": metrics(expert(quantized["gate"], quantized["up"],
                                                quantized["down"], inputs), exact_output),
            })
        results.append({"layer": layer_index, "expert": expert_index, "role": role,
                        "input_rms": float(np.sqrt(np.square(inputs.astype(np.float64)).mean())),
                        "exact_output_rms": float(np.sqrt(np.square(exact_output.astype(np.float64)).mean())),
                        "arms": arms})
    print(json.dumps({
        "schema": "ane-fieldfare.authentic-requant-distribution.v1", "status": "validated",
        "checkpoint": str(root), "fixtures": results,
        "input": {"positions": POSITIONS, "construction": "deterministic sin/cos, FP16-rounded, RMS near normalized expert input; no prompt calibration"},
        "quantization": "independent per-output-channel minmax affine uint4 by fixed target block; no training, calibration, or iterative fitting",
        "compile_count": 0, "load_count": 0,
        "claim_boundary": "CPU-only deterministic distribution screen across four authentic experts. It measures requantization error against the exact compact checkpoint, not ANE execution, routed-token quality, or generation coherence."
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
