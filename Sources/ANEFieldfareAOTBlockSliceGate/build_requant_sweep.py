#!/usr/bin/env python3
"""Deterministically coarsen authentic block-64 weights to affine uint4 slices."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
import numpy as np

from build_model import bf16_array, checked_slice, directory_bytes


HIDDEN, INTERMEDIATE, ORIGINAL_GROUP, POSITIONS = 2816, 704, 64, 64


def decoded_matrix(blob: bytes, base: int, tensors: dict, name: str) -> np.ndarray:
    record = tensors[name]
    rows, columns = map(int, record["shape"])
    packed = np.frombuffer(checked_slice(blob, base, record), dtype=np.uint8)
    codes = np.empty(packed.size * 2, dtype=np.uint8)
    codes[0::2], codes[1::2] = packed & 15, packed >> 4
    codes = codes.reshape(rows, columns).astype(np.float32)
    groups = columns // ORIGINAL_GROUP
    scales = bf16_array(checked_slice(blob, base, tensors[name + "_scales"])).reshape(rows, groups)
    biases = bf16_array(checked_slice(blob, base, tensors[name + "_biases"])).reshape(rows, groups)
    return codes * np.repeat(scales, ORIGINAL_GROUP, axis=1) + np.repeat(biases, ORIGINAL_GROUP, axis=1)


def requantize(weight: np.ndarray, block: int):
    slices, squared, maximum = [], 0.0, 0.0
    for begin in range(0, weight.shape[1], block):
        exact = weight[:, begin:begin + block]
        minimum = exact.min(axis=1)
        maximum_value = exact.max(axis=1)
        scale = ((maximum_value - minimum) / 15.0).astype(np.float16)
        bias = minimum.astype(np.float16)
        safe = np.where(scale.astype(np.float32) == 0, 1.0, scale.astype(np.float32))
        codes = np.rint((exact - bias.astype(np.float32)[:, None]) / safe[:, None])
        codes = np.clip(codes, 0, 15).astype(np.uint8)
        reconstructed = codes.astype(np.float32) * scale.astype(np.float32)[:, None] + bias.astype(np.float32)[:, None]
        delta = reconstructed - exact
        squared += float(np.square(delta.astype(np.float64)).sum())
        maximum = max(maximum, float(np.abs(delta).max()))
        slices.append((begin, codes, scale, bias))
    return slices, {"rmse": (squared / weight.size) ** 0.5, "max_absolute_error": maximum,
                    "values": int(weight.size)}


def build_program(matrices: dict):
    def project(x, slices, rows: int, prefix: str):
        partials = []
        for index, (begin, codes, scale, bias) in enumerate(slices):
            width = codes.shape[1]
            xg = mb.slice_by_size(x=x, begin=[0, begin, 0, 0],
                                  size=[1, width, 1, POSITIONS], name=f"{prefix}_x_{index}")
            q = codes[:, :, None, None].astype(types.np_uint4_dtype)
            s = scale.reshape(rows, 1, 1, 1)
            w = mb.constexpr_blockwise_shift_scale(data=q, scale=s, name=f"{prefix}_w_{index}")
            qy = mb.conv(x=xg, weight=w, name=f"{prefix}_q_{index}")
            sx = mb.reduce_sum(x=xg, axes=[1], keep_dims=True, name=f"{prefix}_sx_{index}")
            correction = mb.mul(x=sx, y=bias.reshape(1, rows, 1, 1), name=f"{prefix}_b_{index}")
            partials.append(mb.add(x=qy, y=correction, name=f"{prefix}_p_{index}"))
        level = 0
        while len(partials) > 1:
            partials = [partials[i] if i + 1 == len(partials) else
                        mb.add(x=partials[i], y=partials[i + 1], name=f"{prefix}_sum_{level}_{i//2}")
                        for i in range(0, len(partials), 2)]
            level += 1
        return partials[0]

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, HIDDEN, 1, POSITIONS), dtype=types.fp16)],
                opset_version=ct.target.iOS18)
    def program(x):
        g = project(x, matrices["gate"][0], INTERMEDIATE, "gate")
        u = project(x, matrices["up"][0], INTERMEDIATE, "up")
        g2 = mb.mul(x=g, y=g, name="gate_square")
        g3 = mb.mul(x=g2, y=g, name="gate_cube")
        inner = mb.add(x=g, y=mb.mul(x=g3, y=np.float16(0.044715)), name="gelu_inner")
        tanh = mb.tanh(x=mb.mul(x=inner, y=np.float16(0.7978845608028654)), name="gelu_tanh")
        gelu = mb.mul(x=mb.mul(x=g, y=np.float16(0.5)), y=mb.add(x=tanh, y=np.float16(1.0)), name="gelu")
        activated = mb.mul(x=gelu, y=u, name="activated")
        return mb.cast(x=project(activated, matrices["down"][0], HIDDEN, "down"),
                       dtype="fp32", name="expert")

    return program


def main() -> int:
    if len(sys.argv) != 6:
        raise SystemExit("usage: build_requant_sweep.py <gturbo-root> <layer> <expert> <block> <output.mlpackage>")
    root, layer_index, expert_index = Path(sys.argv[1]).resolve(), int(sys.argv[2]), int(sys.argv[3])
    block, output = int(sys.argv[4]), Path(sys.argv[5]).resolve()
    if block not in (128, 256, 512, 2816): raise ValueError("block must be 128, 256, 512, or 2816")
    layout = json.loads((root / "packed_experts/layout.json").read_text())
    layer = next(x for x in layout["layers"] if int(x["layer"]) == layer_index)
    expert = next(x for x in layer["experts"] if int(x["expert"]) == expert_index)
    blob = (root / "packed_experts" / layer["file"]).read_bytes()
    base, tensors = int(expert["offset"]), expert["tensors"]
    matrices = {}
    total_squared = 0.0; total_values = 0; max_error = 0.0
    for name in ("gate", "up", "down"):
        exact = decoded_matrix(blob, base, tensors, name)
        slices, error = requantize(exact, block)
        matrices[name] = (slices, error)
        total_squared += error["rmse"] ** 2 * error["values"]
        total_values += error["values"]
        max_error = max(max_error, error["max_absolute_error"])
    model = ct.convert(build_program(matrices), convert_to="mlprogram",
                       minimum_deployment_target=ct.target.macOS15, compute_precision=ct.precision.FLOAT16)
    if output.exists(): shutil.rmtree(output)
    model.save(str(output))
    print(json.dumps({
        "schema": "ane-fieldfare.aot-authentic-requant-build.v1", "status": "accepted",
        "source": {"checkpoint": str(root), "layer": layer_index, "expert": expert_index},
        "quantization": {"bits": 4, "method": "independent per-output-channel minmax affine",
                         "target_block": block, "calibration": False, "training": False,
                         "iterative_fitting": False},
        "slice_counts": {name: len(matrices[name][0]) for name in matrices},
        "total_conv_slices": sum(len(matrices[name][0]) for name in matrices),
        "weight_error": {"rmse": (total_squared / total_values) ** 0.5,
                         "max_absolute_error": max_error,
                         "per_matrix": {name: matrices[name][1] for name in matrices}},
        "package_stored_bytes": directory_bytes(output), "compile_count": 0, "load_count": 0,
        "numerical_result": "not_run",
        "claim_boundary": "Deterministic training-free authentic-weight requantization only. ANE compile/load/evaluation and expert-output fidelity require the private gate."
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
