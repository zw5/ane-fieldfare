#!/usr/bin/env python3
"""Build one exact authentic expert from compiler-authoritative int4 slices."""

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


HIDDEN, INTERMEDIATE, GROUP, POSITIONS = 2816, 704, 64, 64


def unpack(raw: bytes, rows: int, columns: int) -> np.ndarray:
    packed = np.frombuffer(raw, dtype=np.uint8)
    codes = np.empty(packed.size * 2, dtype=np.uint8)
    codes[0::2], codes[1::2] = packed & 15, packed >> 4
    return codes.reshape(rows, columns)


def matrix(blob: bytes, base: int, tensors: dict, name: str):
    record = tensors[name]
    rows, columns = map(int, record["shape"])
    groups = columns // GROUP
    code_raw = checked_slice(blob, base, record)
    scale_raw = checked_slice(blob, base, tensors[name + "_scales"])
    bias_raw = checked_slice(blob, base, tensors[name + "_biases"])
    return {
        "codes": unpack(code_raw, rows, columns),
        "scales": bf16_array(scale_raw).reshape(rows, groups),
        "biases": bf16_array(bias_raw).reshape(rows, groups),
        "rows": rows, "columns": columns,
        "bytes": len(code_raw) + len(scale_raw) + len(bias_raw),
    }


def build_program(gate: dict, up: dict, down: dict):
    def project(x, value: dict, prefix: str):
        rows, columns = value["rows"], value["columns"]
        partials = []
        for group in range(columns // GROUP):
            begin = group * GROUP
            xg = mb.slice_by_size(x=x, begin=[0, begin, 0, 0],
                                  size=[1, GROUP, 1, POSITIONS], name=f"{prefix}_x_{group}")
            q = value["codes"][:, begin:begin + GROUP, None, None].astype(types.np_uint4_dtype)
            s = value["scales"][:, group].reshape(rows, 1, 1, 1).astype(np.float16)
            w = mb.constexpr_blockwise_shift_scale(data=q, scale=s, name=f"{prefix}_w_{group}")
            qy = mb.conv(x=xg, weight=w, name=f"{prefix}_q_{group}")
            sx = mb.reduce_sum(x=xg, axes=[1], keep_dims=True, name=f"{prefix}_sx_{group}")
            b = value["biases"][:, group].reshape(1, rows, 1, 1).astype(np.float16)
            partials.append(mb.add(x=qy, y=mb.mul(x=sx, y=b, name=f"{prefix}_b_{group}"),
                                   name=f"{prefix}_p_{group}"))
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
        g = project(x, gate, "gate")
        u = project(x, up, "up")
        g2 = mb.mul(x=g, y=g, name="gate_square")
        g3 = mb.mul(x=g2, y=g, name="gate_cube")
        inner = mb.add(x=g, y=mb.mul(x=g3, y=np.float16(0.044715), name="gelu_cubic"),
                       name="gelu_inner")
        tanh = mb.tanh(x=mb.mul(x=inner, y=np.float16(0.7978845608028654),
                               name="gelu_scaled"), name="gelu_tanh")
        gelu = mb.mul(x=mb.mul(x=g, y=np.float16(0.5), name="gelu_half_gate"),
                      y=mb.add(x=tanh, y=np.float16(1.0), name="gelu_plus"), name="gelu")
        activated = mb.mul(x=gelu, y=u, name="activated")
        return mb.cast(x=project(activated, down, "down"), dtype="fp32", name="expert")

    return program


def main() -> int:
    if len(sys.argv) != 5:
        raise SystemExit("usage: build_full_expert.py <gturbo-root> <layer> <expert> <output.mlpackage>")
    root, layer_index, expert_index, output = Path(sys.argv[1]).resolve(), int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4]).resolve()
    layout = json.loads((root / "packed_experts/layout.json").read_text())
    layer = next(x for x in layout["layers"] if int(x["layer"]) == layer_index)
    expert = next(x for x in layer["experts"] if int(x["expert"]) == expert_index)
    blob = (root / "packed_experts" / layer["file"]).read_bytes()
    base, tensors = int(expert["offset"]), expert["tensors"]
    gate, up, down = (matrix(blob, base, tensors, name) for name in ("gate", "up", "down"))
    if (gate["rows"], gate["columns"], up["rows"], up["columns"], down["rows"], down["columns"]) != (704, 2816, 704, 2816, 2816, 704):
        raise ValueError("expert shape mismatch")
    model = ct.convert(build_program(gate, up, down), convert_to="mlprogram",
                       minimum_deployment_target=ct.target.macOS15, compute_precision=ct.precision.FLOAT16)
    if output.exists(): shutil.rmtree(output)
    model.save(str(output))
    exact_bytes = gate["bytes"] + up["bytes"] + down["bytes"]
    print(json.dumps({
        "schema": "ane-fieldfare.aot-block-slice-full-expert-build.v1", "status": "accepted",
        "source": {"checkpoint": str(root), "layer": layer_index, "expert": expert_index},
        "shape": {"hidden": HIDDEN, "intermediate": INTERMEDIATE, "positions": POSITIONS,
                  "group_size": GROUP, "int4_slice_count": 99},
        "checkpoint_expert_bytes": exact_bytes, "package_stored_bytes": directory_bytes(output),
        "fp16_weight_equivalent_bytes": 2 * (2 * HIDDEN * INTERMEDIATE + INTERMEDIATE * HIDDEN),
        "preservation": "Every packed uint4 code and BF16 scale/bias from gate, up, and down is copied deterministically; no fitting or training.",
        "compile_count": 0, "load_count": 0, "numerical_result": "not_run",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
