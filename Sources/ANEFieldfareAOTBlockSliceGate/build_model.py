#!/usr/bin/env python3
"""Build an exact authentic block-64 expert projection as 44 AOT int4 slices."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import struct
import sys

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
import numpy as np


HIDDEN = 2816
ROWS = 704
GROUP = 64
GROUPS = HIDDEN // GROUP
POSITIONS = 64


def bf16_array(raw: bytes) -> np.ndarray:
    words = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    return (words << 16).view(np.float32)


def checked_slice(blob: bytes, base: int, record: dict) -> bytes:
    start = base + int(record["offset"])
    end = start + int(record["size"])
    if start < 0 or end > len(blob):
        raise ValueError(f"tensor range [{start}, {end}) exceeds {len(blob)} bytes")
    return blob[start:end]


def unpack_codes(raw: bytes) -> np.ndarray:
    packed = np.frombuffer(raw, dtype=np.uint8)
    codes = np.empty(packed.size * 2, dtype=np.uint8)
    codes[0::2] = packed & 0x0F
    codes[1::2] = packed >> 4
    return codes.reshape(ROWS, HIDDEN)


def build_program(codes: np.ndarray, scales: np.ndarray, biases: np.ndarray):
    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, HIDDEN, 1, POSITIONS), dtype=types.fp16)],
        opset_version=ct.target.iOS18,
    )
    def program(x):
        partials = []
        for group in range(GROUPS):
            begin = group * GROUP
            x_slice = mb.slice_by_size(
                x=x, begin=[0, begin, 0, 0], size=[1, GROUP, 1, POSITIONS],
                name=f"input_slice_{group}",
            )
            group_codes = codes[:, begin : begin + GROUP, None, None].astype(
                types.np_uint4_dtype
            )
            group_scales = scales[:, group].reshape(ROWS, 1, 1, 1).astype(np.float16)
            weight = mb.constexpr_blockwise_shift_scale(
                data=group_codes, scale=group_scales, name=f"weight_group_{group}"
            )
            quantized = mb.conv(x=x_slice, weight=weight, name=f"quantized_group_{group}")
            channel_sum = mb.reduce_sum(
                x=x_slice, axes=[1], keep_dims=True, name=f"input_sum_{group}"
            )
            correction = mb.mul(
                x=channel_sum,
                y=biases[:, group].reshape(1, ROWS, 1, 1).astype(np.float16),
                name=f"bias_correction_{group}",
            )
            partials.append(mb.add(x=quantized, y=correction, name=f"partial_{group}"))
        while len(partials) > 1:
            next_level = []
            for index in range(0, len(partials), 2):
                next_level.append(
                    partials[index]
                    if index + 1 == len(partials)
                    else mb.add(x=partials[index], y=partials[index + 1],
                                name=f"sum_{len(partials)}_{index // 2}")
                )
            partials = next_level
        return mb.cast(x=partials[0], dtype="fp32", name="checkpoint_gate")

    return program


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> int:
    if len(sys.argv) != 5:
        raise SystemExit("usage: build_model.py <gturbo-root> <layer> <expert> <output.mlpackage>")
    root = Path(sys.argv[1]).resolve()
    layer_index, expert_index = int(sys.argv[2]), int(sys.argv[3])
    output = Path(sys.argv[4]).resolve()
    layout = json.loads((root / "packed_experts/layout.json").read_text())
    layer = next(item for item in layout["layers"] if int(item["layer"]) == layer_index)
    expert = next(item for item in layer["experts"] if int(item["expert"]) == expert_index)
    blob = (root / "packed_experts" / layer["file"]).read_bytes()
    tensors = expert["tensors"]
    gate = tensors["gate"]
    if gate["shape"] != [ROWS, HIDDEN] or gate["bits"] != 4:
        raise ValueError(f"unexpected gate tensor: {gate}")
    base = int(expert["offset"])
    code_bytes = checked_slice(blob, base, gate)
    scale_bytes = checked_slice(blob, base, tensors["gate_scales"])
    bias_bytes = checked_slice(blob, base, tensors["gate_biases"])
    codes = unpack_codes(code_bytes)
    scales = bf16_array(scale_bytes).reshape(ROWS, GROUPS)
    biases = bf16_array(bias_bytes).reshape(ROWS, GROUPS)

    model = ct.convert(
        build_program(codes, scales, biases), convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
    )
    if output.exists():
        shutil.rmtree(output)
    model.save(str(output))
    print(json.dumps({
        "schema": "ane-fieldfare.aot-block-slice-build.v1",
        "status": "accepted",
        "coremltools_version": ct.__version__,
        "source": {"root": str(root), "layer": layer_index, "expert": expert_index,
                   "projection": "gate"},
        "shape": {"rows": ROWS, "columns": HIDDEN, "positions": POSITIONS,
                  "groups": GROUPS, "group_size": GROUP},
        "construction": "44 per-output-channel uint4 constexpr convolutions over 64-input slices; each partial adds the checkpoint BF16 affine bias times the slice input sum; balanced explicit sum",
        "checkpoint_bytes": {"codes": len(code_bytes), "scales": len(scale_bytes),
                             "biases": len(bias_bytes),
                             "total": len(code_bytes) + len(scale_bytes) + len(bias_bytes)},
        "package_stored_bytes": directory_bytes(output),
        "preservation": "Every packed uint4 code and every BF16 scale/bias value is copied deterministically; no fitting or training.",
        "compile_count": 0,
        "load_count": 0,
        "numerical_result": "not_run",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
