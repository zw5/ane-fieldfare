#!/usr/bin/env python3
"""Build C1's Gemma-shaped, blockwise-int4 AOT compiler input.

Core ML is used only to serialize the official constexpr representation.  This
script never creates MLModel or calls predict; those operations may fall back
and are not evidence of ANE execution.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    OptimizationConfig,
    linear_quantize_weights,
)
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".build" / "aot-compressed-projection"
HIDDEN = 2816
INTERMEDIATE = 704
POSITIONS = 64
GROUP = 64
SCALES = {4: np.float32(1.0 / 1024.0), 8: np.float32(1.0 / 16384.0)}


def lattice_weights(bits: int) -> np.ndarray:
    """Known symmetric lattice, with extrema in every 64-value block."""
    row = np.arange(INTERMEDIATE, dtype=np.int32)[:, None]
    column = np.arange(HIDDEN, dtype=np.int32)[None, :]
    maximum = 7 if bits == 4 else 127
    modulus = maximum * 2 + 1
    # Every 64-wide block includes both extrema. The selected scales are exact
    # in FP16 and BF16, making a separately constructed affine oracle possible.
    if bits == 4:
        codes = ((row * 5 + column * 3) % modulus) - maximum
    else:
        # A 64-wide block cannot cover all 255 codes, so explicitly seed its
        # first two columns with the extrema that determine the quantizer scale.
        codes = ((row * 29 + column * 17) % modulus) - maximum
        block_column = column % GROUP
        codes = np.where(block_column == 0, -maximum, codes)
        codes = np.where(block_column == 1, maximum, codes)
    return (codes.astype(np.float32) * SCALES[bits]).reshape(INTERMEDIATE, HIDDEN, 1, 1)


def program_for(weight: np.ndarray):
    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, HIDDEN, 1, POSITIONS), dtype=types.fp16)],
        opset_version=ct.target.iOS18,
    )
    def program(x):
        return mb.conv(x=x, weight=weight, name="projection")

    return program


def package_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    variants = []
    for bits, block_size, label in (
        (4, GROUP, "int4-block64"),
        (4, HIDDEN, "int4-per-channel"),
        (8, GROUP, "int8-block64"),
        (8, HIDDEN, "int8-per-channel"),
    ):
        weight = lattice_weights(bits)
        fp16_source_bytes = weight.size * 2
        model = ct.convert(
            program_for(weight),
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16,
        )
        config = OptimizationConfig(global_config=OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype=f"int{bits}",
            granularity="per_block",
            block_size=[1, block_size, 1, 1],
            weight_threshold=0,
        ))
        compressed = linear_quantize_weights(model, config=config)
        package = OUT / f"projection-{label}.mlpackage"
        compressed.save(str(package))
        variants.append({
            "bits": bits,
            "block_size": block_size,
            "label": label,
            "package": str(package),
            "source_fp16_weight_bytes": fp16_source_bytes,
            "package_stored_bytes": package_bytes(package),
        })

    receipt = {
        "schema": "ane-fieldfare.aot-compressed-projection-build.v1",
        "status": "accepted",
        "stage": "compiler_input_serialized",
        "coremltools_version": ct.__version__,
        "variants": variants,
        "shape": {
            "input_channels": HIDDEN,
            "output_channels": INTERMEDIATE,
            "positions": POSITIONS,
            "block_size": GROUP,
        },
        "quantization": "symmetric_int4/int8 block64 plus int8 per-channel control",
        "compile_count": 0,
        "load_count": 0,
        "numerical_result": "not_run",
        "claim_boundary": (
            "Official package serialization only. This is not AOT compilation, "
            "ANE assignment, private-ANE loading, execution, fidelity, bandwidth, "
            "or residency evidence."
        ),
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
