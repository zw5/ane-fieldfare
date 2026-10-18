#!/usr/bin/env python3
"""Build the compiler-authored Core ML state recurrence used by the state gate."""

from pathlib import Path
import shutil

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / ".build" / "state-recurrence"
PACKAGE = OUT / "state-recurrence.mlpackage"


@mb.program(
    input_specs=[
        mb.TensorSpec(shape=(1, 256, 1, 64), dtype=types.fp16),
        mb.StateTensorSpec(shape=(1, 256, 1, 64), dtype=types.fp16),
    ],
    opset_version=ct.target.iOS18,
)
def recurrence(x, cache):
    previous = mb.read_state(input=cache, name="read_cache")
    weight = np.eye(256, dtype=np.float16).reshape(256, 256, 1, 1)
    projected = mb.conv(x=x, weight=weight, name="identity_projection")
    updated = mb.add(x=previous, y=projected, name="y")
    mb.coreml_update_state(state=cache, value=updated, name="write_cache")
    return updated


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    model = ct.convert(
        recurrence,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
    )
    model.author = "ANE Fieldfare"
    model.short_description = "Compiler-authored FP16 state recurrence: cache <- cache + x"
    model.save(str(PACKAGE))
    print(PACKAGE)


if __name__ == "__main__":
    main()
