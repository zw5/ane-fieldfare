#!/usr/bin/env python3
"""Build a two-function compiler input for private ANE artifact probing.

Core ML is only the closed compiler entry point here.  This script intentionally
does not execute MLModel.predict: public runtime success can fall back and is not
an ANE hardware or residency oracle.
"""

from pathlib import Path
import json
import shutil

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".build" / "multifunction-probe"


def make_program(scale: float):
    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, 64), dtype=types.fp16)],
        opset_version=ct.target.iOS18,
    )
    def program(x):
        return mb.mul(x=x, y=np.float16(scale), name="output")

    return program


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    source_paths = []
    for name, scale in (("times_two", 2.0), ("times_three", 3.0)):
        model = ct.convert(
            make_program(scale),
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16,
        )
        path = OUT / f"{name}.mlpackage"
        model.save(str(path))
        source_paths.append(path)

    descriptor = ct.utils.MultiFunctionDescriptor()
    descriptor.add_function(str(source_paths[0]), "main", "times_two")
    descriptor.add_function(str(source_paths[1]), "main", "times_three")
    descriptor.default_function_name = "times_two"
    combined = OUT / "combined.mlpackage"
    ct.utils.save_multifunction(descriptor, str(combined))

    receipt = {
        "schema": "ane-fieldfare.coreml-multifunction.v1",
        "status": "accepted",
        "coremltools_version": ct.__version__,
        "package": str(combined),
        "functions": ["times_two", "times_three"],
        "compile_count": 0,
        "load_count": 0,
        "numerical_result": "not_run",
        "claim_boundary": (
            "Two functions were packaged as compiler input only. This is not ANE execution, device assignment, "
            "a multiprocedure HWX, or residency evidence. Validation requires one inspected multiprocedure/model.hwx, "
            "private-ANE loading, distinct per-procedure numerical oracles, and a residency-slot count."
        ),
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
