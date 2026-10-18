#!/usr/bin/env python3
"""Combine an authentic layer's expert packages into one multifunction artifact."""

import json
from pathlib import Path
import shutil
import sys

import coremltools as ct


def main() -> int:
    if len(sys.argv) not in (5, 6):
        raise SystemExit(
            "usage: combine_layer.py <artifact-directory> <layer> <expert-count> "
            "<output.mlpackage> [expert-start]"
        )
    root = Path(sys.argv[1]).resolve()
    layer = int(sys.argv[2])
    count = int(sys.argv[3])
    output = Path(sys.argv[4]).resolve()
    start = int(sys.argv[5]) if len(sys.argv) == 6 else 0
    if not 1 <= count <= 128 or start < 0 or start + count > 128:
        raise ValueError("expert range must be contained in 0...127")

    descriptor = ct.utils.MultiFunctionDescriptor()
    names = []
    for expert in range(start, start + count):
        package = root / f"expert-l{layer}-e{expert}.mlpackage"
        if not package.is_dir():
            raise FileNotFoundError(package)
        name = f"expert_{expert}"
        descriptor.add_function(str(package), "main", name)
        names.append(name)
    descriptor.default_function_name = names[0]
    if output.exists():
        shutil.rmtree(output)
    ct.utils.save_multifunction(descriptor, str(output))
    print(json.dumps({
        "schema": "ane-fieldfare.aot-block-slice-layer-function-build.v1",
        "status": "accepted",
        "layer": layer,
        "expert_start": start,
        "function_count": count,
        "functions": names,
        "package": str(output),
        "coremltools_version": ct.__version__,
        "compile_count": 0,
        "load_count": 0,
        "claim_boundary": (
            "Compiler-authored multifunction package only; private compilation, loading, "
            "procedure selection, residency, and numerical execution require the live gate."
        ),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
