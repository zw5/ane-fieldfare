#!/usr/bin/env python3
"""Combine eight exact expert packages into one compiler-authored artifact."""

import json
from pathlib import Path
import shutil
import sys

import coremltools as ct


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: combine_eight.py <artifact-directory> <output.mlpackage>")
    root, output = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    descriptor = ct.utils.MultiFunctionDescriptor()
    names = []
    for expert in range(8):
        name = f"expert_{expert}"
        descriptor.add_function(str(root / f"expert-l0-e{expert}.mlpackage"), "main", name)
        names.append(name)
    descriptor.default_function_name = names[0]
    if output.exists(): shutil.rmtree(output)
    ct.utils.save_multifunction(descriptor, str(output))
    print(json.dumps({"schema": "ane-fieldfare.aot-block-slice-eight-function-build.v1",
        "status": "accepted", "functions": names, "package": str(output),
        "coremltools_version": ct.__version__, "compile_count": 0, "load_count": 0,
        "claim_boundary": "Compiler-authored multifunction package only; private loading, procedure IDs, execution, sharing, and residency are not yet established."}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
