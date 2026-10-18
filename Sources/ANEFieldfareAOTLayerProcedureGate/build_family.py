#!/usr/bin/env python3
"""Combine exact per-expert packages into one named procedure family."""
from pathlib import Path
import json, shutil, sys
import coremltools as ct

if len(sys.argv) not in (4, 5):
    raise SystemExit("usage: build_family.py <count> <package-output> <package-root> [first-expert]")
count, output, root = int(sys.argv[1]), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve()
first = int(sys.argv[4]) if len(sys.argv) == 5 else 0
prior = Path(".build/aot-block-slice").resolve()
descriptor = ct.utils.MultiFunctionDescriptor()
for expert in range(first, first + count):
    package = (prior if expert < 8 else root) / f"expert-l0-e{expert}.mlpackage"
    if not package.exists(): raise FileNotFoundError(package)
    descriptor.add_function(str(package), "main", f"expert_{expert}")
descriptor.default_function_name = f"expert_{first}"
if output.exists(): shutil.rmtree(output)
ct.utils.save_multifunction(descriptor, str(output))
print(json.dumps({"schema":"ane-fieldfare.aot-layer-procedure-family-build.v1",
    "status":"accepted","procedure_count":count,"first_expert":first,"package":str(output),
    "coremltools_version":ct.__version__,"compile_count":0,"load_count":0},indent=2,sort_keys=True))
