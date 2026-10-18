#!/usr/bin/env python3
"""Plan or package up to 96 authentic experts into one multifunction artifact.

The dry-run path reads only checkpoint metadata and file sizes.  It never opens
the expert blobs, decodes uint4 codes, constructs FP16 weights, invokes
coremltools, compiles, loads, or evaluates a model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MAX_PROCEDURES = 96
MATRIX_NAMES = ("gate", "up", "down")
TENSOR_NAMES = tuple(
    name
    for matrix in MATRIX_NAMES
    for name in (matrix, f"{matrix}_scales", f"{matrix}_biases")
)


def load_identities(path: Path) -> list[tuple[int, int]]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError("identity file must be a JSON list")
    identities: list[tuple[int, int]] = []
    for index, item in enumerate(raw):
        if isinstance(item, dict) and set(item) == {"layer", "expert"}:
            layer, expert = item["layer"], item["expert"]
        elif isinstance(item, list) and len(item) == 2:
            layer, expert = item
        else:
            raise ValueError(f"identity {index} must be [layer, expert] or an exact layer/expert object")
        if isinstance(layer, bool) or isinstance(expert, bool):
            raise ValueError(f"identity {index} contains a boolean index")
        identities.append((int(layer), int(expert)))
    if not identities:
        raise ValueError("at least one identity is required")
    if len(identities) > MAX_PROCEDURES:
        raise ValueError(f"procedure count {len(identities)} exceeds the {MAX_PROCEDURES}-procedure cap")
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate layer/expert identity")
    if any(layer < 0 or expert < 0 for layer, expert in identities):
        raise ValueError("layer and expert indices must be nonnegative")
    return identities


def procedure_name(layer: int, expert: int) -> str:
    return f"expert_l{layer:02d}_e{expert:03d}"


def build_manifest(checkpoint: Path, identities: list[tuple[int, int]]) -> dict:
    layout_path = checkpoint / "packed_experts/layout.json"
    layout = json.loads(layout_path.read_text())
    layers = {int(item["layer"]): item for item in layout["layers"]}
    procedures = []
    tensor_payload_total = 0
    expert_span_total = 0
    fp16_total = 0

    for ordinal, (layer_index, expert_index) in enumerate(identities):
        if layer_index not in layers:
            raise ValueError(f"layer {layer_index} is absent from checkpoint layout")
        layer = layers[layer_index]
        experts = {int(item["expert"]): item for item in layer["experts"]}
        if expert_index not in experts:
            raise ValueError(f"expert {expert_index} is absent from layer {layer_index}")
        expert = experts[expert_index]
        blob_path = checkpoint / "packed_experts" / layer["file"]
        blob_bytes = blob_path.stat().st_size
        expert_base = int(expert["offset"])
        expert_size = int(expert["size"])
        if expert_base < 0 or expert_base + expert_size > blob_bytes:
            raise ValueError(f"expert l{layer_index} e{expert_index} span exceeds {layer['file']}")

        mappings = []
        payload_bytes = 0
        fp16_bytes = 0
        for tensor_name in TENSOR_NAMES:
            record = expert["tensors"].get(tensor_name)
            if record is None:
                raise ValueError(f"missing tensor {tensor_name} for l{layer_index} e{expert_index}")
            relative_offset, size = int(record["offset"]), int(record["size"])
            if relative_offset < 0 or size < 0 or relative_offset + size > expert_size:
                raise ValueError(f"tensor {tensor_name} exceeds expert l{layer_index} e{expert_index}")
            shape = [int(value) for value in record["shape"]]
            if tensor_name in MATRIX_NAMES:
                if int(record.get("bits", 0)) != 4:
                    raise ValueError(f"tensor {tensor_name} is not authentic uint4 storage")
                matrix_fp16_bytes = 2
                for dimension in shape:
                    matrix_fp16_bytes *= dimension
                fp16_bytes += matrix_fp16_bytes
            mappings.append({
                "name": tensor_name,
                "representation": "uint4_codes" if tensor_name in MATRIX_NAMES else "bf16_affine",
                "shape": shape,
                "relative_offset": relative_offset,
                "absolute_offset": expert_base + relative_offset,
                "stored_bytes": size,
            })
            payload_bytes += size

        tensor_payload_total += payload_bytes
        expert_span_total += expert_size
        fp16_total += fp16_bytes
        procedures.append({
            "ordinal": ordinal,
            "procedure": procedure_name(layer_index, expert_index),
            "oracle_identity": {"layer": layer_index, "expert": expert_index},
            "source": {
                "blob": str(blob_path),
                "blob_bytes": blob_bytes,
                "expert_offset": expert_base,
                "expert_span_bytes": expert_size,
                "tensor_payload_bytes": payload_bytes,
                "tensors": mappings,
            },
        })

    return {
        "schema": "ane-fieldfare.aot-cross-layer-package-plan.v1",
        "status": "present",
        "mode": "dry_run",
        "checkpoint": str(checkpoint),
        "procedure_cap": MAX_PROCEDURES,
        "procedure_count": len(procedures),
        "default_procedure": procedures[0]["procedure"],
        "procedures": procedures,
        "stored_byte_arithmetic": {
            "checkpoint_tensor_payload_bytes": tensor_payload_total,
            "checkpoint_expert_spans_bytes": expert_span_total,
            "checkpoint_padding_bytes": expert_span_total - tensor_payload_total,
            "fp16_matrix_equivalent_bytes": fp16_total,
            "formula": "sum selected uint4 code bytes plus BF16 scale/bias bytes; expert spans additionally include checkpoint padding",
            "combined_mlpackage_bytes": "not_known_until_packaging",
            "compiled_artifact_bytes": "not_known_until_compilation",
        },
        "compile_count": 0,
        "load_count": 0,
        "numerical_result": "not_run",
        "claim_boundary": (
            "Metadata-validated deterministic cross-layer procedure plan only. No tensor was "
            "decoded or materialized as FP16. A 40-handle full-model topology is an inference "
            "until a compiled cross-layer artifact passes independent numerical and lifecycle oracles."
        ),
    }


def package(manifest: dict, package_root: Path, output: Path) -> None:
    import coremltools as ct

    descriptor = ct.utils.MultiFunctionDescriptor()
    for procedure in manifest["procedures"]:
        identity = procedure["oracle_identity"]
        source = package_root / f"expert-l{identity['layer']}-e{identity['expert']}.mlpackage"
        if not source.is_dir():
            raise FileNotFoundError(source)
        descriptor.add_function(str(source), "main", procedure["procedure"])
    descriptor.default_function_name = manifest["default_procedure"]
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    ct.utils.save_multifunction(descriptor, str(output))
    manifest["mode"] = "package_only"
    manifest["status"] = "accepted"
    manifest["package"] = str(output)
    manifest["stored_byte_arithmetic"]["combined_mlpackage_bytes"] = sum(
        item.stat().st_size for item in output.rglob("*") if item.is_file()
    )
    manifest["claim_boundary"] = (
        "Compiler-authored multifunction package only. It has not been compiled, loaded, "
        "executed, or numerically validated; the 40-handle full-model topology remains an inference."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("identities", type=Path, help="ordered JSON list of layer/expert identities")
    parser.add_argument("--dry-run", action="store_true", help="emit metadata-only plan (default)")
    parser.add_argument("--package-root", type=Path, help="directory of prebuilt authentic expert packages")
    parser.add_argument("--output", type=Path, help="output multifunction .mlpackage")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    manifest = build_manifest(checkpoint, load_identities(args.identities.resolve()))
    if args.package_root is not None or args.output is not None:
        if args.dry_run:
            parser.error("--dry-run cannot be combined with packaging arguments")
        if args.package_root is None or args.output is None:
            parser.error("--package-root and --output are required together")
        package(manifest, args.package_root.resolve(), args.output.resolve())
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
