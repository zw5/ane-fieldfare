#!/usr/bin/env python3
"""Build the smallest public Core ML updatable affine model.

This is a compiler-authored control for the private mutable-procedure contract.
It deliberately uses Core ML's documented updatable-weight declaration rather
than inserting private manifest keys or editing compiled artifacts.
"""

import json
import pathlib
import sys
import tempfile

import coremltools as ct
import numpy as np
from coremltools.models.compute_plan import MLComputePlan
from coremltools.models import datatypes
from coremltools.models.neural_network import NeuralNetworkBuilder
from coremltools.models.neural_network.update_optimizer_utils import SgdParams


def main() -> int:
    destination = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else pathlib.Path(
        tempfile.mkdtemp(prefix="ane-fieldfare-updatable-")
    )
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / "updatable.mlmodel"
    compiled_path = destination / "updatable.mlmodelc"

    features = [("activation", datatypes.Array(64))]
    outputs = [("prediction", datatypes.Array(64))]
    builder = NeuralNetworkBuilder(features, outputs)
    weights = np.eye(64, dtype=np.float32) * 2.0
    builder.add_inner_product(
        name="streamed_weight",
        W=weights,
        b=None,
        input_channels=64,
        output_channels=64,
        has_bias=False,
        input_name="activation",
        output_name="prediction",
    )
    builder.make_updatable(["streamed_weight"])
    builder.set_mean_squared_error_loss(
        name="mse", input_feature=("prediction", datatypes.Array(64))
    )
    builder.set_sgd_optimizer(SgdParams(lr=0.01, batch=1, momentum=0.0))
    builder.set_epochs(1)
    ct.models.utils.save_spec(builder.spec, str(model_path))
    ct.models.utils.compile_model(str(model_path), destination_path=str(compiled_path))

    compute_plan = MLComputePlan.load_from_path(str(compiled_path))
    layer = compute_plan.model_structure.neuralnetwork.layers[0]
    usage = compute_plan.get_compute_device_usage_for_neuralnetwork_layer(layer=layer)
    preferred_device = type(usage.preferred_compute_device).__name__
    supported_devices = [type(device).__name__ for device in usage.supported_compute_devices]

    files = []
    for path in sorted(compiled_path.rglob("*")):
        if path.is_file():
            files.append({"path": str(path.relative_to(compiled_path)), "bytes": path.stat().st_size})
    print(json.dumps({
        "schema": "ane-fieldfare.public-updatable-compiler-control.v1",
        "coremltools_version": ct.__version__,
        "source_model": str(model_path),
        "compiled_model": str(compiled_path),
        "specification_version": builder.spec.specificationVersion,
        "is_updatable": builder.spec.isUpdatable,
        "layer_is_updatable": builder.spec.neuralNetwork.layers[0].isUpdatable,
        "compute_plan": {
            "layer": layer.name,
            "preferred_device": preferred_device,
            "supported_devices": supported_devices,
        },
        "files": files,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
