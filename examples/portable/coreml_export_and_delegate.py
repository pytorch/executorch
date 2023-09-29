# Copyright © 2023 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.

import argparse

import executorch.exir as exir
import torch

from executorch.backends.coreml.compiler import CoreMLBackend
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.compile_spec_schema import CompileSpec

from ..models import MODEL_NAME_TO_MODEL
from ..models.model_factory import EagerModelFactory

# Script to export a model with coreml delegation.

_CAPTURE_CONFIG = exir.CaptureConfig(enable_aot=True, _unlift=False)
_EDGE_COMPILE_CONFIG = exir.EdgeCompileConfig(
    _check_ir_validity=False,
)


def export_to_executorch_program(model, example_inputs, compute_units):
    m = model.eval()
    edge = exir.capture(m, example_inputs, _CAPTURE_CONFIG).to_edge(
        _EDGE_COMPILE_CONFIG
    )

    lowered_module = to_backend(
        "CoreMLBackend",
        edge.exported_program,
        [CompileSpec("compute_units", bytes(compute_units, "utf-8"))],
    )

    class CompositeModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lowered_module = lowered_module

        def forward(self, *args):
            return self.lowered_module(*args)

    composite_model = CompositeModule()
    composite_model(*example_inputs)

    exec_prog = (
        exir.capture(composite_model, example_inputs, _CAPTURE_CONFIG)
        .to_edge(_EDGE_COMPILE_CONFIG)
        .to_executorch()
    )

    return exec_prog


def serialize_executorch_program(exec_prog, model_name, compute_units):
    buffer = exec_prog.buffer
    filename = f"{model_name}_{compute_units}.pte"
    print(f"Saving exported program to {filename}")
    with open(filename, "wb") as file:
        file.write(buffer)
    return


compute_units = ["cpu_only", "cpu_and_gpu", "cpu_and_ane", "all"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        help=f"Provide model name. Valid ones: {list(MODEL_NAME_TO_MODEL.keys())}",
    )

    parser.add_argument(
        "-c",
        "--compute_units",
        required=False,
        default="all",
        help=f"Provide compute units. Valid ones: {compute_units}",
    )

    args = parser.parse_args()

    if args.model_name not in MODEL_NAME_TO_MODEL:
        raise RuntimeError(
            f"Model {args.model_name} is not a valid name. "
            f"Available models are {list(MODEL_NAME_TO_MODEL.keys())}."
        )

    if args.compute_units not in compute_units:
        raise RuntimeError(
            f"{args.compute_units} is invalid. "
            f"Valid compute units are {compute_units}."
        )

    model, example_inputs = EagerModelFactory.create_model(
        *MODEL_NAME_TO_MODEL[args.model_name]
    )

    exec_program = export_to_executorch_program(
        model, example_inputs, args.compute_units
    )
    serialize_executorch_program(exec_program, args.model_name, args.compute_units)
