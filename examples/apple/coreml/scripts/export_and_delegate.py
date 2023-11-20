# Copyright © 2023 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.

import argparse

import pathlib
import sys

import executorch.exir as exir

from executorch.backends.apple.coreml.compiler import CoreMLBackend

from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.compile_spec_schema import CompileSpec

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
sys.path.append(str(EXAMPLES_DIR.absolute()))

from models import MODEL_NAME_TO_MODEL
from models.model_factory import EagerModelFactory

# Script to export a model with coreml delegation.

_CAPTURE_CONFIG = exir.CaptureConfig(enable_aot=True, _unlift=False)
_EDGE_COMPILE_CONFIG = exir.EdgeCompileConfig(
    _check_ir_validity=False,
)


def lower_module_to_coreml(module, compute_units):
    module = module.eval()
    edge = exir.capture(module, example_inputs, _CAPTURE_CONFIG).to_edge(
        _EDGE_COMPILE_CONFIG
    )

    lowered_module = to_backend(
        CoreMLBackend.__name__,
        edge.exported_program,
        [CompileSpec("compute_units", bytes(compute_units, "utf-8"))],
    )

    return lowered_module


def export_lowered_module_to_executorch_program(lowered_module, example_inputs):
    lowered_module(*example_inputs)
    exec_prog = (
        exir.capture(lowered_module, example_inputs, _CAPTURE_CONFIG)
        .to_edge(_EDGE_COMPILE_CONFIG)
        .to_executorch()
    )

    return exec_prog


def save_executorch_program(exec_prog, model_name, compute_units):
    buffer = exec_prog.buffer
    filename = f"{model_name}_coreml_{compute_units}.pte"
    print(f"Saving exported program to {filename}")
    with open(filename, "wb") as file:
        file.write(buffer)
    return


def save_processed_bytes(processed_bytes, model_name, compute_units):
    filename = f"{model_name}_coreml_{compute_units}.bin"
    print(f"Saving processed bytes to {filename}")
    with open(filename, "wb") as file:
        file.write(processed_bytes)
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

    parser.add_argument("--save_processed_bytes", action=argparse.BooleanOptionalAction)

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

    lowered_module = lower_module_to_coreml(
        model,
        args.compute_units,
    )

    exec_program = export_lowered_module_to_executorch_program(
        lowered_module,
        example_inputs,
    )

    save_executorch_program(exec_program, args.model_name, args.compute_units)

    if args.save_processed_bytes:
        save_processed_bytes(
            lowered_module.processed_bytes, args.model_name, args.compute_units
        )
