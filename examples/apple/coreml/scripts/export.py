# Copyright © 2023 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.

import argparse
import copy

import pathlib
import sys

import coremltools as ct

import executorch.exir as exir

import torch

from executorch.backends.apple.coreml.compiler import CoreMLBackend

from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.devtools.etrecord import generate_etrecord
from executorch.exir import to_edge

from executorch.exir.backend.backend_api import to_backend
from torch.export import export

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
sys.path.append(str(EXAMPLES_DIR.absolute()))

from models import MODEL_NAME_TO_MODEL
from models.model_factory import EagerModelFactory

# Script to export a model with coreml delegation.

_EDGE_COMPILE_CONFIG = exir.EdgeCompileConfig(
    _check_ir_validity=False,
    _skip_dim_order=True,  # TODO(T182928844): enable dim_order in backend
)


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        help=f"Provide model name. Valid ones: {list(MODEL_NAME_TO_MODEL.keys())}",
    )

    parser.add_argument(
        "-c",
        "--compute_unit",
        required=False,
        default=ct.ComputeUnit.ALL.name.lower(),
        help=f"Provide compute unit for the model. Valid ones: {[[compute_unit.name.lower() for compute_unit in ct.ComputeUnit]]}",
    )

    parser.add_argument(
        "-precision",
        "--compute_precision",
        required=False,
        default=ct.precision.FLOAT16.value,
        help=f"Provide compute precision for the model. Valid ones: {[[precision.value for precision in ct.precision]]}",
    )

    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False,
    )
    parser.add_argument("--use_partitioner", action=argparse.BooleanOptionalAction)
    parser.add_argument("--generate_etrecord", action=argparse.BooleanOptionalAction)
    parser.add_argument("--save_processed_bytes", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    return args


def partition_module_to_coreml(module):
    module = module.eval()


def lower_module_to_coreml(module, compile_specs):
    module = module.eval()
    edge = to_edge(export(module, example_inputs), compile_config=_EDGE_COMPILE_CONFIG)
    # All of the subsequent calls on the edge_dialect_graph generated above (such as delegation or
    # to_executorch()) are done in place and the graph is also modified in place. For debugging purposes
    # we would like to keep a copy of the original edge dialect graph and hence we create a deepcopy of
    # it here that will later then be serialized into a etrecord.
    edge_copy = copy.deepcopy(edge)

    lowered_module = to_backend(
        CoreMLBackend.__name__,
        edge.exported_program(),
        compile_specs,
    )

    return lowered_module, edge_copy


def export_lowered_module_to_executorch_program(lowered_module, example_inputs):
    lowered_module(*example_inputs)
    exec_prog = to_edge(
        export(lowered_module, example_inputs), compile_config=_EDGE_COMPILE_CONFIG
    ).to_executorch(config=exir.ExecutorchBackendConfig(extract_delegate_segments=True))

    return exec_prog


def save_executorch_program(exec_prog, model_name, compute_unit):
    buffer = exec_prog.buffer
    filename = f"{model_name}_coreml_{compute_unit}.pte"
    print(f"Saving exported program to {filename}")
    with open(filename, "wb") as file:
        file.write(buffer)
    return


def save_processed_bytes(processed_bytes, model_name, compute_unit):
    filename = f"{model_name}_coreml_{compute_unit}.bin"
    print(f"Saving processed bytes to {filename}")
    with open(filename, "wb") as file:
        file.write(processed_bytes)
    return


def generate_compile_specs_from_args(args):
    model_type = CoreMLBackend.MODEL_TYPE.MODEL
    if args.compile:
        model_type = CoreMLBackend.MODEL_TYPE.COMPILED_MODEL

    compute_precision = ct.precision(args.compute_precision)
    compute_unit = ct.ComputeUnit[args.compute_unit.upper()]

    return CoreMLBackend.generate_compile_specs(
        compute_precision=compute_precision,
        compute_unit=compute_unit,
        model_type=model_type,
    )


if __name__ == "__main__":
    args = parse_args()

    if args.model_name not in MODEL_NAME_TO_MODEL:
        raise RuntimeError(
            f"Model {args.model_name} is not a valid name. "
            f"Available models are {list(MODEL_NAME_TO_MODEL.keys())}."
        )

    valid_compute_units = [compute_unit.name.lower() for compute_unit in ct.ComputeUnit]
    if args.compute_unit not in valid_compute_units:
        raise RuntimeError(
            f"{args.compute_unit} is invalid. "
            f"Valid compute units are {valid_compute_units}."
        )

    model, example_inputs, _ = EagerModelFactory.create_model(
        *MODEL_NAME_TO_MODEL[args.model_name]
    )

    compile_specs = generate_compile_specs_from_args(args)
    lowered_module = None

    if args.use_partitioner:
        model.eval()
        exir_program_aten = torch.export.export(model, example_inputs)
        edge_program_manager = exir.to_edge(exir_program_aten)
        edge_copy = copy.deepcopy(edge_program_manager)
        partitioner = CoreMLPartitioner(
            skip_ops_for_coreml_delegation=None, compile_specs=compile_specs
        )
        delegated_program_manager = edge_program_manager.to_backend(partitioner)
        exec_program = delegated_program_manager.to_executorch(
            config=exir.ExecutorchBackendConfig(extract_delegate_segments=True)
        )
    else:
        lowered_module, edge_copy = lower_module_to_coreml(
            module=model,
            compile_specs=compile_specs,
        )
        exec_program = export_lowered_module_to_executorch_program(
            lowered_module,
            example_inputs,
        )

    save_executorch_program(exec_program, args.model_name, args.compute_unit)
    generate_etrecord(f"{args.model_name}_coreml_etrecord.bin", edge_copy, exec_program)

    if args.save_processed_bytes and lowered_module is not None:
        save_processed_bytes(
            lowered_module.processed_bytes, args.model_name, args.compute_unit
        )
