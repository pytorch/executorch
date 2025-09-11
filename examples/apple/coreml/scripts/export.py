# Copyright © 2023 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.

import argparse
import collections
import copy

import pathlib
import sys

import coremltools as ct

import executorch.exir as exir

import torch

# pyre-fixme[21]: Could not find module `executorch.backends.apple.coreml.compiler`.
from executorch.backends.apple.coreml.compiler import CoreMLBackend

# pyre-fixme[21]: Could not find module `executorch.backends.apple.coreml.partition`.
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.devtools.etrecord import generate_etrecord
from executorch.exir import to_edge

from executorch.exir.backend.backend_api import to_backend
from executorch.extension.export_util.utils import save_pte_program

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
sys.path.append(str(EXAMPLES_DIR.absolute()))

from executorch.examples.models import MODEL_NAME_TO_MODEL
from executorch.examples.models.model_factory import EagerModelFactory

# Script to export a model with coreml delegation.

_EDGE_COMPILE_CONFIG = exir.EdgeCompileConfig(
    _check_ir_validity=False,
    _skip_dim_order=True,  # TODO(T182928844): enable dim_order in backend
)


def is_fbcode():
    return not hasattr(torch.version, "git_version")


_CAN_RUN_WITH_PYBINDINGS = (sys.platform == "darwin") and not is_fbcode()
if _CAN_RUN_WITH_PYBINDINGS:
    from executorch.runtime import Runtime


def parse_args() -> argparse.Namespace:
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
    parser.add_argument(
        "--dynamic_shapes",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False,
    )
    parser.add_argument(
        "--run_with_pybindings",
        action=argparse.BooleanOptionalAction,
    )

    args = parser.parse_args()
    return args


def partition_module_to_coreml(module):
    module = module.eval()


def lower_module_to_coreml(module, compile_specs, example_inputs):
    module = module.eval()
    edge = to_edge(
        torch.export.export(module, example_inputs, strict=True),
        compile_config=_EDGE_COMPILE_CONFIG,
    )
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
        torch.export.export(lowered_module, example_inputs, strict=True),
        compile_config=_EDGE_COMPILE_CONFIG,
    ).to_executorch(config=exir.ExecutorchBackendConfig(extract_delegate_segments=True))

    return exec_prog


def get_pte_base_name(args: argparse.Namespace) -> str:
    pte_name = args.model_name
    if args.compile:
        pte_name += "_compiled"
    pte_name = f"{pte_name}_coreml_{args.compute_unit}"
    return pte_name


def save_processed_bytes(processed_bytes, base_name: str):
    filename = f"{base_name}.bin"
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


def run_with_pybindings(executorch_program, eager_reference, example_inputs, precision):
    if not _CAN_RUN_WITH_PYBINDINGS:
        raise RuntimeError("Cannot run with pybindings on this platform.")

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
    }[precision]

    runtime = Runtime.get()
    program = runtime.load_program(executorch_program.buffer)
    method = program.load_method("forward")
    et_outputs = method.execute(*example_inputs)[0]
    eager_outputs = eager_reference(*example_inputs)
    if isinstance(eager_outputs, collections.OrderedDict):
        eager_outputs = eager_outputs["out"]
    if isinstance(eager_outputs, list | tuple):
        eager_outputs = eager_outputs[0]

    mse = ((et_outputs - eager_outputs) ** 2).mean().sqrt()
    print(f"Mean square error: {mse}")
    assert mse < 0.1, "Mean square error is too high."

    if dtype == torch.float32:
        assert torch.allclose(
            et_outputs, eager_outputs, atol=1e-02, rtol=1e-02
        ), f"""Outputs do not match eager reference:
        \tet_outputs (first 5)={et_outputs.reshape(-1)[0:5]}
        \teager_outputs (first 5)={eager_outputs.reshape(-1)[0:5]}"""


def main():
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

    model, example_args, example_kwargs, dynamic_shapes = (
        EagerModelFactory.create_model(*MODEL_NAME_TO_MODEL[args.model_name])
    )
    if not args.dynamic_shapes:
        dynamic_shapes = None

    compile_specs = generate_compile_specs_from_args(args)
    pte_base_name = get_pte_base_name(args)
    if args.use_partitioner:
        model = model.eval()
        ep = torch.export.export(
            model,
            args=example_args,
            kwargs=example_kwargs,
            dynamic_shapes=dynamic_shapes,
        )
        print(ep)
        delegated_program = exir.to_edge_transform_and_lower(
            ep,
            partitioner=[CoreMLPartitioner(compile_specs=compile_specs)],
            generate_etrecord=args.generate_etrecord,
        )
        exec_program = delegated_program.to_executorch()
        save_pte_program(exec_program, pte_base_name)
        if args.generate_etrecord:
            exec_program.get_etrecord().save(f"{pte_base_name}_coreml_etrecord.bin")
        if args.run_with_pybindings:
            run_with_pybindings(
                executorch_program=exec_program,
                eager_reference=model,
                example_inputs=example_args,
                precision=args.compute_precision,
            )
    else:
        lowered_module, edge_copy = lower_module_to_coreml(
            module=model,
            example_inputs=example_args,
            compile_specs=compile_specs,
        )
        exec_program = export_lowered_module_to_executorch_program(
            lowered_module,
            example_args,
        )
        save_pte_program(exec_program, pte_base_name)
        if args.generate_etrecord:
            generate_etrecord(
                f"{args.model_name}_coreml_etrecord.bin", edge_copy, exec_program
            )

        if args.save_processed_bytes:
            save_processed_bytes(
                lowered_module.processed_bytes,
                pte_base_name,
            )
        if args.run_with_pybindings:
            run_with_pybindings(
                executorch_program=exec_program,
                eager_reference=model,
                example_inputs=example_args,
                precision=args.compute_precision,
            )


if __name__ == "__main__":
    main()
