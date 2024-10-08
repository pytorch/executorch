# Copyright © 2024 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.

import argparse
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import coremltools as ct
from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.exir import EdgeProgramManager

from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.tracer import Value
from tabulate import tabulate


def get_root_dir_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent.parent


sys.path.append(str((get_root_dir_path() / "examples").resolve()))

from inspector_utils import (
    build_devtools_runner_including_coreml,
    ComparisonResult,
    create_inspector_coreml,
    create_inspector_reference,
    get_comparison_result,
    module_to_edge,
)

from models import MODEL_NAME_TO_MODEL
from models.model_factory import EagerModelFactory


def args_parser() -> argparse.ArgumentParser:
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

    parser.add_argument(
        "-env",
        "--conda_environment_name",
        required=False,
        default="executorch",
        help="Provide conda environment name.",
    )

    return parser


def get_compile_specs_from_args(args):
    model_type = CoreMLBackend.MODEL_TYPE.MODEL
    if args.compile:
        model_type = CoreMLBackend.MODEL_TYPE.COMPILED_MODEL

    compute_precision = ct.precision(args.compute_precision)
    compute_unit = ct.ComputeUnit[args.compute_unit.upper()]

    return CoreMLBackend.generate_compile_specs(
        compute_precision=compute_precision,
        compute_unit=compute_unit,
        model_type=model_type,
        minimum_deployment_target=ct.target.iOS17,
    )


def compare_intermediate_tensors(
    edge_program: EdgeProgramManager,
    example_inputs: Tuple[Value, ...],
    coreml_compile_specs: List[CompileSpec],
    model_name: str,
    working_dir_path: Path,
) -> ComparisonResult:
    inspector_coreml = create_inspector_coreml(
        edge_program=edge_program,
        compile_specs=coreml_compile_specs,
        example_inputs=example_inputs,
        model_name=model_name,
        working_dir_path=working_dir_path,
        root_dir_path=get_root_dir_path(),
    )

    inspector_reference = create_inspector_reference(
        edge_program=edge_program,
        example_inputs=example_inputs,
        model_name=model_name,
        working_dir_path=working_dir_path,
        root_dir_path=get_root_dir_path(),
    )

    return get_comparison_result(
        inspector1=inspector_reference,
        tag1="reference",
        inspector2=inspector_coreml,
        tag2="coreml",
    )


def main() -> None:
    parser = args_parser()
    args = parser.parse_args()

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

    build_devtools_runner_including_coreml(
        root_dir_path=get_root_dir_path(), conda_env_name=args.conda_environment_name
    )

    model, example_inputs, _ = EagerModelFactory.create_model(
        *MODEL_NAME_TO_MODEL[args.model_name]
    )

    model.eval()
    edge_program = module_to_edge(
        module=model,
        example_inputs=example_inputs,
    )

    coreml_compile_specs = get_compile_specs_from_args(args)

    with tempfile.TemporaryDirectory() as temp_dir_name:
        working_dir_path = Path(temp_dir_name) / "debugger"
        working_dir_path.mkdir(parents=True, exist_ok=True)
        comparison_result = compare_intermediate_tensors(
            edge_program=edge_program,
            example_inputs=example_inputs,
            coreml_compile_specs=coreml_compile_specs,
            model_name=args.model_name,
            working_dir_path=working_dir_path,
        )

        print(
            tabulate(comparison_result.to_dataframe(), headers="keys", tablefmt="grid")
        )


if __name__ == "__main__":
    main()  # pragma: no cover
