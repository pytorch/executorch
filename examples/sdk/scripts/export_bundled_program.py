# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

import argparse

from typing import List

import torch
import os
from executorch.bundled_program.config import BundledConfig, MethodInputType, MethodOutputType
from executorch.bundled_program.core import create_bundled_program
from executorch.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)
from executorch.exir.schema import Program

from ...models import MODEL_NAME_TO_MODEL
from ...models.model_factory import EagerModelFactory
from ...portable.utils import export_to_exec_prog


def save_bundled_program(
    program: Program,
    method_names: List[str],
    bundled_inputs: List[List[MethodInputType]],
    bundled_expected_outputs: List[List[MethodOutputType]],
    output_path: str,
) -> None:
    """
    Generates a bundled program from the given ET program and saves it to the specified path.

    Args:
        program: The Executorch program to bundle.
        method_names: A list of method names in the program to bundle test cases.
        bundled_inputs: Representative inputs for each method.
        bundled_expected_outputs: Expected outputs of representative inputs for each method.
        output_path: Path to save the bundled program.
    """

    bundled_config = BundledConfig(
        method_names, bundled_inputs, bundled_expected_outputs
    )

    bundled_program = create_bundled_program(program, bundled_config)
    bundled_program_buffer = serialize_from_bundled_program_to_flatbuffer(
        bundled_program
    )

    with open(output_path, "wb") as file:
        file.write(bundled_program_buffer)


def export_to_bundled_program(
    model_name: str,
    output_directory: str,
    model: torch.nn.Module,
    example_inputs: MethodInputType,
) -> None:
    """
    Exports the given eager model to bundled program.

    Args:
        model_name: Name of the bundled program to export.
        output_directory: Directory where the bundled program should be saved.
        model: The eager model to export.
        example_inputs: An example input for model's all method for single execution.
                        To simplify, here we assume that all inference methods have the same inputs.
    """


    print("Exporting ET program...")

    # pyre-ignore[6]
    program = export_to_exec_prog(model, example_inputs).executorch_program

    print("Creating bundled test cases...")
    method_names = [method.name for method in program.execution_plan]

    # Just as an example to show how multiple input sets can be bundled along to all methods. Here we
    # create a list called bundled_inputs, every element of which contains all test infos for the method
    # sharing same index in the method_names forwarded to BundledConfig against which it will be tested.
    # Each element is a list with the example_inputs tuple used twice. Each instance of example_inputs
    # is a MethodInputType (Tuple[Union[torch.tenor, int, bool, float]]), which represents one test
    # set for the method.
    bundled_inputs = [[example_inputs, example_inputs] for _ in program.execution_plan]

    bundled_expected_outputs = [
        [[getattr(model, method_names[i])(*x)] for x in bundled_inputs[i]]
        for i in range(len(program.execution_plan))
    ]

    bundled_program_name = f"{model_name}_bundled.bpte"
    output_path = os.path.join(output_directory, bundled_program_name)

    print(f"Saving exported program to {output_path}")
    save_bundled_program(
        program=program,
        method_names=method_names,
        bundled_inputs=bundled_inputs,
        bundled_expected_outputs=bundled_expected_outputs,
        output_path=output_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        help=f"provide a model name. Valid ones: {list(MODEL_NAME_TO_MODEL.keys())}",
    )
    parser.add_argument(
        "-d",
        "--dir",
        default=".",
        help=f"the directory to store the exported bundled program. Default is current directory.",
    )

    args = parser.parse_args()

    if args.model_name not in MODEL_NAME_TO_MODEL:
        raise RuntimeError(
            f"Model {args.model_name} is not a valid name. "
            f"Available models are {list(MODEL_NAME_TO_MODEL.keys())}."
        )

    model, example_inputs = EagerModelFactory.create_model(
        *MODEL_NAME_TO_MODEL[args.model_name]
    )

    export_to_bundled_program(args.model_name, args.dir, model, example_inputs)
