# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

# pyre-unsafe

import argparse

from typing import List

import torch
from executorch.devtools import BundledProgram
from executorch.devtools.bundled_program.config import (
    MethodInputType,
    MethodTestCase,
    MethodTestSuite,
)
from executorch.devtools.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)

from executorch.exir import ExecutorchProgramManager
from executorch.extension.export_util.utils import export_to_exec_prog

from ...models import MODEL_NAME_TO_MODEL
from ...models.model_factory import EagerModelFactory


def save_bundled_program(
    executorch_program: ExecutorchProgramManager,
    method_test_suites: List[MethodTestSuite],
    output_path: str,
):
    """
    Generates a bundled program from the given ET program and saves it to the specified path.

    Args:
        executorch_program: The ExecuTorch program to bundle.
        method_test_suites: The MethodTestSuites which contains test cases to include in the bundled program.
        output_path: Path to save the bundled program.
    """

    bundled_program = BundledProgram(executorch_program, method_test_suites)
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
    executorch_program = export_to_exec_prog(model, example_inputs)

    print("Creating bundled test cases...")
    method_names = [
        method.name for method in executorch_program.executorch_program.execution_plan
    ]

    # A model could have multiple entry point methods and each of them can have multiple inputs bundled for testing.
    # This example demonstrates a model which has multiple entry point methods, whose name listed in method_names, to which we want
    # to bundle two input test cases (example_inputs is used two times) for each inference method.
    program_inputs = {
        m_name: [example_inputs, example_inputs] for m_name in method_names
    }

    method_test_suites: List[MethodTestSuite] = []
    for m_name in method_names:
        method_inputs = program_inputs[m_name]

        # To create a bundled program, we first create every test cases from input. We leverage eager model
        # to generate expected output for each test input, and use MethodTestCase to hold the information of
        # each test case. We gather all MethodTestCase for same method into one MethodTestSuite, and generate
        # bundled program by all MethodTestSuites.
        method_test_cases: List[MethodTestCase] = []
        for method_input in method_inputs:
            method_test_cases.append(
                MethodTestCase(
                    inputs=method_input,
                    expected_outputs=model(*method_input),
                )
            )

        method_test_suites.append(
            MethodTestSuite(
                method_name=m_name,
                test_cases=method_test_cases,
            )
        )

    save_bundled_program(
        executorch_program, method_test_suites, f"{model_name}_bundled.bpte"
    )


def main() -> None:
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
        help="the directory to store the exported bundled program. Default is current directory.",
    )

    args = parser.parse_args()

    if args.model_name not in MODEL_NAME_TO_MODEL:
        raise RuntimeError(
            f"Model {args.model_name} is not a valid name. "
            f"Available models are {list(MODEL_NAME_TO_MODEL.keys())}."
        )

    model, example_inputs, _ = EagerModelFactory.create_model(
        *MODEL_NAME_TO_MODEL[args.model_name]
    )

    export_to_bundled_program(args.model_name, args.dir, model, example_inputs)


if __name__ == "__main__":
    main()  # pragma: no cover
