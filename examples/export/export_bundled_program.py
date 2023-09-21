# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

import argparse
from typing import Dict, List, Tuple, Union

import torch

from executorch.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.bundled_program.core import create_bundled_program
from executorch.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)
from executorch.exir import ExecutorchProgram

from ..models import MODEL_NAME_TO_MODEL
from ..models.model_factory import EagerModelFactory

from .utils import export_to_exec_prog, save_pte_program


def save_bundled_program(
    inputs: Dict[str, List[Tuple[Union[torch.Tensor, int, bool]]]],
    exec_prog: ExecutorchProgram,
    graph_module: torch.nn.Module,
    output_path: str,
):

    # Here inputs is Dict[str, List[Tuple[Union[torch.Tensor, int, bool]]]]. Each Tuple is one input test
    # case for the model, each List contains all test cases for a method, and each str is a method name.

    method_test_suites: List[MethodTestSuite] = []
    for method_name, method_inputs in inputs.items():
        # To create a bundled program, we first create every test cases from input. We leverage graph_module
        # to generate expected output for each test input, and use MethodTestCase to hold the information of
        # each test case. We gather all MethodTestCase for same method into one MethodTestSuite, and generate
        # bundled program by all MethodTestSuites.
        method_test_cases: List[MethodTestCase] = []
        for method_input in method_inputs:
            method_test_cases.append(
                MethodTestCase(
                    inputs=method_input,
                    expected_outputs=[graph_module(*method_input)],
                )
            )

        method_test_suites.append(
            MethodTestSuite(
                method_name=method_name,
                test_cases=method_test_cases,
            )
        )

    bundled_program = create_bundled_program(exec_prog.program, method_test_suites)
    bundled_program_buffer = serialize_from_bundled_program_to_flatbuffer(
        bundled_program
    )

    with open(output_path, "wb") as file:
        file.write(bundled_program_buffer)


def export_to_pte(model_name, model, example_inputs):
    exec_prog = export_to_exec_prog(model, example_inputs)
    save_pte_program(exec_prog.buffer, model_name)

    # Here is an exmaple of how to bundle multiple inputs sets along multiple methods.
    # Here we set up a dictionary, which maps the method name to the corresponding list
    # of inputs cases. Here we create a list with the example_inputs tuple used twice to
    # mimic multiple set of input for each methods. Each instance of example_inputs
    # is a Tuple[Union[torch.tenor, int, bool]] which represents one test set for the model.

    bundled_inputs = {
        method.name: [example_inputs, example_inputs]
        for method in exec_prog.program.execution_plan
    }
    print(f"Saving exported program to {model_name}_bundled.pte")
    save_bundled_program(bundled_inputs, exec_prog, model, f"{model_name}_bundled.pte")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        help=f"provide a model name. Valid ones: {list(MODEL_NAME_TO_MODEL.keys())}",
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

    export_to_pte(args.model_name, model, example_inputs)
