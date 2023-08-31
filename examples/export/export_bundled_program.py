# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

import argparse

from executorch.bundled_program.config import BundledConfig
from executorch.bundled_program.core import create_bundled_program
from executorch.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)

from ..models import MODEL_NAME_TO_MODEL
from ..models.model_factory import EagerModelFactory

from .utils import export_to_exec_prog, save_pte_program


def save_bundled_program(
    inputs,
    exec_prog,
    graph_module,
    output_path,
):
    # Here inputs is List[Tuple[Union[torch.tenor, int, bool]]]. Each tuple is one input test
    # set for the model. If we wish to test the model with multiple inputs then they can be
    # appended to this list. len(inputs) == number of test sets we want to run.
    #
    # If we have multiple execution plans in this program then we add another list of tuples
    # to test that corresponding execution plan. Index of list of tuples will match the index
    # of the execution plan against which it will be tested.
    bundled_inputs = [inputs for _ in range(len(exec_prog.program.execution_plan))]

    # For each input tuple we run the graph module and put the resulting output in a list. This
    # is repeated over all the tuples present in the input list and then repeated for each execution
    # plan we want to test against.
    expected_outputs = [
        [[graph_module(*x)] for x in inputs]
        for i in range(len(exec_prog.program.execution_plan))
    ]

    bundled_config = BundledConfig(bundled_inputs, expected_outputs)

    bundled_program = create_bundled_program(exec_prog.program, bundled_config)
    bundled_program_buffer = serialize_from_bundled_program_to_flatbuffer(
        bundled_program
    )

    with open(output_path, "wb") as file:
        file.write(bundled_program_buffer)


def export_to_pte(model_name, model, example_inputs):
    exec_prog = export_to_exec_prog(model, example_inputs)
    save_pte_program(exec_prog.buffer, model_name)

    # Just as an example to show how multiple input sets can be bundled along, here we
    # create a list with the example_inputs tuple used twice. Each instance of example_inputs
    # is a Tuple[Union[torch.tenor, int, bool]] which represents one test set for the model.
    bundled_inputs = [example_inputs, example_inputs]
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
