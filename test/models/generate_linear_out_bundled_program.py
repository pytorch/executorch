#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This is a tool to create sample BundledProgram flatbuffer for executor_runner.
This file and generate_linear_out_model.py share the same model.
To update the bundled program flatbuffer file, just simply run:
buck2 run executorch/test/models:generate_linear_out_bundled_program
Then commit the updated file (if there are any changes).
"""

import subprocess

import executorch.exir as exir

import torch
from executorch.bundled_program.config import BundledConfig
from executorch.bundled_program.core import create_bundled_program
from executorch.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)
from executorch.exir import ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass, ToOutVarPass
from executorch.exir.print_program import pretty_print

from executorch.test.models.linear_model import LinearModel


def main() -> None:
    model = LinearModel()

    trace_inputs = (torch.ones(2, 2, dtype=torch.float),)

    # Trace to FX Graph.
    exec_prog = (
        exir.capture(model, trace_inputs)
        .to_edge()
        .to_executorch(
            config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(),
                to_out_var_pass=ToOutVarPass(),
            )
        )
    )
    # Emit in-memory representation.
    program = exec_prog.program

    # Emit in-memory representation.
    pretty_print(program)

    # Serialize to flatbuffer.
    program.version = 0

    bundled_inputs = [
        [
            [
                torch.rand(2, 2, dtype=torch.float),
            ]
            for _ in range(10)
        ]
        for _ in range(len(program.execution_plan))
    ]

    arbitrary_attachments = {
        "PROGRAM_ATTACHMENT_A_KEY": b"PROGRAM_ATTACHEMENT_A_VALUE",
        "PROGRAM_ATTACHMENT_B_KEY": b"PROGRAM_ATTACHEMENT_B_VALUE",
    }

    metadatas = [
        {
            "metadata_{}_A_KEY_BYTES_VAL".format(i): b"metadata_A_VALUE",
            "metadata_{}_B_KEY_INT_VAL".format(i): 1,
            "metadata_{}_C_KEY_FLOAT_VAL".format(i): 1.0,
            "metadata_{}_D_KEY_BOOL_VAL".format(i): False,
            "metadata_{}_E_KEY_STR_VAL".format(i): "metadata_E_VALUE",
        }
        for i in range(len(program.execution_plan))
    ]

    bundled_expected_outputs = [
        [[model(*x)] for x in bundled_inputs[i]]
        for i in range(len(program.execution_plan))
    ]

    bundled_config = BundledConfig(
        bundled_inputs, bundled_expected_outputs, metadatas, **arbitrary_attachments
    )

    bundled_program = create_bundled_program(program, bundled_config)
    pretty_print(bundled_program)

    bundled_program_flatbuffer = serialize_from_bundled_program_to_flatbuffer(
        bundled_program
    )

    fbsource_base_path = (
        subprocess.run(["hg", "root"], stdout=subprocess.PIPE).stdout.decode().strip()
    )
    with open(
        f"{fbsource_base_path}/fbcode/executorch/test/models/linear_out_bundled_program.ff",
        "wb",
    ) as file:
        file.write(bundled_program_flatbuffer)


if __name__ == "__main__":
    main()
