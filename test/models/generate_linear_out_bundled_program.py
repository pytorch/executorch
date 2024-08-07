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
from typing import List

import torch
from executorch.devtools import BundledProgram
from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.devtools.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)
from executorch.exir import ExecutorchBackendConfig, to_edge

from executorch.exir.passes import MemoryPlanningPass, ToOutVarPass
from executorch.exir.print_program import pretty_print

from executorch.test.models.linear_model import LinearModel
from torch.export import export


def main() -> None:
    model = LinearModel()

    trace_inputs = (torch.ones(2, 2, dtype=torch.float),)

    # Trace to FX Graph.
    exec_prog = to_edge(export(model, trace_inputs)).to_executorch(
        config=ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(),
            to_out_var_pass=ToOutVarPass(),
        )
    )
    # Emit in-memory representation.
    pretty_print(exec_prog.executorch_program)

    # Serialize to flatbuffer.
    exec_prog.executorch_program.version = 0

    # Create test sets
    method_test_cases: List[MethodTestCase] = []
    for _ in range(10):
        x = [
            torch.rand(2, 2, dtype=torch.float),
        ]
        method_test_cases.append(MethodTestCase(inputs=x, expected_outputs=model(*x)))
    method_test_suites = [
        MethodTestSuite(method_name="forward", test_cases=method_test_cases)
    ]

    bundled_program = BundledProgram(exec_prog, method_test_suites)

    bundled_program_flatbuffer = serialize_from_bundled_program_to_flatbuffer(
        bundled_program
    )

    fbsource_base_path = (
        subprocess.run(["hg", "root"], stdout=subprocess.PIPE).stdout.decode().strip()
    )
    with open(
        f"{fbsource_base_path}/fbcode/executorch/test/models/linear_out_bundled_program.bpte",
        "wb",
    ) as file:
        file.write(bundled_program_flatbuffer)


if __name__ == "__main__":
    main()
