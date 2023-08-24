# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Any, Optional, Tuple

import executorch.exir as exir

import torch

from executorch.codegen.tools.pybindings.selective_build import (
    _get_io_metadata_for_program_operators,
    _get_program_from_buffer,
    _get_program_operators,
    _IOMetaData,
)
from executorch.exir import CaptureConfig
from executorch.exir.print_program import pretty_print
from executorch.exir.scalar_type import ScalarType
from executorch.exir.schema import Program


class ModuleAdd(torch.nn.Module):
    """The module to serialize and execute."""

    def __init__(self):
        super(ModuleAdd, self).__init__()

    def forward(self, x, y):
        return x + y

    def get_methods_to_export(self):
        return ("forward",)


class ModuleMulti(torch.nn.Module):
    """The module to serialize and execute."""

    def __init__(self):
        super(ModuleMulti, self).__init__()

    def forward(self, x, y):
        return x + y

    def forward2(self, x, y):
        return x + y + 1

    def get_methods_to_export(self):
        return ("forward", "forward2")


def create_program(
    eager_module: Optional[torch.nn.Module] = None,
) -> Tuple[Program, Tuple[Any, ...]]:
    """Returns an executorch program based on ModuleAdd, along with inputs."""

    if eager_module is None:
        eager_module = ModuleAdd()

    # Trace the test module and create a serialized Executorch program.
    inputs = (torch.ones(2, 2), torch.ones(2, 2))
    input_map = {}
    for method in eager_module.get_methods_to_export():
        input_map[method] = inputs

    # These cleanup passes are required to convert the `add` op to its out
    # variant, along with some other transformations.
    exec_prog = (
        exir.capture_multiple(eager_module, input_map, config=CaptureConfig())
        .to_edge()
        .to_executorch()
    )

    # Create the Executorch program from the graph.
    pretty_print(exec_prog.program)
    return (exec_prog, inputs)


class PybindingsTest(unittest.TestCase):
    def test_dump_operators(self):
        # Create and serialize a program.
        orig_program, _ = create_program()

        # Deserialize the program and demonstrate that we could get its operator
        # list.
        program = _get_program_from_buffer(orig_program.buffer)
        operators = _get_program_operators(program)
        self.assertEqual(operators, ["aten::add.out"])

    def test_get_op_io_meta(self):
        # Checking whether get_op_io_meta returns the correct metadata for all its ios.
        orig_program, inputs = create_program()

        # Deserialize the program and demonstrate that we could get its operator
        # list.
        program = _get_program_from_buffer(orig_program.buffer)
        program_op_io_metadata = _get_io_metadata_for_program_operators(program)

        self.assertTrue(len(program_op_io_metadata) == 1)
        self.assertTrue(isinstance(program_op_io_metadata, dict))

        self.assertTrue("aten::add.out" in program_op_io_metadata)
        self.assertTrue(isinstance(program_op_io_metadata["aten::add.out"], set))
        self.assertTrue(len(program_op_io_metadata["aten::add.out"]) == 1)

        for op_io_metadata in program_op_io_metadata["aten::add.out"]:
            self.assertTrue(len(op_io_metadata) == 5)
            self.assertTrue(isinstance(op_io_metadata, tuple))

            for io_idx, io_metadata in enumerate(op_io_metadata):
                self.assertTrue(isinstance(io_metadata, _IOMetaData))
                if io_idx == 2:
                    # TODO(gasoonjia): Create a enum class to map KernelTypes to int, remove the hardcoded 2 and 5 below.
                    self.assertEqual(io_metadata.kernel_type, 2)
                else:
                    self.assertEqual(io_metadata.kernel_type, 5)
                    self.assertEqual(io_metadata.dtype, ScalarType.FLOAT)
                    self.assertEqual(io_metadata.dim_order, [0, 1])
