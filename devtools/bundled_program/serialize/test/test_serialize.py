# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.devtools.bundled_program.core import BundledProgram

from executorch.devtools.bundled_program.serialize import (
    convert_to_flatbuffer,
    deserialize_from_flatbuffer_to_bundled_program,
    serialize_from_bundled_program_to_flatbuffer,
    serialize_from_bundled_program_to_json,
)
from executorch.devtools.bundled_program.util.test_util import (
    get_common_executorch_program,
)
from executorch.exir import to_edge


class TestSerialize(unittest.TestCase):
    def test_none_input_round_trip(self) -> None:
        class OptionalInput(torch.nn.Module):
            def forward(self, x, optional, y):
                return x - y if optional is None else x + optional + y

        inputs = (torch.ones(2), None, torch.full((2,), 3.0))
        program = to_edge(torch.export.export(OptionalInput(), inputs)).to_executorch()
        bundle = BundledProgram(
            program,
            [MethodTestSuite("forward", [MethodTestCase(inputs)])],
        )
        with self.assertRaisesRegex(AssertionError, "does not meet Program"):
            BundledProgram(
                program,
                [
                    MethodTestSuite(
                        "forward", [MethodTestCase((inputs[0], False, inputs[2]))]
                    )
                ],
            )
        schema = bundle.serialize_to_schema()
        for data in (
            serialize_from_bundled_program_to_flatbuffer(bundle),
            convert_to_flatbuffer(serialize_from_bundled_program_to_json(schema)),
        ):
            restored = deserialize_from_flatbuffer_to_bundled_program(data)
            self.assertIsNone(
                restored.method_test_suites[0].test_cases[0].inputs[1].val
            )
            self.assertEqual(schema, restored)

    def test_bundled_program_serialization(self) -> None:
        executorch_program, method_test_suites = get_common_executorch_program()

        bundled_program = BundledProgram(executorch_program, method_test_suites)
        flat_buffer_bundled_program = serialize_from_bundled_program_to_flatbuffer(
            bundled_program
        )
        regenerate_bundled_program_in_schema = (
            deserialize_from_flatbuffer_to_bundled_program(flat_buffer_bundled_program)
        )
        self.assertEqual(
            bundled_program.serialize_to_schema(),
            regenerate_bundled_program_in_schema,
            "Regenerated bundled program mismatches original one",
        )
