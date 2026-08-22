# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

from executorch.devtools.bundled_program.core import BundledProgram

from executorch.devtools.bundled_program.serialize import (
    deserialize_from_flatbuffer_to_bundled_program,
    serialize_from_bundled_program_to_flatbuffer,
)
from executorch.devtools.bundled_program.util.test_util import (
    get_common_executorch_program,
)


class TestSerialize(unittest.TestCase):
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
