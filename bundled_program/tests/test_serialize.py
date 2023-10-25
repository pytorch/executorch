# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

from executorch.bundled_program.core import create_bundled_program

from executorch.bundled_program.serialize import (
    deserialize_from_flatbuffer_to_bundled_program,
    serialize_from_bundled_program_to_flatbuffer,
)
from executorch.bundled_program.tests.common import get_common_program
from executorch.exir.print_program import pretty_print


class TestSerialize(unittest.TestCase):
    def test_bundled_program_serialization(self) -> None:
        program, method_test_suites = get_common_program()

        bundled_program = create_bundled_program(program, method_test_suites)
        pretty_print(bundled_program)
        flat_buffer_bundled_program = serialize_from_bundled_program_to_flatbuffer(
            bundled_program
        )
        regenerate_bundled_program = deserialize_from_flatbuffer_to_bundled_program(
            flat_buffer_bundled_program
        )
        pretty_print(regenerate_bundled_program)
        self.assertEqual(
            bundled_program,
            regenerate_bundled_program,
            "Regenerated bundled program mismatches original one",
        )
