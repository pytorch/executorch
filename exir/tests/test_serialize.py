# pyre-strict

import unittest

from executorch.exir.print_program import pretty_print

from executorch.exir.serialize import (
    deserialize_from_flatbuffer,
    serialize_to_flatbuffer,
)
from executorch.exir.tests.common import get_test_program


class TestSerialize(unittest.TestCase):
    def test_serialization(self) -> None:
        program = get_test_program()
        flatbuffer_from_py = serialize_to_flatbuffer(program)
        pretty_print(program)
        self.assertEqual(program, deserialize_from_flatbuffer(flatbuffer_from_py))

    def test_large_buffer_sizes(self) -> None:
        """
        This test make sure when the non_const_buffer_sizes contains integers overflowing
        a signed/unsigned 32 bit interger, we can still serialize the model and
        get the same program by deserialization.
        """
        program = get_test_program()
        program.execution_plan[0].non_const_buffer_sizes = [0, 2**48]
        flatbuffer_from_py = serialize_to_flatbuffer(program)
        pretty_print(program)
        self.assertEqual(program, deserialize_from_flatbuffer(flatbuffer_from_py))
