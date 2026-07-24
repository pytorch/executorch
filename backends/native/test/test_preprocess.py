# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import struct
import unittest

import torch
import torch.nn as nn

from executorch.backends.native.fat_pte import FAT_MAGIC
from executorch.backends.native.partitioner import NativePartitioner
from executorch.backends.native.specializations import (
    _SPECIALIZATION_REGISTRY,
    register_specialization,
)
from executorch.exir import to_edge_transform_and_lower
from executorch.exir._serialize._flatbuffer_program import _flatbuffer_to_program
from executorch.exir.backend.backend_details import PreprocessResult


def _lower(model, example_inputs, specializations=None):
    ep = torch.export.export(model, example_inputs)
    edge = to_edge_transform_and_lower(
        ep,
        partitioner=[NativePartitioner(specializations=specializations)],
    )
    return edge


def _get_delegate_blob(edge):
    """Extract the single delegate's processed bytes from the lowered program."""
    et = edge.to_executorch()
    delegates = et.executorch_program.backend_delegate_data
    assert len(delegates) == 1, f"Expected 1 delegate blob, got {len(delegates)}"
    return bytes(delegates[0].data)


class TestSpecializationRegistry(unittest.TestCase):
    def setUp(self):
        self._saved = dict(_SPECIALIZATION_REGISTRY)

    def tearDown(self):
        _SPECIALIZATION_REGISTRY.clear()
        _SPECIALIZATION_REGISTRY.update(self._saved)

    def test_register_and_lookup(self):
        register_specialization(
            "TestBackend", lambda ep: PreprocessResult(processed_bytes=b"test")
        )
        self.assertIn("TestBackend", _SPECIALIZATION_REGISTRY)

    def test_unregistered_backend_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            _lower(
                nn.Linear(2, 2),
                (torch.randn(1, 2),),
                specializations=["UnregisteredBackend"],
            )
        self.assertIn("UnregisteredBackend", str(ctx.exception))
        self.assertIn("not registered", str(ctx.exception))

    def test_no_specializations_produces_plain_flatbuffer(self):
        edge = _lower(nn.Linear(2, 2), (torch.randn(1, 2),))
        blob = _get_delegate_blob(edge)
        self.assertNotEqual(blob[:4], FAT_MAGIC)
        program = _flatbuffer_to_program(blob)
        self.assertGreater(len(program.execution_plan[0].operators), 0)

    def test_registered_backend_produces_fat_pte(self):
        register_specialization(
            "TestAccel", lambda ep: PreprocessResult(processed_bytes=b"accel_payload")
        )
        edge = _lower(
            nn.Linear(2, 2), (torch.randn(1, 2),), specializations=["TestAccel"]
        )
        blob = _get_delegate_blob(edge)
        self.assertEqual(blob[:4], FAT_MAGIC)


class TestPreprocessSerialization(unittest.TestCase):
    def _lower_and_deserialize(self, model, example_inputs):
        edge = _lower(model, example_inputs)
        blob = _get_delegate_blob(edge)
        return _flatbuffer_to_program(blob)

    def test_linear_op_names(self):
        program = self._lower_and_deserialize(nn.Linear(4, 4), (torch.randn(1, 4),))
        op_names = [op.name for op in program.execution_plan[0].operators]
        self.assertIn("aten::linear", op_names)

    def test_add_op_name(self):
        class AddModel(nn.Module):
            def forward(self, x, y):
                return x + y

        program = self._lower_and_deserialize(
            AddModel(), (torch.randn(2, 3), torch.randn(2, 3))
        )
        op_names = [op.name for op in program.execution_plan[0].operators]
        self.assertIn("aten::add", op_names)

    def test_reinplace_converts_relu(self):
        class ReluModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 8)

            def forward(self, x):
                return torch.relu(self.linear(x))

        program = self._lower_and_deserialize(ReluModel(), (torch.randn(1, 8),))
        op_names = [op.name for op in program.execution_plan[0].operators]
        self.assertIn("aten::linear", op_names)
        self.assertIn("aten::relu_", op_names)

    def test_fat_pte_contains_both_payloads(self):
        saved = dict(_SPECIALIZATION_REGISTRY)
        register_specialization(
            "FakeAccel", lambda ep: PreprocessResult(processed_bytes=b"FAKE_ACCEL_DATA")
        )
        try:
            edge = _lower(
                nn.Linear(2, 2),
                (torch.randn(1, 2),),
                specializations=["FakeAccel"],
            )
            blob = _get_delegate_blob(edge)
            magic, version, count = struct.unpack_from("<4sII", blob, 0)
            self.assertEqual(magic, FAT_MAGIC)
            self.assertEqual(count, 2)
            self.assertIn(b"FAKE_ACCEL_DATA", blob)
        finally:
            _SPECIALIZATION_REGISTRY.clear()
            _SPECIALIZATION_REGISTRY.update(saved)

    def test_fat_pte_native_payload_is_valid(self):
        """The native slice inside a fat PTE is a valid flatbuffer program."""
        saved = dict(_SPECIALIZATION_REGISTRY)
        register_specialization(
            "FakeAccel", lambda ep: PreprocessResult(processed_bytes=b"FAKE")
        )
        try:
            edge = _lower(
                nn.Linear(2, 2),
                (torch.randn(1, 2),),
                specializations=["FakeAccel"],
            )
            blob = _get_delegate_blob(edge)
            _, _, count = struct.unpack_from("<4sII", blob, 0)
            self.assertEqual(count, 2)

            from executorch.backends.native.fat_pte import _ENTRY_FMT, _ENTRY_SIZE

            header_size = 12
            bid, offset, size = struct.unpack_from("<" + _ENTRY_FMT, blob, header_size)
            data_start = header_size + count * _ENTRY_SIZE
            native_blob = blob[data_start + offset : data_start + offset + size]
            program = _flatbuffer_to_program(native_blob)
            op_names = [op.name for op in program.execution_plan[0].operators]
            self.assertIn("aten::linear", op_names)
        finally:
            _SPECIALIZATION_REGISTRY.clear()
            _SPECIALIZATION_REGISTRY.update(saved)


if __name__ == "__main__":
    unittest.main()
