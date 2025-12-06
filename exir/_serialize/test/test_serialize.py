#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import copy
import unittest

from executorch.exir._serialize._cord import Cord
from executorch.exir._serialize._named_data_store import NamedDataStoreOutput
from executorch.exir._serialize._serialize import serialize_for_executorch
from executorch.exir._serialize.data_serializer import (
    DataEntry,
    DataPayload,
    DataSerializer,
)
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.emit._emit_program import EmitterOutput
from executorch.exir.schema import (
    EValue,
    ExtraTensorInfo,
    ScalarType,
    Tensor,
    TensorDataLocation,
    TensorShapeDynamism,
)
from executorch.exir.tests.common import get_test_program


class MockDataSerializer(DataSerializer):
    """A mock DataSerializer for testing purposes."""

    def __init__(self) -> None:
        self.serialize_call_count = 0
        self.last_payload = None

    def serialize(self, data: DataPayload) -> Cord:
        self.serialize_call_count += 1
        self.last_payload = data
        # Return a simple cord with identifiable content
        return Cord(b"mock_ptd_data")

    def deserialize(self, blob: Cord) -> DataPayload:
        return DataPayload(buffers=[], named_data={})


class TestSerializeForExecutorch(unittest.TestCase):
    """Tests for serialize_for_executorch function."""

    @classmethod
    def setUpClass(cls) -> None:
        # Cache common fixtures - Program creation can be expensive
        cls._program = get_test_program()
        cls._config = ExecutorchBackendConfig()

    def _create_emitter_output(
        self,
        external_constant_buffer: list[bytes] | None = None,
        external_constant_map: dict[str, dict[str, int]] | None = None,
    ) -> EmitterOutput:
        """Creates EmitterOutput using cached program."""
        return EmitterOutput(
            program=copy.deepcopy(self._program),
            debug_handle_map={},
            method_to_delegate_debug_id_map={},
            instruction_id_to_num_outs_map={},
            mutable_data=None,
            external_constant_buffer=external_constant_buffer or [],
            external_constant_map=external_constant_map or {},
        )

    def _create_emitter_output_with_external_tensor(
        self,
        fqn: str,
        external_tag: str,
        tensor_data: bytes,
    ) -> EmitterOutput:
        """Creates an EmitterOutput with an external tensor for testing."""
        program = copy.deepcopy(self._program)

        # Add a tensor with external location to the execution plan
        external_tensor = Tensor(
            scalar_type=ScalarType.FLOAT,
            storage_offset=0,
            sizes=[2, 2],
            dim_order=[0, 1],
            requires_grad=False,
            layout=0,
            data_buffer_idx=0,
            allocation_info=None,
            shape_dynamism=TensorShapeDynamism.STATIC,
            extra_tensor_info=ExtraTensorInfo(
                fully_qualified_name=fqn,
                location=TensorDataLocation.EXTERNAL,
            ),
        )
        program.execution_plan[0].values.append(EValue(val=external_tensor))

        return EmitterOutput(
            program=program,
            debug_handle_map={},
            method_to_delegate_debug_id_map={},
            instruction_id_to_num_outs_map={},
            mutable_data=None,
            external_constant_buffer=[tensor_data],
            external_constant_map={external_tag: {fqn: 0}},
        )

    def test_with_no_external_data(self) -> None:
        """Test basic serialization without named data store."""
        emitter_output = self._create_emitter_output()
        data_serializer = MockDataSerializer()

        pte, ptd_files = serialize_for_executorch(
            emitter_output=emitter_output,
            config=self._config,
            data_serializer=data_serializer,
            named_data_store=None,
        )

        # Should return a non-empty PTE cord
        self.assertGreater(len(bytes(pte)), 0)
        # Should return empty PTD files dict when no external data
        self.assertEqual(ptd_files, {})
        # DataSerializer should not be called when no external data
        self.assertEqual(data_serializer.serialize_call_count, 0)

    def test_with_named_data_store(self) -> None:
        """Test serialization with named data that goes into external PTD files."""
        emitter_output = self._create_emitter_output()
        data_serializer = MockDataSerializer()

        # Create named data store with multiple external tags
        named_data_store = NamedDataStoreOutput(
            buffers=[b"buffer1", b"buffer2"],
            pte_data={},
            external_data={
                "weights.ptd": {
                    "buffer1": DataEntry(
                        buffer_index=0, alignment=16, tensor_layout=None
                    ),
                },
                "biases.ptd": {
                    "buffer2": DataEntry(
                        buffer_index=1, alignment=8, tensor_layout=None
                    ),
                },
            },
        )

        _, ptd_files = serialize_for_executorch(
            emitter_output=emitter_output,
            config=self._config,
            data_serializer=data_serializer,
            named_data_store=named_data_store,
        )
        # Should have two PTD files
        self.assertEqual(len(ptd_files), 2)
        self.assertIn("weights.ptd", ptd_files)
        self.assertIn("biases.ptd", ptd_files)
        # DataSerializer should be called twice
        self.assertEqual(data_serializer.serialize_call_count, 2)

    def test_with_named_data_store_duplicates(self) -> None:
        """Test serialization with multiple keys pointing to the same buffer."""
        emitter_output = self._create_emitter_output()
        data_serializer = MockDataSerializer()

        # Create named data store with multiple keys pointing to same buffers
        named_data_store = NamedDataStoreOutput(
            buffers=[b"buffer1", b"buffer2"],
            pte_data={},
            external_data={
                "weights.ptd": {
                    "weight1": DataEntry(
                        buffer_index=0, alignment=16, tensor_layout=None
                    ),
                    "weight2": DataEntry(
                        buffer_index=0, alignment=16, tensor_layout=None
                    ),  # same buffer
                    "weight3": DataEntry(
                        buffer_index=1, alignment=8, tensor_layout=None
                    ),
                },
            },
        )

        _, ptd_files = serialize_for_executorch(
            emitter_output=emitter_output,
            config=self._config,
            data_serializer=data_serializer,
            named_data_store=named_data_store,
        )
        # Should have one PTD file
        self.assertEqual(len(ptd_files), 1)
        self.assertIn("weights.ptd", ptd_files)
        # DataSerializer should be called once
        self.assertEqual(data_serializer.serialize_call_count, 1)

        # Verify the payload passed to the data serializer.
        payload = data_serializer.last_payload
        # Should have only 2 buffers despite 3 named entries
        self.assertEqual(len(payload.buffers), 2)
        self.assertEqual(payload.buffers[0], b"buffer1")
        self.assertEqual(payload.buffers[1], b"buffer2")
        # Should have 3 named_data entries
        self.assertEqual(len(payload.named_data), 3)
        # weight1 and weight2 should reference the same buffer index
        self.assertEqual(payload.named_data["weight1"].buffer_index, 0)
        self.assertEqual(payload.named_data["weight2"].buffer_index, 0)
        self.assertEqual(payload.named_data["weight3"].buffer_index, 1)

    def test_with_external_tensor(self) -> None:
        """Test serialization with an external tensor from emitter output."""
        fqn = "tensor1"
        external_tag = "weights.ptd"
        tensor_data = b"buffer"

        emitter_output = self._create_emitter_output_with_external_tensor(
            fqn=fqn,
            external_tag=external_tag,
            tensor_data=tensor_data,
        )
        data_serializer = MockDataSerializer()

        # Named data store is required when there are external tensors
        named_data_store = NamedDataStoreOutput(
            buffers=[],
            pte_data={},
            external_data={external_tag: {}},
        )

        _, ptd_files = serialize_for_executorch(
            emitter_output=emitter_output,
            config=self._config,
            data_serializer=data_serializer,
            named_data_store=named_data_store,
        )

        # Should have the external PTD file
        self.assertIn(external_tag, ptd_files)

    def test_with_named_data_store_and_external_tensor(self) -> None:
        """Test serialization with external data from named data store and emitter."""
        emitter_output = self._create_emitter_output_with_external_tensor(
            fqn="buffer1",
            external_tag="weights.ptd",
            tensor_data=b"buffer1",
        )
        data_serializer = MockDataSerializer()

        # Create named data store with external data
        named_data_store = NamedDataStoreOutput(
            buffers=[b"buffer2"],
            pte_data={},
            external_data={
                "weights.ptd": {
                    "buffer2": DataEntry(
                        buffer_index=0, alignment=16, tensor_layout=None
                    ),
                },
            },
        )

        _, ptd_files = serialize_for_executorch(
            emitter_output=emitter_output,
            config=self._config,
            data_serializer=data_serializer,
            named_data_store=named_data_store,
        )

        # External data should produce PTD file
        self.assertIn("weights.ptd", ptd_files)
        self.assertEqual(data_serializer.serialize_call_count, 1)

        payload = data_serializer.last_payload
        self.assertEqual(len(payload.buffers), 2)
        self.assertEqual(payload.buffers[0], b"buffer1")
        self.assertEqual(payload.buffers[1], b"buffer2")
        # Should have 2 named_data entries
        self.assertEqual(len(payload.named_data), 2)


if __name__ == "__main__":
    unittest.main()
