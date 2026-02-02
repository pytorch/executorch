# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch

from executorch.exir._serialize._named_data_store import NamedDataStore
from executorch.exir._serialize.data_serializer import DataEntry
from executorch.exir.scalar_type import ScalarType
from executorch.exir.tensor_layout import TensorLayout


class TestNamedDataStore(unittest.TestCase):
    def test_add(self) -> None:
        store = NamedDataStore()
        store.add_named_data("key1", b"data1", None, None)
        store.add_named_data("key2", b"data2", 16, "file1")
        store.add_named_data("key3", b"data3", 16, "file1")

        output = store.get_named_data_store_output()

        self.assertEqual(len(output.buffers), 3)
        self.assertEqual(output.buffers[0], b"data1")
        self.assertEqual(output.buffers[1], b"data2")
        self.assertEqual(output.buffers[2], b"data3")

        self.assertEqual(len(output.pte_data), 1)
        self.assertEqual(output.pte_data["key1"], DataEntry(0, 1, None))

        self.assertEqual(len(output.external_data), 1)
        self.assertEqual(len(output.external_data["file1"]), 2)
        self.assertEqual(output.external_data["file1"]["key2"], DataEntry(1, 16, None))
        self.assertEqual(output.external_data["file1"]["key3"], DataEntry(2, 16, None))

    def test_add_torch_tensor(self) -> None:
        store = NamedDataStore()
        t0 = torch.tensor([[1, 2], [3, 4]], dtype=torch.int)
        t1 = torch.randn(2, 3, 4, 5).contiguous(memory_format=torch.channels_last)

        store.add_named_data("key0", t0, None, None)
        store.add_named_data("key1", t1, 16, None)

        output = store.get_named_data_store_output()
        self.assertEqual(len(output.buffers), 2)
        self.assertEqual(output.buffers[0], bytes(t0.untyped_storage()))
        self.assertEqual(output.buffers[1], bytes(t1.untyped_storage()))

        self.assertEqual(len(output.pte_data), 2)
        self.assertEqual(
            output.pte_data["key0"],
            DataEntry(0, 1, TensorLayout(ScalarType.INT, [2, 2], [0, 1])),
        )
        self.assertEqual(
            output.pte_data["key1"],
            DataEntry(
                1, 16, TensorLayout(ScalarType.FLOAT, [2, 3, 4, 5], [0, 2, 3, 1])
            ),
        )
        self.assertEqual(len(output.external_data), 0)

    def test_add_invalid_torch_tensor_layout(self) -> None:
        store = NamedDataStore()
        t0 = torch.tensor([[1, 2], [3, 4]], dtype=torch.int)
        # TensorLayout does not match the torch.tensor.
        self.assertRaises(
            ValueError,
            store.add_named_data,
            "key",
            t0,
            1,
            None,
            TensorLayout(ScalarType.FLOAT, [1, 2], [0, 1]),
        )

    def test_add_invalid_torch_tensor_dim_order(self) -> None:
        store = NamedDataStore()
        t0 = torch.randn(2, 3, 4, 5)
        t0 = t0.permute(0, 2, 3, 1)
        # Non-contiguous tensor is not supported.
        self.assertRaises(ValueError, store.add_named_data, "key", t0, 1, None)

    def test_add_duplicate_name_and_data(self) -> None:
        store = NamedDataStore()
        store.add_named_data("key", b"data", None, None)
        store.add_named_data("key", b"data", None, None)

        output = store.get_named_data_store_output()

        self.assertEqual(len(output.buffers), 1)
        self.assertEqual(output.buffers[0], b"data")

        self.assertEqual(len(output.pte_data), 1)
        self.assertEqual(output.pte_data["key"], DataEntry(0, 1, None))

        self.assertEqual(len(output.external_data), 0)

    def test_add_same_data_with_different_alignment(self) -> None:
        store = NamedDataStore()
        store.add_named_data("key", b"data", 3, None)
        store.add_named_data("key1", b"data", 4, None)

        output = store.get_named_data_store_output()

        self.assertEqual(len(output.buffers), 1)
        self.assertEqual(output.buffers[0], b"data")

        self.assertEqual(len(output.pte_data), 2)
        self.assertEqual(output.pte_data["key"], DataEntry(0, 3, None))
        self.assertEqual(output.pte_data["key1"], DataEntry(0, 4, None))

        self.assertEqual(len(output.external_data), 0)

    def test_add_duplicate_key_fail(self) -> None:
        store = NamedDataStore()
        store.add_named_data("key", b"data", None, None)

        # Cannot add item with the same key and different data.
        self.assertRaises(ValueError, store.add_named_data, "key", b"data1", None, None)
        self.assertRaises(
            ValueError, store.add_named_data, "key", b"data1", 16, "file1"
        )

        output = store.get_named_data_store_output()

        self.assertEqual(len(output.buffers), 1)
        self.assertEqual(output.buffers[0], b"data")

        self.assertEqual(len(output.pte_data), 1)
        self.assertEqual(output.pte_data["key"], DataEntry(0, 1, None))
        self.assertEqual(len(output.external_data), 0)

    def test_add_same_data_with_different_tensor_layout(self) -> None:
        store = NamedDataStore()
        tensor_layout1 = TensorLayout(ScalarType.FLOAT, [1, 2], [0, 1])
        tensor_layout2 = TensorLayout(ScalarType.FLOAT, [2, 1], [0, 1])
        store.add_named_data("key", b"data", None, None, tensor_layout1)
        store.add_named_data("key1", b"data", None, None, tensor_layout2)

        output = store.get_named_data_store_output()
        self.assertEqual(len(output.buffers), 1)
        self.assertEqual(output.buffers[0], b"data")

        self.assertEqual(output.pte_data["key"], DataEntry(0, 1, tensor_layout1))
        self.assertEqual(output.pte_data["key1"], DataEntry(0, 1, tensor_layout2))

    def test_merge(self) -> None:
        store1 = NamedDataStore()
        tensor_layout1 = TensorLayout(ScalarType.FLOAT, [1, 2], [0, 1])
        store1.add_named_data("key1", b"data1", None, None, tensor_layout1)
        store1.add_named_data("key2", b"data2", 16, "file1")

        # Check items in the store1.
        output = store1.get_named_data_store_output()
        self.assertEqual(len(output.buffers), 2)
        self.assertEqual(len(output.pte_data), 1)
        self.assertEqual(len(output.external_data), 1)
        self.assertEqual(len(output.external_data["file1"]), 1)

        store2 = NamedDataStore()
        store2.add_named_data("key1", b"data1", None, None, tensor_layout1)
        store2.add_named_data("key3", b"data3", None, None)
        store2.add_named_data("key4", b"data4", 16, "file1")
        store2.add_named_data("key5", b"data5", 16, "file2")

        # Check items in store2.
        output2 = store2.get_named_data_store_output()
        self.assertEqual(len(output2.buffers), 4)
        self.assertEqual(len(output2.pte_data), 2)
        self.assertEqual(len(output2.external_data), 2)
        self.assertEqual(len(output2.external_data["file1"]), 1)
        self.assertEqual(len(output2.external_data["file2"]), 1)

        # Merge store2 into store1.
        store1.merge_named_data_store(output2)

        # Check items in store2 are merged into store1.
        output = store1.get_named_data_store_output()
        # key1, data1 exist in both store1 and store2, so we only have one copy of it.
        self.assertEqual(len(output.buffers), 5)
        self.assertEqual(len(output.pte_data), 2)
        # Confirm DataEntry is correct.
        self.assertEqual(output.pte_data["key1"], DataEntry(0, 1, tensor_layout1))
        self.assertEqual(len(output.external_data), 2)
        self.assertEqual(len(output.external_data["file1"]), 2)
        self.assertEqual(len(output.external_data["file2"]), 1)

    def test_merge_duplicate_error(self) -> None:
        store1 = NamedDataStore()
        store1.add_named_data("key1", b"data1", None, None)

        # Check items in the store1.
        output = store1.get_named_data_store_output()
        self.assertEqual(len(output.buffers), 1)
        self.assertEqual(len(output.pte_data), 1)

        store2 = NamedDataStore()
        store2.add_named_data("key1", b"data2", None, None)

        # Check items in store2.
        output2 = store2.get_named_data_store_output()
        self.assertEqual(len(output2.buffers), 1)
        self.assertEqual(len(output2.pte_data), 1)

        # Merge store2 into store1 raises error as key1 is already in store1
        # with different data.
        self.assertRaises(ValueError, store1.merge_named_data_store, output2)
