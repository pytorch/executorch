# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import hashlib
import unittest
from typing import Any, cast

import torch

from executorch.exir._serialize._named_data_store import NamedDataStore
from executorch.exir._serialize.data_serializer import DataEntry
from executorch.exir.scalar_type import ScalarType
from executorch.exir.tensor_layout import TensorLayout


class _SizedBuffer:
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size


class TestNamedDataStore(unittest.TestCase):
    def test_externalize_pte_data_counts_aliased_buffer_once(self) -> None:
        store = NamedDataStore()
        layout = TensorLayout(ScalarType.FLOAT, [1], [0])
        store.add_named_data("key_a", b"aaaaaa", 16, None, layout)
        store.add_named_data("key_a_alias", b"aaaaaa", 32, None, layout)
        store.add_named_data("key_b", b"bbbb", 16, None, layout)
        expected_buffers = list(store.buffers)
        expected_entries = copy.deepcopy(store.pte_data)

        store.externalize_pte_data(10, "test_constants")
        output = store.get_named_data_store_output()

        self.assertEqual(output.buffers, expected_buffers)
        self.assertEqual(output.pte_data, {})
        self.assertEqual(len(output.external_data), 1)
        self.assertEqual(next(iter(output.external_data.values())), expected_entries)

    def test_externalize_pte_data_rollover_is_insertion_order_independent(
        self,
    ) -> None:
        def externalize(
            order: list[str],
        ) -> dict[str, list[tuple[str, bytes, int, TensorLayout | None]]]:
            data = {
                "key_a": (b"a" * 6, 16),
                "key_a_alias": (b"a" * 6, 32),
                "key_b": (b"b" * 4, 32),
                "key_c": (b"c" * 5, 64),
            }
            store = NamedDataStore()
            for key in order:
                payload, alignment = data[key]
                store.add_named_data(key, payload, alignment)
            store.externalize_pte_data(10, "test_constants")
            output = store.get_named_data_store_output()
            return {
                tag: [
                    (
                        key,
                        output.buffers[entry.buffer_index],
                        entry.alignment,
                        entry.tensor_layout,
                    )
                    for key, entry in entries.items()
                ]
                for tag, entries in output.external_data.items()
            }

        forward = externalize(["key_a", "key_a_alias", "key_b", "key_c"])
        reverse = externalize(["key_c", "key_b", "key_a_alias", "key_a"])

        self.assertEqual(len(forward), 2)
        self.assertEqual(forward, reverse)

    def test_externalize_pte_data_rejects_oversized_buffer_atomically(self) -> None:
        store = NamedDataStore()
        store.add_named_data("external", b"ext", 8, "existing")
        store.add_named_data("key_a", b"aaaa", 16)
        store.add_named_data("key_z", b"z" * 11, 32)
        before = (
            list(store.buffers),
            copy.deepcopy(store.pte_data),
            copy.deepcopy(store.external_data),
            dict(store.key_to_buffer_idx),
        )

        with self.assertRaisesRegex(ValueError, "exceeds external data shard cap"):
            store.externalize_pte_data(10, "test_constants")

        after = (
            list(store.buffers),
            store.pte_data,
            store.external_data,
            store.key_to_buffer_idx,
        )
        self.assertEqual(after, before)

    def test_externalize_pte_data_accepts_production_max_buffer_metadata(
        self,
    ) -> None:
        store = NamedDataStore()
        store.buffers = [cast(bytes, _SizedBuffer(1_174_405_120))]
        store.pte_data = {"largest": DataEntry(0, 16, None)}
        store.key_to_buffer_idx = {"largest": 0}

        store.externalize_pte_data(1_500_000_000, "vulkan_constants")

        output = store.get_named_data_store_output()
        self.assertEqual(output.pte_data, {})
        self.assertEqual(
            [list(entries) for entries in output.external_data.values()],
            [["largest"]],
        )

    def test_externalize_pte_data_rejects_invalid_arguments(self) -> None:
        store = NamedDataStore()
        invalid_caps: list[Any] = [True, 0, -1, 1.5, "10"]
        for cap in invalid_caps:
            with self.subTest(cap=cap), self.assertRaisesRegex(
                ValueError, "positive integer"
            ):
                store.externalize_pte_data(cap, "test_constants")

        with self.assertRaisesRegex(ValueError, "prefix must be nonempty"):
            store.externalize_pte_data(10, "")

    def test_externalize_pte_data_rejects_tag_collision_atomically(self) -> None:
        store = NamedDataStore()
        key = "inline"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        tag = f"test_constants_{digest}"
        store.add_named_data("external", b"ext", 8, tag)
        store.add_named_data(key, b"inline", 16)
        before = (
            list(store.buffers),
            copy.deepcopy(store.pte_data),
            copy.deepcopy(store.external_data),
        )

        with self.assertRaisesRegex(ValueError, "external data tag collision"):
            store.externalize_pte_data(10, "test_constants")

        self.assertEqual(
            (list(store.buffers), store.pte_data, store.external_data), before
        )

    def test_externalize_pte_data_preserves_existing_external_data_and_is_idempotent(
        self,
    ) -> None:
        store = NamedDataStore()
        store.add_named_data("external", b"ext", 8, "existing")
        store.add_named_data("inline", b"inline", 16)

        store.externalize_pte_data(10, "test_constants")
        first = copy.deepcopy(store.get_named_data_store_output())
        store.externalize_pte_data(10, "test_constants")
        second = store.get_named_data_store_output()

        self.assertEqual(first, second)
        self.assertEqual(list(second.external_data["existing"]), ["external"])
        self.assertEqual(len(second.external_data), 2)

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

    def test_fingerprint_collision(self) -> None:
        """Two blobs with same length and first 32 bytes but different content
        must not be deduped."""
        store = NamedDataStore()
        prefix = b"A" * 32
        data1 = prefix + b"X" * 100
        data2 = prefix + b"Y" * 100
        self.assertEqual(len(data1), len(data2))

        store.add_named_data("key1", data1, None, None)
        store.add_named_data("key2", data2, None, None)

        output = store.get_named_data_store_output()
        self.assertEqual(len(output.buffers), 2)
        self.assertEqual(output.buffers[0], data1)
        self.assertEqual(output.buffers[1], data2)
        self.assertEqual(output.pte_data["key1"].buffer_index, 0)
        self.assertEqual(output.pte_data["key2"].buffer_index, 1)

    def test_fingerprint_collision_with_dedup(self) -> None:
        """After a fingerprint collision, a true duplicate of the first blob
        must still be deduped correctly."""
        store = NamedDataStore()
        prefix = b"A" * 32
        data1 = prefix + b"X" * 100
        data2 = prefix + b"Y" * 100

        store.add_named_data("key1", data1, None, None)
        store.add_named_data("key2", data2, None, None)
        store.add_named_data("key3", data1, None, None)  # duplicate of key1

        output = store.get_named_data_store_output()
        self.assertEqual(len(output.buffers), 2)
        self.assertEqual(output.pte_data["key1"].buffer_index, 0)
        self.assertEqual(output.pte_data["key2"].buffer_index, 1)
        self.assertEqual(output.pte_data["key3"].buffer_index, 0)
