# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

from executorch.exir._serialize._named_data_store import BufferEntry, NamedDataStore


class TestNamedDataStore(unittest.TestCase):
    def test_add(self) -> None:
        store = NamedDataStore()
        store.add_named_data("key1", b"data1", None, None)
        store.add_named_data("key2", b"data2", 16, "file1")
        store.add_named_data("key3", b"data3", 16, "file1")

        output = store.get_named_data_store_output()

        self.assertEqual(len(output.buffers), 3)
        self.assertEqual(output.buffers[0], BufferEntry(b"data1", 1))
        self.assertEqual(output.buffers[1], BufferEntry(b"data2", 16))
        self.assertEqual(output.buffers[2], BufferEntry(b"data3", 16))

        self.assertEqual(len(output.pte_data), 1)
        self.assertEqual(output.pte_data["key1"], 0)

        self.assertEqual(len(output.external_data), 1)
        self.assertEqual(len(output.external_data["file1"]), 2)
        self.assertEqual(output.external_data["file1"]["key2"], 1)
        self.assertEqual(output.external_data["file1"]["key3"], 2)

    def test_add_duplicate_name_and_data(self) -> None:
        store = NamedDataStore()
        store.add_named_data("key", b"data", None, None)
        store.add_named_data("key", b"data", None, None)

        output = store.get_named_data_store_output()

        self.assertEqual(len(output.buffers), 1)
        self.assertEqual(output.buffers[0], BufferEntry(b"data", 1))

        self.assertEqual(len(output.pte_data), 1)
        self.assertEqual(output.pte_data["key"], 0)

        self.assertEqual(len(output.external_data), 0)

    def test_add_same_data_with_different_alignment(self) -> None:
        store = NamedDataStore()
        store.add_named_data("key", b"data", 3, None)
        store.add_named_data("key1", b"data", 4, None)

        output = store.get_named_data_store_output()

        self.assertEqual(len(output.buffers), 1)
        # Check that we take the LCM of the two alignments (3, 4) = 12
        self.assertEqual(output.buffers[0], BufferEntry(b"data", 12))

        self.assertEqual(len(output.pte_data), 2)
        self.assertEqual(output.pte_data["key"], 0)
        self.assertEqual(output.pte_data["key1"], 0)

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
        self.assertEqual(output.buffers[0], BufferEntry(b"data", 1))

        self.assertEqual(len(output.pte_data), 1)
        self.assertEqual(output.pte_data["key"], 0)
        self.assertEqual(len(output.external_data), 0)
