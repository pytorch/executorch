# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import struct
import unittest

from executorch.backends.native.fat_pte import (
    _ENTRY_FMT,
    _ENTRY_SIZE,
    build_fat_result,
    FAT_MAGIC,
    FAT_VERSION,
    pack_fat_blob,
)
from executorch.exir.backend.backend_details import PreprocessResult


class TestPackFatBlob(unittest.TestCase):
    def test_single_specialization(self):
        payload = b"hello"
        blob = pack_fat_blob([("TestBackend", payload)])

        magic, version, count = struct.unpack_from("<4sII", blob, 0)
        self.assertEqual(magic, FAT_MAGIC)
        self.assertEqual(version, FAT_VERSION)
        self.assertEqual(count, 1)

        header_size = 12
        bid, offset, size = struct.unpack_from("<" + _ENTRY_FMT, blob, header_size)
        self.assertEqual(bid.rstrip(b"\x00"), b"TestBackend")
        self.assertEqual(offset, 0)
        self.assertEqual(size, len(payload))

        data_start = header_size + _ENTRY_SIZE
        self.assertEqual(blob[data_start:], payload)

    def test_multiple_specializations(self):
        entries = [("A", b"aaa"), ("B", b"bbbb"), ("C", b"cc")]
        blob = pack_fat_blob(entries)

        _, _, count = struct.unpack_from("<4sII", blob, 0)
        self.assertEqual(count, 3)

        header_size = 12
        offsets_sizes = []
        for i in range(3):
            bid, off, sz = struct.unpack_from(
                "<" + _ENTRY_FMT, blob, header_size + i * _ENTRY_SIZE
            )
            offsets_sizes.append((bid.rstrip(b"\x00"), off, sz))

        self.assertEqual(offsets_sizes[0], (b"A", 0, 3))
        self.assertEqual(offsets_sizes[1], (b"B", 3, 4))
        self.assertEqual(offsets_sizes[2], (b"C", 7, 2))

        data_start = header_size + 3 * _ENTRY_SIZE
        self.assertEqual(blob[data_start:], b"aaabbbbcc")

    def test_empty_specializations(self):
        blob = pack_fat_blob([])
        magic, version, count = struct.unpack_from("<4sII", blob, 0)
        self.assertEqual(count, 0)
        self.assertEqual(len(blob), 12)

    def test_backend_id_over_32_bytes_raises(self):
        long_name = "A" * 33
        with self.assertRaises(ValueError):
            pack_fat_blob([(long_name, b"x")])


class TestBuildFatResult(unittest.TestCase):
    def test_merges_results(self):
        r1 = PreprocessResult(processed_bytes=b"native", debug_handle_map={1: [1]})
        r2 = PreprocessResult(processed_bytes=b"accel", debug_handle_map={2: [2]})

        result = build_fat_result([("NativeBackend", r1), ("AccelBackend", r2)])

        magic, version, count = struct.unpack_from("<4sII", result.processed_bytes, 0)
        self.assertEqual(magic, FAT_MAGIC)
        self.assertEqual(count, 2)

        # debug_handle_map defaults to first result's
        self.assertEqual(result.debug_handle_map, {1: [1]})

    def test_explicit_debug_handle_map(self):
        r1 = PreprocessResult(processed_bytes=b"a", debug_handle_map={1: [1]})
        custom_map = {99: [99]}
        result = build_fat_result([("X", r1)], debug_handle_map=custom_map)
        self.assertEqual(result.debug_handle_map, custom_map)

    def test_single_result(self):
        r = PreprocessResult(processed_bytes=b"only")
        result = build_fat_result([("Solo", r)])
        _, _, count = struct.unpack_from("<4sII", result.processed_bytes, 0)
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
