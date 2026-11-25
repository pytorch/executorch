# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The export's KV capacity summary matches the runtime's admission rules."""

import unittest

from executorch.examples.models.muse_glimmer.export.export_solo import (
    format_memory,
    kv_capacity_summary,
)

# Muse Glimmer 30B: 52 layers of 2 KV heads x 128 dims; every fourth layer is
# global, the rest slide over 2048 positions.
LAYERS = 52
KV_HEADS = [2] * LAYERS
HEAD_DIMS = [128] * LAYERS
WINDOWS = [0 if (LAYERS - i - 1) % 4 == 0 else 2048 for i in range(LAYERS)]


def summary(**overrides):
    args = {
        "kv_layout": "cell",
        "max_cells": 131072,
        "max_kv_seqs": 16,
        "max_seq_len": 131072,
        "max_step_tokens": 512,
    }
    args.update(overrides)
    return kv_capacity_summary(KV_HEADS, HEAD_DIMS, WINDOWS, **args)


class KVCapacitySummaryTest(unittest.TestCase):
    def test_format_memory_uses_mib_below_one_gib(self):
        self.assertEqual(format_memory(0), "0.0 MiB")
        self.assertEqual(format_memory(512 * 2**20), "512.0 MiB")
        # Units follow the rounded value: just under 1 GiB is not 1024.0 MiB.
        self.assertEqual(format_memory(2**30 - 1), "1.0 GiB")
        self.assertEqual(format_memory(1073636966), "1023.9 MiB")
        self.assertEqual(format_memory(2**30), "1.0 GiB")

    def test_geometry(self):
        self.assertEqual(WINDOWS.count(0), 13)
        self.assertEqual(WINDOWS.count(2048), 39)

    def test_cell_layout_divides_the_pool_less_padding(self):
        lines = summary()
        self.assertIn("shared by every resident session", lines[0])
        self.assertIn("1 padding cell", lines[0])
        # floor((131072 - 1) / n) for 1, 4 and 16 resident sessions.
        self.assertIn("1 -> 131071, 4 -> 32767, 16 -> 8191", lines[1])
        # 52 x 2 x 2 x 128 x 2 bytes; 131072 cells of it.
        self.assertIn("53248 bytes per cell", lines[2])
        self.assertIn("6.5 GiB", lines[2])

    def test_cell_layout_caps_the_context_at_max_seq_len(self):
        lines = summary(max_cells=1 << 16, max_seq_len=8192)
        self.assertIn("1 -> 8192, 4 -> 8192, 16 -> 4095", lines[1])

    def test_sequence_layout_rings_sliding_layers(self):
        lines = summary(kv_layout="sequence")
        self.assertIn("16 sequence columns of up to 131072 tokens", lines[0])
        # 17 columns x 1024 bytes x (13 x 131072 + 39 x (2048 + 512 - 1)) rows.
        self.assertIn("29.2 GiB", lines[1])

    def test_sequence_layout_short_context_never_exceeds_it(self):
        # Rings never hold more rows than the context.
        lines = summary(kv_layout="sequence", max_kv_seqs=63, max_seq_len=1024)
        # 64 columns x 52 layers x 1024 rows x 1024 bytes per row.
        self.assertIn("3.2 GiB", lines[1])
        # 2 columns x 52 layers x 1024 rows x 1024 bytes per row.
        lines = summary(kv_layout="sequence", max_kv_seqs=1, max_seq_len=1024)
        self.assertIn("104.0 MiB", lines[1])


if __name__ == "__main__":
    unittest.main()
