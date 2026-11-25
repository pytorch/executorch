# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The export's KV capacity summary, checked against the runtime's admission
arithmetic on Muse Glimmer's own layer geometry and synthetic DFlash pools."""

import unittest

from executorch.examples.models.muse_glimmer.export.export_solo import (
    DraftKVCapacity,
    format_memory,
    kv_capacity_summary,
)
from executorch.examples.models.muse_glimmer.model.model import MuseGlimmerConfig

# The default (30B) configuration's per-layer geometry, as the model builds its
# attention layers and offgraph_kv_cache_geometry() publishes it.
CONFIG = MuseGlimmerConfig()
KV_HEADS = [CONFIG.n_kv_heads] * CONFIG.n_layers
HEAD_DIMS = [CONFIG.head_dim] * CONFIG.n_layers
WINDOWS = [CONFIG.layer_window_size(i) for i in range(CONFIG.n_layers)]
# 2 (K and V) x KV heads x head dim x BF16 per layer.
BYTES_PER_CELL = sum(2 * h * d * 2 for h, d in zip(KV_HEADS, HEAD_DIMS))


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


def draft(cells, *, owners=4, block=16, layers=2):
    """A small synthetic draft: `layers` flat layers of 2 heads x 128 dims."""
    return DraftKVCapacity(
        kv_heads=[2] * layers,
        head_dims=[128] * layers,
        windows=[0] * layers,
        cells=cells,
        max_owners=owners,
        block=block,
        max_write=512 + owners * block,
    )


class KVCapacitySummaryTest(unittest.TestCase):
    def test_format_memory_follows_the_rounded_value(self):
        self.assertEqual(format_memory(0), "0.0 MiB")
        self.assertEqual(format_memory(512 * 2**20), "512.0 MiB")
        self.assertEqual(format_memory(1073636966), "1023.9 MiB")
        # Just under 1 GiB rounds to 1024.0 MiB, so it reads in GiB.
        self.assertEqual(format_memory(2**30 - 1), "1.0 GiB")
        self.assertEqual(format_memory(2**30), "1.0 GiB")

    def test_cell_layout_divides_the_pool_less_padding(self):
        lines = summary()
        self.assertIn("shared by every resident session", lines[0])
        self.assertIn("1 padding cell", lines[0])
        # floor((131072 - 1) / n) for 1, 4 and 16 resident sessions.
        self.assertIn("1 -> 131071, 4 -> 32767, 16 -> 8191", lines[1])
        self.assertIn(f"{BYTES_PER_CELL} bytes per cell", lines[2])
        self.assertIn(format_memory(131072 * BYTES_PER_CELL), lines[2])

    def test_cell_layout_caps_the_context_at_max_seq_len(self):
        lines = summary(max_cells=1 << 16, max_seq_len=8192)
        self.assertIn("1 -> 8192, 4 -> 8192, 16 -> 4095", lines[1])

    def test_sequence_layout_rings_sliding_layers(self):
        lines = summary(kv_layout="sequence")
        self.assertIn("16 sequence columns of up to 131072 tokens", lines[0])
        rows = sum(
            2 * h * d * (131072 if w == 0 else min(131072, w + 512 - 1))
            for h, d, w in zip(KV_HEADS, HEAD_DIMS, WINDOWS)
        )
        self.assertIn(format_memory(rows * 17 * 2), lines[1])

    def test_sequence_layout_rings_never_exceed_the_context(self):
        lines = summary(kv_layout="sequence", max_kv_seqs=1, max_seq_len=1024)
        # 2 columns x every layer's 1024 rows.
        self.assertIn(format_memory(2 * 1024 * BYTES_PER_CELL), lines[1])

    def test_dflash_draft_pool_limits_the_cell_context(self):
        # (131072 - 1 - 4 x 16) // n: 8187 < the target's 8191 at 16 sessions.
        lines = summary(draft=draft(131072))
        self.assertIn("1 -> 131007, 4 -> 32751, 16 -> 8187", lines[1])
        # A smaller --draft-max-cells: (32768 - 65) // 16 = 2043.
        lines = summary(draft=draft(32768))
        self.assertIn("4 -> 8175, 16 -> 2043", lines[1])

    def test_dflash_context_must_exceed_the_block(self):
        # (336 - 65) // 16 = 16, the block: unusable; 337 leaves 17.
        self.assertIn("16 -> none", summary(draft=draft(336))[1])
        self.assertIn("16 -> 17", summary(draft=draft(337))[1])

    def test_dflash_draft_storage_is_counted(self):
        lines = summary(draft=draft(32768))
        draft_bytes = 32768 * 2 * 2 * 2 * 128 * 2
        self.assertIn("4 x 16 block cells", lines[3])
        self.assertIn(f"draft {2 * 2 * 2 * 128 * 2} bytes per cell", lines[4])
        self.assertIn(format_memory(draft_bytes), lines[4])
        self.assertIn(
            format_memory(131072 * BYTES_PER_CELL + draft_bytes) + " with the target",
            lines[4],
        )

    def test_dflash_on_the_sequence_layout(self):
        lines = summary(kv_layout="sequence", draft=draft(2048))
        self.assertIn("each session may hold 2032 tokens", lines[2])
        self.assertIn("with the target", lines[3])


if __name__ == "__main__":
    unittest.main()
