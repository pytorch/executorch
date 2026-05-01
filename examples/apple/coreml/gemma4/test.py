# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end smoke tests for the Gemma 4 → CoreML export pipeline.

These tests use a tiny synthetic Gemma 4 config (random weights, ~10 layers)
so they finish in seconds and do not need a HuggingFace checkpoint.  They
verify the assertion that this export script makes: the existing
``examples/models/gemma4`` text-decoder lowers cleanly through
``CoreMLPartitioner`` with no portable fallbacks.
"""

import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from export_gemma4_text_decoder_coreml import build_model, export  # noqa: E402

import coremltools as ct  # noqa: E402

from executorch.examples.models.gemma4.text_decoder.gemma4_config import (  # noqa: E402
    Gemma4Config,
)


def _tiny_config() -> Gemma4Config:
    """Return a 10-layer synthetic Gemma 4 config.

    Matches the layer pattern Gemma 4 ships with (4 sliding + 1 full,
    repeated twice) and Gemma 4's MQA / partial RoPE / per-layer head_dim
    structure, just at much smaller dimensions.
    """
    return Gemma4Config(
        hidden_size=64,
        num_hidden_layers=10,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        global_head_dim=32,
        vocab_size=128,
        intermediate_size=128,
        sliding_window=64,
        sliding_window_pattern=5,
        layer_types=(
            ["sliding_attention"] * 4
            + ["full_attention"]
            + ["sliding_attention"] * 4
            + ["full_attention"]
        ),
        num_kv_shared_layers=4,
        max_seq_len=128,
        max_context_len=128,
        hidden_size_per_layer_input=8,
        vocab_size_per_layer_input=128,
    )


class TestGemma4CoreMLExport(unittest.TestCase):
    def test_eager_forward_runs(self):
        """The synthetic config produces a runnable Gemma4TextModel."""
        config = _tiny_config()
        model = build_model(config, checkpoint_path=None, dtype=torch.float32)
        with torch.no_grad():
            out = model(torch.zeros(1, 8, dtype=torch.long))
        self.assertEqual(out.shape, (1, 1, config.vocab_size))

    def test_full_export_pipeline_lowers_to_coreml(self):
        """Run the real export entry point and assert we got a fully delegated PTE."""
        config = _tiny_config()
        # fp32 here — the on-device fp16 numerics path is exercised when the
        # user passes --dtype fp16 to the CLI; this test is about the export
        # plumbing, not numeric quality.
        model = build_model(config, checkpoint_path=None, dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "tiny_gemma4.pte")
            export(
                model,
                input_len=8,
                minimum_deployment_target=ct.target.iOS18,
                compute_precision=ct.precision.FLOAT32,
                output_path=output_path,
            )
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)


if __name__ == "__main__":
    unittest.main()
