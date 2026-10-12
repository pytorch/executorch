# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.examples.models.llama.attention import AttentionSkip
from executorch.examples.models.llama.llama_transformer import (
    Transformer,
    TransformerBlock,
)
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.rope import Rope


class TestTransformerBlock(unittest.TestCase):
    def test_zero_centered_gamma_applies_to_main_norms(self) -> None:
        args = ModelArgs(
            dim=4,
            hidden_dim=8,
            n_heads=1,
            n_kv_heads=1,
            head_dim=4,
            norm_zero_centered_gamma=True,
        )
        block = TransformerBlock(args, AttentionSkip()).eval()
        model = Transformer(args, torch.nn.ModuleList([block]), Rope(args)).eval()

        self.assertTrue(block.ffn_norm.add_unit_offset)
        self.assertTrue(model.norm.add_unit_offset)

        center_zero_args = ModelArgs(
            dim=4,
            hidden_dim=8,
            n_heads=1,
            n_kv_heads=1,
            head_dim=4,
            norm_zero_centered_gamma=True,
            output_norm_gain_center_type="zero",
        )
        center_zero_model = Transformer(
            center_zero_args,
            torch.nn.ModuleList(),
            Rope(center_zero_args),
        ).eval()
        self.assertFalse(center_zero_model.norm.add_unit_offset)

    def test_attention_skip_applies_gated_zero_branch_residual(self) -> None:
        args = ModelArgs(
            dim=4,
            hidden_dim=8,
            n_heads=1,
            n_kv_heads=1,
            head_dim=4,
            use_residual_gate=True,
        )
        block = TransformerBlock(
            args,
            AttentionSkip(),
            mlp_type="skip",
            layer_id=2,
        ).eval()
        x = torch.randn(1, 3, args.dim)
        freqs = torch.empty(0)

        output, update = block(x, freqs, freqs, {})
        assert block.add_attn is not None
        expected = block.add_attn(stream=x, branch=torch.zeros_like(x))

        self.assertIsNone(update)
        torch.testing.assert_close(output, expected)
        self.assertFalse(torch.equal(output, x))
