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
from executorch.examples.models.llama.norm import (
    RMSNormWithInputScale,
    ScalelessRMSNorm,
)
from executorch.examples.models.llama.rope import Rope


class TestTransformerBlock(unittest.TestCase):
    @staticmethod
    def _model_args(**overrides) -> ModelArgs:
        args = {
            "dim": 4,
            "hidden_dim": 8,
            "n_heads": 1,
            "n_kv_heads": 1,
            "head_dim": 4,
            **overrides,
        }
        return ModelArgs(**args)

    def test_dedicated_weightless_post_ffn_norm(self) -> None:
        args = self._model_args()
        args.weightless_post_ffn_norm = True  # pyrefly: ignore[missing-attribute]

        block = TransformerBlock(args, AttentionSkip()).eval()

        self.assertIsInstance(block.post_ffn_norm, ScalelessRMSNorm)

    def test_learnable_post_ffn_norm_takes_precedence(self) -> None:
        args = self._model_args(use_ffn_learnable_scales=True)
        args.weightless_post_ffn_norm = True  # pyrefly: ignore[missing-attribute]

        block = TransformerBlock(args, AttentionSkip()).eval()

        self.assertIsInstance(block.post_ffn_norm, RMSNormWithInputScale)

    def test_shared_post_ffn_norm_does_not_select_weightless_norm(self) -> None:
        args = self._model_args(post_ffn_norm=True)

        block = TransformerBlock(args, AttentionSkip()).eval()

        self.assertFalse(hasattr(block, "post_ffn_norm"))

    def test_default_has_no_post_ffn_norm(self) -> None:
        block = TransformerBlock(self._model_args(), AttentionSkip()).eval()

        self.assertFalse(hasattr(block, "post_ffn_norm"))

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
