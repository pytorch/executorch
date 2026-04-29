# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for quant/recipe.py. CPU only — no CUDA, no model, no torchao."""

import unittest

from parameterized import parameterized

from .recipe import QuantConfig, QuantRecipe, QuantRule

_Q4 = QuantConfig(4, 32, True, "min_max")
_Q8 = QuantConfig(8, 32, True, "min_max")


class TestQuantRecipeGetConfig(unittest.TestCase):
    """Tests for ``QuantRecipe.get_config`` — the core matching logic."""

    @parameterized.expand(
        [
            (
                "first_match_wins",
                [QuantRule(r".*v_proj\.weight", _Q8), QuantRule(r".*\.weight", _Q4)],
                "layers.0.self_attn.v_proj.weight",
                8,
            ),
            (
                "fallthrough_to_catchall",
                [QuantRule(r".*v_proj\.weight", _Q8), QuantRule(r".*\.weight", _Q4)],
                "layers.0.self_attn.q_proj.weight",
                4,
            ),
            (
                "none_rule_skips",
                [
                    QuantRule(r"embed_tokens\.weight", None),
                    QuantRule(r".*\.weight", _Q4),
                ],
                "embed_tokens.weight",
                None,
            ),
            (
                "unmatched_returns_none",
                [QuantRule(r"foo", _Q4)],
                "bar.weight",
                None,
            ),
            (
                "empty_recipe",
                [],
                "anything",
                None,
            ),
            (
                "fullmatch_not_partial",
                [QuantRule(r"foo", _Q4)],
                "foo.bar",
                None,
            ),
            (
                "fullmatch_exact",
                [QuantRule(r"foo", _Q4)],
                "foo",
                4,
            ),
        ]
    )
    def test_get_config(self, _name, rules, fqn, expected_bits):
        recipe = QuantRecipe(rules=rules)
        config = recipe.get_config(fqn)
        if expected_bits is None:
            self.assertIsNone(config)
        else:
            self.assertEqual(config.bits, expected_bits)


class TestQuantRecipeLayerFilter(unittest.TestCase):
    """Tests for the ``layers`` field on ``QuantRule``."""

    def test_layer_filter(self):
        edge = set(range(5)) | set(range(55, 60))
        recipe = QuantRecipe(
            rules=[
                QuantRule(r".*norm\.weight", None),
                QuantRule(r".*\.(v_proj|down_proj)\.weight", _Q8, layers=edge),
                QuantRule(r".*\.weight", _Q4),
            ]
        )
        # Edge v_proj → 8-bit
        self.assertEqual(recipe.get_config("layers.0.self_attn.v_proj.weight").bits, 8)
        self.assertEqual(recipe.get_config("layers.58.self_attn.v_proj.weight").bits, 8)
        # Middle v_proj → falls through → 4-bit
        self.assertEqual(recipe.get_config("layers.30.self_attn.v_proj.weight").bits, 4)
        # q_proj always 4-bit
        self.assertEqual(recipe.get_config("layers.0.self_attn.q_proj.weight").bits, 4)
        # Non-layer FQN skips layer-filtered rule, hits catch-all
        self.assertEqual(recipe.get_config("lm_head.weight").bits, 4)

    def test_layer_filter_with_none_config(self):
        """Skip rule scoped to specific layers."""
        recipe = QuantRecipe(
            rules=[
                QuantRule(r".*\.weight", None, layers={0}),
                QuantRule(r".*\.weight", _Q4),
            ]
        )
        self.assertIsNone(recipe.get_config("layers.0.mlp.gate_proj.weight"))
        self.assertEqual(recipe.get_config("layers.1.mlp.gate_proj.weight").bits, 4)


class TestProductionRecipes(unittest.TestCase):
    """Regression tests for the production recipes in quantize_and_save.py."""

    def test_default_recipe(self):
        from executorch.examples.models.gemma4_31b.quantize_and_save import (
            GEMMA4_31B_DEFAULT_RECIPE,
        )

        r = GEMMA4_31B_DEFAULT_RECIPE
        self.assertIsNone(r.get_config("layers.0.input_layernorm.weight"))
        self.assertIsNone(r.get_config("layers.5.self_attn.q_norm.weight"))
        self.assertIsNone(r.get_config("norm.weight"))
        embed_cfg = r.get_config("embed_tokens.weight")
        self.assertEqual(embed_cfg.bits, 8)
        self.assertEqual(embed_cfg.group_size, 5376)
        for fqn in (
            "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.v_proj.weight",
            "layers.0.mlp.gate_proj.weight",
            "layers.0.mlp.down_proj.weight",
            "lm_head.weight",
        ):
            cfg = r.get_config(fqn)
            self.assertEqual(cfg.bits, 4, fqn)
            self.assertEqual(cfg.method, "min_max", fqn)

    def test_sensitive_recipe(self):
        from executorch.examples.models.gemma4_31b.quantize_and_save import (
            GEMMA4_31B_SENSITIVE_RECIPE,
        )

        r = GEMMA4_31B_SENSITIVE_RECIPE
        self.assertIsNone(r.get_config("layers.0.input_layernorm.weight"))
        embed_cfg = r.get_config("embed_tokens.weight")
        self.assertEqual(embed_cfg.bits, 8)
        self.assertEqual(embed_cfg.group_size, 5376)
        # Edge v_proj/down_proj → int8
        self.assertEqual(r.get_config("layers.0.self_attn.v_proj.weight").bits, 8)
        self.assertEqual(r.get_config("layers.0.mlp.down_proj.weight").bits, 8)
        self.assertEqual(r.get_config("layers.58.self_attn.v_proj.weight").bits, 8)
        # Middle v_proj/down_proj → int4
        self.assertEqual(r.get_config("layers.30.self_attn.v_proj.weight").bits, 4)
        self.assertEqual(r.get_config("layers.30.mlp.down_proj.weight").bits, 4)
        # q_proj always int4
        self.assertEqual(r.get_config("layers.0.self_attn.q_proj.weight").bits, 4)
        self.assertEqual(r.get_config("layers.30.self_attn.q_proj.weight").bits, 4)


if __name__ == "__main__":
    unittest.main()
