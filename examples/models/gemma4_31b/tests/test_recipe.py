# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for the gemma4_31b production quant recipes."""

import unittest


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
