# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import types
import unittest

import torch

from executorch.backends.cuda.nvfp4_tensor import CudaNVFP4Tensor
from executorch.examples.models.muse_glimmer.source_transformations.nvfp4_attn_act import (
    mark_attention_act,
)


def make_weight(n, k):
    q = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8)
    s = torch.randint(48, 64, (n, k // 16), dtype=torch.uint8)
    return CudaNVFP4Tensor(q, s, torch.tensor([0.5]))


class TestMarkAttentionAct(unittest.TestCase):
    def test_mark_attention_act(self):
        def linear(n, k):
            lin = torch.nn.Linear(k, n, bias=False)
            lin.weight = torch.nn.Parameter(make_weight(n, k), requires_grad=False)
            return lin

        layers = []
        for _ in range(2):
            attn = torch.nn.Module()
            attn.qko_proj, attn.v_proj, attn.o_proj = (
                linear(64, 256),
                linear(32, 256),
                linear(256, 64),
            )
            mlp = torch.nn.Module()
            mlp.gate_up_proj, mlp.down_proj = linear(128, 256), linear(256, 64)
            layers.append(types.SimpleNamespace(self_attn=attn, mlp=mlp))
        model = types.SimpleNamespace(layers=layers)
        self.assertEqual(mark_attention_act(model, "mxfp8"), 6)
        for layer in layers:
            for name in ("qko_proj", "v_proj", "o_proj"):
                self.assertEqual(
                    getattr(layer.self_attn, name).weight.act_dtype, "mxfp8"
                )
            for name in ("gate_up_proj", "down_proj"):
                self.assertEqual(getattr(layer.mlp, name).weight.act_dtype, "fp4")

    def test_mark_attention_act_separate_projections(self):
        attn = torch.nn.Module()
        for name in ("q_proj", "k_proj", "og_proj", "v_proj", "o_proj"):
            lin = torch.nn.Linear(256, 64, bias=False)
            lin.weight = torch.nn.Parameter(make_weight(64, 256), requires_grad=False)
            setattr(attn, name, lin)
        attn.q_norm = torch.nn.LayerNorm(64)
        model = types.SimpleNamespace(layers=[types.SimpleNamespace(self_attn=attn)])
        self.assertEqual(mark_attention_act(model, "bf16"), 5)
        self.assertEqual(attn.q_proj.weight.act_dtype, "bf16")

    def test_mark_attention_act_requires_nvfp4_projections(self):
        attn = torch.nn.Module()
        attn.qko_proj = torch.nn.Linear(256, 64, bias=False)
        model = types.SimpleNamespace(layers=[types.SimpleNamespace(self_attn=attn)])
        with self.assertRaises(ValueError):
            mark_attention_act(model, "mxfp8")


if __name__ == "__main__":
    unittest.main()
