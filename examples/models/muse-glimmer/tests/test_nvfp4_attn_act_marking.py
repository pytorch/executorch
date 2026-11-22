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


if __name__ == "__main__":
    unittest.main()
