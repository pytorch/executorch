# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.test.harness.stages import StageType
from executorch.backends.xnnpack._passes.insert_pad_qdq import InsertPadQDQPass
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
)
from executorch.backends.xnnpack.test.tester import Quantize, RunPasses, Tester
from executorch.exir.dialects._ops import ops as exir_ops


class TestInsertPadQDQ(unittest.TestCase):
    Q = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
    PAD = exir_ops.edge.aten.constant_pad_nd.default

    class PadConv(torch.nn.Module):
        # Even kernel + 'same' padding decomposes into constant_pad_nd -> conv.
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(2, 2, (2, 2), padding="same")

        def forward(self, x):
            return self.conv(x)

    def _run(self, passes, quantize):
        tester = Tester(self.PadConv().eval(), (torch.randn(1, 2, 8, 8),))
        if quantize:
            tester.quantize(
                Quantize(quantization_config=get_symmetric_quantization_config())
            )
        artifact = (
            tester.export()
            .to_edge()
            .run_passes(RunPasses(passes))
            .get_artifact(StageType.RUN_PASSES)
        )
        return artifact.exported_program().graph_module.graph

    def _pad(self, graph):
        return next(n for n in graph.nodes if n.target == self.PAD)

    def _num_quant(self, graph):
        return sum(1 for n in graph.nodes if n.target == self.Q)

    def test_inserts_qdq_after_quantized_pad(self):
        # dequant -> pad -> conv: the pass quantizes the pad's output, so the pad
        # is now consumed by an inserted quantize.
        graph = self._run([InsertPadQDQPass], quantize=True)
        pad = self._pad(graph)
        self.assertTrue(all(user.target == self.Q for user in pad.users))

    def test_idempotent_skips_already_quantized_pad(self):
        # A second run must see the pad's user is already a quantize and skip it,
        # so no extra quantize is inserted.
        once = self._num_quant(self._run([InsertPadQDQPass], quantize=True))
        twice = self._num_quant(
            self._run([InsertPadQDQPass, InsertPadQDQPass], quantize=True)
        )
        self.assertEqual(once, twice)

    def test_noop_for_fp32_pad(self):
        # The pad is not fed by a dequant, so the pass leaves the graph untouched.
        graph = self._run([InsertPadQDQPass], quantize=False)
        self.assertEqual(self._num_quant(graph), 0)
        pad = self._pad(graph)
        self.assertFalse(any(user.target == self.Q for user in pad.users))
