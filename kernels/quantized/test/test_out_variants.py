#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.ao.quantization.fx._decomposed  # noqa[F401] 'torch.ao.quantization.fx._decomposed' imported but unused
from executorch.exir.dialects._ops import ops


class TestOutVariants(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_choose_qparams_to_out_variant_works(self) -> None:
        self.assertIsNotNone(ops.edge.quantized_decomposed.choose_qparams.Tensor_out)
        choose_qparams = ops.edge.quantized_decomposed.choose_qparams.tensor
        out_variant = choose_qparams.to_out_variant()
        self.assertEqual(
            out_variant.name(), "quantized_decomposed::choose_qparams.Tensor_out"
        )
