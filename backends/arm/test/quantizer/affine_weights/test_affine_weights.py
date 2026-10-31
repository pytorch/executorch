# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import replace
from unittest.mock import Mock

import torch
import tosa_serializer as ts
from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.backends.arm.operators.ops_quant_utils import add_input_weight_zp_consts
from executorch.backends.arm.quantizer.arm_quantizer import get_symmetric_quantization_config
from torchao.quantization.pt2e.quantizer import DerivedQuantizationSpec, QuantizationSpec


class AffineWeightsTest(unittest.TestCase):
    def test_affine_weight_specs_and_derived_channel_bias_are_supported(self) -> None:
        for channel in (False, True):
            with self.subTest(channel=channel):
                config = get_symmetric_quantization_config(is_per_channel=channel)
                self.assertIsInstance(config.weight, QuantizationSpec)
                spec = replace(config.weight, qscheme=torch.per_channel_affine if channel else torch.per_tensor_affine, quant_min=-128)
                config = replace(config, weight=spec)
                self.assertIs(config.get_weight_qspec(), spec)
                graph = torch.fx.Graph()
                node = graph.call_function(torch.ops.aten.linear.default, (graph.placeholder("input"), graph.placeholder("weight"), graph.placeholder("bias")))
                bias = config.get_bias_qspec(node)
                self.assertIsInstance(bias, DerivedQuantizationSpec)
                self.assertEqual(bias.ch_axis, 0 if channel else None)

    def test_equal_channel_weight_zero_points_serialize_as_one_scalar(self) -> None:
        graph = torch.fx.Graph()
        node = graph.placeholder("conv")
        node.meta["input_qparams"] = {0: QuantArgs(0.1, 0, -127, 127, torch.int8), 1: QuantArgs([0.01, 0.02], [-28, -28], -128, 127, torch.int8, per_channel=True)}
        inputs = [Mock(dtype=ts.DType.INT8), Mock(dtype=ts.DType.INT8)]
        serializer = Mock()
        input_const, weight_const = Mock(), Mock()
        input_const.name, weight_const.name = "input_zp", "weight_zp"
        serializer.addConst.side_effect = [input_const, weight_const]
        add_input_weight_zp_consts(serializer, node, inputs, "conv")
        self.assertEqual(serializer.addConst.call_args_list[1].args, ([1], ts.DType.INT8, [-28]))
        node.meta["input_qparams"][1] = QuantArgs([0.01, 0.02], [-28, -27], -128, 127, torch.int8, per_channel=True)
        with self.assertRaisesRegex(ValueError, "common weight zero point"):
            add_input_weight_zp_consts(serializer, node, inputs, "conv")
