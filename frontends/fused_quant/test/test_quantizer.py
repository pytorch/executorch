# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import unittest
from collections.abc import Callable

import executorch.exir.dialects.fused_quant.ops  # noqa: F401
import torch
from executorch.backends.transforms.fused_quant.graph_utils import (
    is_per_channel_quant_node,
    is_per_tensor_quant_node,
)
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.frontends.fused_quant.quantizer.quantizer import (
    ConvolutionQuantizer,
    FusedQuantQuantizer,
    get_core_aten_quantizers,
)
from parameterized import parameterized
from torch import fx, nn
from torchao.quantization.pt2e import MinMaxObserver, PerChannelMinMaxObserver
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.quantization.pt2e.quantizer import QuantizationConfig, QuantizationSpec


def _qconfig(weight_per_channel: bool) -> QuantizationConfig:
    activation = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_affine,
        observer_or_fake_quant_ctr=MinMaxObserver,
    )
    weight = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-127,
        quant_max=127,
        qscheme=(
            torch.per_channel_symmetric
            if weight_per_channel
            else torch.per_tensor_symmetric
        ),
        ch_axis=0 if weight_per_channel else None,
        observer_or_fake_quant_ctr=(
            PerChannelMinMaxObserver if weight_per_channel else MinMaxObserver
        ),
    )
    return QuantizationConfig(activation, activation, weight, None)


def _core_quantizer() -> FusedQuantQuantizer:
    return FusedQuantQuantizer(
        get_core_aten_quantizers(
            _qconfig(weight_per_channel=False), _qconfig(weight_per_channel=True)
        )
    )


def _quantize(
    model: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    quantizer: FusedQuantQuantizer,
) -> fx.GraphModule:
    """Export to core ATen, then annotate, calibrate and convert, ready to fuse."""
    exported = torch.export.export(model.eval(), inputs).run_decompositions()
    prepared = prepare_pt2e(exported.module(), quantizer)
    prepared(*inputs)
    return convert_pt2e(prepared)


class _Unary(nn.Module):
    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


class _Binary(nn.Module):
    def __init__(
        self, fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.fn(x, y)


class GetCoreAtenQuantizersTest(unittest.TestCase):
    def test_every_op_fuses(self) -> None:
        """Each op reaches its quantizer through export's decompositions and fuses.

        A Python scalar operand keeps ``aten.add.Tensor`` (and mul, sub), so the
        scalar cases check that those fuse into the ``.Scalar`` overload.
        """
        torch.manual_seed(0)
        aten, fused_quant = torch.ops.aten, torch.ops.fused_quant
        vector, image = (2, 8), (1, 3, 6, 6)
        cases = [
            (
                "add",
                _Binary(torch.add),
                [vector, vector],
                aten.add.Tensor,
                fused_quant.add.default,
            ),
            (
                "add_scalar",
                _Unary(lambda x: x + 1.0),
                [vector],
                aten.add.Tensor,
                fused_quant.add.Scalar,
            ),
            (
                "mul",
                _Binary(torch.mul),
                [vector, vector],
                aten.mul.Tensor,
                fused_quant.mul.default,
            ),
            (
                "mul_scalar",
                _Unary(lambda x: x * 2.0),
                [vector],
                aten.mul.Tensor,
                fused_quant.mul.Scalar,
            ),
            (
                "sub",
                _Binary(torch.sub),
                [vector, vector],
                aten.sub.Tensor,
                fused_quant.sub.default,
            ),
            (
                "sub_scalar",
                _Unary(lambda x: x - 1.0),
                [vector],
                aten.sub.Tensor,
                fused_quant.sub.Scalar,
            ),
            (
                "bmm",
                _Binary(torch.bmm),
                [(2, 3, 4), (2, 4, 5)],
                aten.bmm.default,
                fused_quant.bmm.default,
            ),
            ("relu", nn.ReLU(), [vector], aten.relu.default, fused_quant.relu.default),
            (
                "max_pool2d",
                nn.MaxPool2d(2),
                [image],
                aten.max_pool2d_with_indices.default,
                fused_quant.max_pool2d_with_indices.default,
            ),
            (
                "avg_pool2d",
                nn.AvgPool2d(2),
                [image],
                aten.avg_pool2d.default,
                fused_quant.avg_pool2d.default,
            ),
            (
                "convolution",
                nn.Conv2d(3, 4, 3),
                [image],
                aten.convolution.default,
                fused_quant.convolution.default,
            ),
            (
                "sigmoid",
                nn.Sigmoid(),
                [vector],
                aten.sigmoid.default,
                fused_quant.sigmoid.default,
            ),
            ("tanh", nn.Tanh(), [vector], aten.tanh.default, fused_quant.tanh.default),
            (
                "hardtanh",
                nn.Hardtanh(),
                [vector],
                aten.hardtanh.default,
                fused_quant.hard_tanh.default,
            ),
            ("gelu", nn.GELU(), [vector], aten.gelu.default, fused_quant.gelu.default),
            (
                "layer_norm",
                nn.LayerNorm(8),
                [vector],
                aten.native_layer_norm.default,
                fused_quant.native_layer_norm.default,
            ),
        ]
        for name, model, shapes, aten_op, fused_op in cases:
            with self.subTest(name):
                inputs = tuple(torch.randn(shape) for shape in shapes)
                quantizer = _core_quantizer()
                graph_module = _quantize(model, inputs, quantizer)

                quantizer.fuse(graph_module)

                targets = {node.target for node in graph_module.graph.nodes}
                self.assertIn(fused_op, targets)
                self.assertNotIn(aten_op, targets)


class ConvolutionQuantizerTest(unittest.TestCase):
    @parameterized.expand(
        [
            # The weight is (C_out, C_in, kH, kW): output channels on axis 0.
            ("convolution", nn.Conv2d, 1, 0),
            # Transposed, it is (C_in, C_out, kH, kW): output channels on axis 1.
            ("transposed", nn.ConvTranspose2d, 1, 1),
            # Grouped, axis 1 holds only C_out // groups channels: per tensor.
            ("grouped_transposed", nn.ConvTranspose2d, 2, None),
        ]
    )
    def test_weight_axis(
        self,
        _name: str,
        conv_type: Callable[..., nn.Module],
        groups: int,
        axis: int | None,
    ) -> None:
        torch.manual_seed(0)
        inputs = (torch.randn(1, 4, 6, 6),)
        graph_module = _quantize(
            conv_type(4, 8, 3, groups=groups), inputs, _core_quantizer()
        )

        (conv,) = graph_module.graph.find_nodes(
            op="call_function", target=torch.ops.aten.convolution.default
        )
        weight = get_arg(conv, "weight", fx.Node)
        if axis is None:
            self.assertTrue(is_per_tensor_quant_node(weight), str(weight.target))
        else:
            self.assertTrue(is_per_channel_quant_node(weight), str(weight.target))
            self.assertEqual(axis, get_arg(weight, "axis", int))

    def test_grouped_transposed_config_needs_per_tensor_weights(self) -> None:
        per_tensor = _qconfig(weight_per_channel=False)
        per_channel = _qconfig(weight_per_channel=True)
        for name, weight in (
            ("unquantized", None),
            ("per_channel", per_channel.weight),
        ):
            with self.subTest(name), self.assertRaisesRegex(
                ValueError, "grouped_transposed_config"
            ):
                ConvolutionQuantizer(
                    weighted_config=per_channel,
                    grouped_transposed_config=dataclasses.replace(
                        per_tensor, weight=weight
                    ),
                )
