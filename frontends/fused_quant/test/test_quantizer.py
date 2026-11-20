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
    get_qparams_from_node,
    is_dequantize_node,
    is_per_channel_quant_node,
    is_per_tensor_quant_node,
)
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.frontends.fused_quant.quantizer.quantizer import (
    ConvolutionQuantizer,
    FusedQuantQuantizer,
    get_core_aten_quantizers,
    OpQuantizer,
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


class _PrequantizedLinear(nn.Module):
    def __init__(self, granularity: str) -> None:
        super().__init__()
        self.granularity = granularity
        self.weight = nn.Parameter(torch.randn(4, 8))
        shape = (4, 2) if granularity == "group" else (4,)
        self.register_buffer("scale", torch.full(shape, 0.125))
        self.register_buffer("zero_point", torch.zeros(shape, dtype=torch.int64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.granularity == "per_tensor":
            weight = torch.ops.quantized_decomposed.quantize_per_tensor.default(
                self.weight, 0.125, 0, -8, 7, torch.int8
            )
            weight = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                weight, 0.125, 0, -8, 7, torch.int8
            )
        elif self.granularity == "per_channel":
            weight = torch.ops.quantized_decomposed.quantize_per_channel.default(
                self.weight, self.scale, self.zero_point, 0, -8, 7, torch.int8
            )
            weight = torch.ops.quantized_decomposed.dequantize_per_channel.default(
                weight, self.scale, self.zero_point, 0, -8, 7, torch.int8
            )
        else:
            weight = torch.ops.torchao.quantize_affine.default(
                self.weight, [1, 4], self.scale, self.zero_point, torch.int8, -8, 7
            )
            weight = torch.ops.torchao.dequantize_affine.default(
                weight,
                [1, 4],
                self.scale,
                self.zero_point,
                torch.int8,
                quant_min=-8,
                quant_max=7,
                output_dtype=torch.float32,
            )
        return nn.functional.linear(x, weight)


class OpQuantizerTest(unittest.TestCase):
    @parameterized.expand([("per_tensor",), ("per_channel",), ("group",)])
    def test_qat_preserves_weight_quantization(self, granularity: str) -> None:
        torch.manual_seed(0)
        inputs = (torch.randn(2, 8),)
        quantizer = FusedQuantQuantizer(
            [
                OpQuantizer(
                    torch.ops.aten.linear.default,
                    torch.ops.fused_quant.linear.default,
                    _qconfig(weight_per_channel=True),
                    weight_names=("weight",),
                    other_names=("bias",),
                )
            ]
        )
        program = torch.export.export(_PrequantizedLinear(granularity).eval(), inputs)
        prepared = prepare_pt2e(program.module(), quantizer)
        (linear,) = prepared.graph.find_nodes(
            op="call_function", target=torch.ops.aten.linear.default
        )
        self.assertTrue(is_dequantize_node(get_arg(linear, "weight", fx.Node)))
        self.assertEqual(len(prepared.graph.find_nodes(op="call_module")), 2)
        prepared(*inputs)
        converted = convert_pt2e(prepared, fold_quantize=False)
        expected = converted(*inputs)

        quantizer.fuse(converted)

        (fused,) = converted.graph.find_nodes(
            op="call_function", target=torch.ops.fused_quant.linear.default
        )
        qparams = get_qparams_from_node(fused, "weight")
        assert qparams is not None
        self.assertEqual((qparams.quant_min, qparams.quant_max), (-8, 7))
        shape = {"per_tensor": (), "per_channel": (4, 1), "group": (4, 2)}[granularity]
        scale = (
            getattr(converted, str(qparams.scale.target))
            if qparams.scale.op == "get_attr"
            else qparams.scale.meta["val"]
        )
        self.assertEqual(tuple(scale.shape), shape)
        torch.testing.assert_close(converted(*inputs), expected)

    def test_default_still_observes_weights(self) -> None:
        inputs = (torch.randn(2, 8),)
        quantizer = FusedQuantQuantizer(
            [
                OpQuantizer(
                    torch.ops.aten.linear.default,
                    torch.ops.fused_quant.linear.default,
                    _qconfig(weight_per_channel=True),
                    weight_names=("weight",),
                    other_names=("bias",),
                )
            ]
        )
        program = torch.export.export(nn.Linear(8, 4).eval(), inputs)
        prepared = prepare_pt2e(program.module(), quantizer)
        self.assertEqual(len(prepared.graph.find_nodes(op="call_module")), 3)


    def test_mixed_float_and_prequantized_weights(self) -> None:
        class MixedWeights(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.prequantized_weight = nn.Parameter(torch.randn(4, 8))
                self.float_weight = nn.Parameter(torch.randn(4, 8))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                weight = torch.ops.quantized_decomposed.quantize_per_tensor.default(
                    self.prequantized_weight, 0.125, 0, -8, 7, torch.int8
                )
                weight = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                    weight, 0.125, 0, -8, 7, torch.int8
                )
                return nn.functional.linear(x, weight) + nn.functional.linear(
                    x, self.float_weight
                )

        inputs = (torch.randn(2, 8),)
        config = _qconfig(weight_per_channel=True)
        quantizer = FusedQuantQuantizer(
            [
                OpQuantizer(
                    torch.ops.aten.linear.default,
                    torch.ops.fused_quant.linear.default,
                    config,
                    weight_names=("weight",),
                    other_names=("bias",),
                )
            ]
        )
        program = torch.export.export(MixedWeights().eval(), inputs)
        prepared = prepare_pt2e(program.module(), quantizer)
        linears = prepared.graph.find_nodes(
            op="call_function", target=torch.ops.aten.linear.default
        )
        self.assertTrue(is_dequantize_node(get_arg(linears[0], "weight", fx.Node)))
        self.assertEqual(get_arg(linears[1], "weight", fx.Node).op, "call_module")
        prepared(*inputs)
        converted = convert_pt2e(prepared)
        expected = converted(*inputs)
        quantizer.fuse(converted)
        linears = converted.graph.find_nodes(
            op="call_function", target=torch.ops.fused_quant.linear.default
        )
        ranges = []
        for linear in linears:
            qparams = get_qparams_from_node(linear, "weight")
            assert qparams is not None
            ranges.append((qparams.quant_min, qparams.quant_max))
        self.assertEqual(ranges, [(-8, 7), (-127, 127)])
        torch.testing.assert_close(converted(*inputs), expected)

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
                "div",
                _Binary(torch.div),
                [vector, vector],
                aten.div.Tensor,
                fused_quant.div.default,
            ),
            (
                "div_scalar",
                _Unary(lambda x: x / 4.0),
                [vector],
                aten.div.Tensor,
                fused_quant.div.Scalar,
            ),
            (
                "minimum",
                _Binary(torch.minimum),
                [vector, vector],
                aten.minimum.default,
                fused_quant.minimum.default,
            ),
            (
                "maximum",
                _Binary(torch.maximum),
                [vector, vector],
                aten.maximum.default,
                fused_quant.maximum.default,
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
                "leaky_relu",
                nn.LeakyReLU(0.1),
                [vector],
                aten.leaky_relu.default,
                fused_quant.leaky_relu.default,
            ),
            (
                "rsqrt",
                _Unary(lambda x: torch.rsqrt(x.abs() + 1.0)),
                [vector],
                aten.rsqrt.default,
                fused_quant.rsqrt.default,
            ),
            (
                "sqrt",
                _Unary(lambda x: torch.sqrt(x.abs())),
                [vector],
                aten.sqrt.default,
                fused_quant.sqrt.default,
            ),
            (
                "pow",
                _Unary(lambda x: x.pow(2)),
                [vector],
                aten.pow.Tensor_Scalar,
                fused_quant.pow.Tensor_Scalar,
            ),
            (
                "clamp",
                _Unary(lambda x: torch.clamp(x, min=-0.5, max=0.5)),
                [vector],
                aten.clamp.default,
                fused_quant.clamp.default,
            ),
            (
                "mean",
                _Unary(lambda x: x.mean(dim=-1, keepdim=True)),
                [vector],
                aten.mean.dim,
                fused_quant.mean.dim,
            ),
            (
                "sum",
                _Unary(lambda x: x.sum(dim=-1, keepdim=True)),
                [vector],
                aten.sum.dim_IntList,
                fused_quant.sum.dim_IntList,
            ),
            (
                "amax",
                _Unary(lambda x: x.amax(dim=-1, keepdim=True)),
                [vector],
                aten.amax.default,
                fused_quant.amax.default,
            ),
            (
                "softmax",
                nn.Softmax(dim=-1),
                [vector],
                aten._softmax.default,
                fused_quant._softmax.default,
            ),
            (
                "log_softmax",
                nn.LogSoftmax(dim=-1),
                [vector],
                aten._log_softmax.default,
                fused_quant._log_softmax.default,
            ),
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
