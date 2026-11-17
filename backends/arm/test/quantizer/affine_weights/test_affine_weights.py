# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch
import tosa_serializer as ts
from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.backends.arm.operators.ops_quant_utils import add_input_weight_zp_consts
from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_quantization_config,
)
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from executorch.backends.arm.test.tester.test_pipeline import TosaPipelineINT
from executorch.backends.test.harness.stages import StageType
from torchao.quantization.pt2e.quantizer import (
    DerivedQuantizationSpec,
    QuantizationSpec,
)


def _affine_config(per_channel: bool) -> QuantizationConfig:
    config = get_symmetric_quantization_config(is_per_channel=per_channel)
    weight = config.weight
    assert isinstance(weight, QuantizationSpec)
    return replace(
        config,
        weight=replace(
            weight,
            qscheme=(
                torch.per_channel_affine if per_channel else torch.per_tensor_affine
            ),
            quant_min=-128,
        ),
    )


@pytest.mark.parametrize("per_channel", [False, True])
def test_affine_weight_specs_and_derived_channel_bias(per_channel: bool) -> None:
    config = _affine_config(per_channel)
    assert config.get_weight_qspec() is config.weight
    graph = torch.fx.Graph()
    node = graph.call_function(
        torch.ops.aten.linear.default,
        (
            graph.placeholder("input"),
            graph.placeholder("weight"),
            graph.placeholder("bias"),
        ),
    )
    bias = config.get_bias_qspec(node)
    assert isinstance(bias, DerivedQuantizationSpec)
    assert bias.ch_axis == (0 if per_channel else None)


def test_equal_channel_weight_zero_points_serialize_as_one_scalar() -> None:
    graph = torch.fx.Graph()
    node = graph.placeholder("conv")
    node.meta["input_qparams"] = {
        0: QuantArgs(0.1, 0, -127, 127, torch.int8),
        1: QuantArgs([0.01, 0.02], [-28, -28], -128, 127, torch.int8, per_channel=True),
    }
    inputs = [Mock(dtype=ts.DType.INT8), Mock(dtype=ts.DType.INT8)]
    serializer = Mock()
    input_const, weight_const = Mock(), Mock()
    input_const.name, weight_const.name = "input_zp", "weight_zp"
    serializer.addConst.side_effect = [input_const, weight_const]
    add_input_weight_zp_consts(serializer, node, inputs, "conv")
    assert serializer.addConst.call_args_list[1].args == ([1], ts.DType.INT8, [-28])
    node.meta["input_qparams"][1] = QuantArgs(
        [0.01, 0.02], [-28, -27], -128, 127, torch.int8, per_channel=True
    )
    with pytest.raises(ValueError, match="common weight zero point"):
        add_input_weight_zp_consts(serializer, node, inputs, "conv")


def _affine_model(
    operator: str, conflicting_zero_points: bool = False
) -> tuple[torch.nn.Module, tuple[torch.Tensor]]:
    model: torch.nn.Linear | torch.nn.Conv2d | torch.nn.ConvTranspose2d
    if operator == "linear":
        model = torch.nn.Linear(8, 3)
        inputs = (torch.linspace(-1, 1, 8).reshape(1, 8),)
    else:
        model = (
            torch.nn.ConvTranspose2d(2, 3, 2)
            if operator == "conv_transpose2d"
            else torch.nn.Conv2d(2, 3, 2)
        )
        inputs = (torch.linspace(-1, 1, 72).reshape(1, 2, 6, 6),)
    axis = 1 if operator == "conv_transpose2d" else 0
    channel_shape = model.weight.movedim(axis, 0).shape
    base = torch.linspace(-0.25, 1, model.weight.numel() // 3).reshape(
        channel_shape[1:]
    )
    weights = torch.stack([base * scale for scale in (1, 2, 3)])
    if conflicting_zero_points:
        weights[0] += 0.25
    with torch.no_grad():
        model.weight.copy_(weights.movedim(0, axis))
        assert model.bias is not None
        model.bias.copy_(torch.tensor([0.1, 0.2, 0.3]))
    return model.eval(), inputs


def _check_affine_qparams(
    pipeline: TosaPipelineINT[tuple[torch.Tensor]], per_channel: bool
) -> None:
    graph_module = pipeline.tester.get_artifact(StageType.QUANTIZE)
    op = (
        torch.ops.quantized_decomposed.dequantize_per_channel.default
        if per_channel
        else torch.ops.quantized_decomposed.dequantize_per_tensor.default
    )
    weights = [
        node
        for node in graph_module.graph.nodes
        if node.target == op and node.args[-1] == torch.int8
    ]
    if per_channel:
        assert len(weights) == 1
        scales = graph_module.get_buffer(str(weights[0].args[1].target))
        zero_points = graph_module.get_buffer(str(weights[0].args[2].target))
        assert scales.numel() == 3
        assert torch.unique(scales).numel() == 3
        assert torch.unique(zero_points).numel() == 1
        assert zero_points[0] != 0
        biases = [
            node
            for node in graph_module.graph.nodes
            if node.target == op and node.args[-1] == torch.int32
        ]
        assert len(biases) == 1
        assert biases[0].args[3] == 0
        assert graph_module.get_buffer(str(biases[0].args[1].target)).numel() == 3
    else:
        weight = next(node for node in weights if node.args[0].op == "get_attr")
        assert weight.args[2] != 0


@pytest.mark.parametrize("operator", ["linear", "conv2d", "conv_transpose2d"])
@pytest.mark.parametrize("per_channel", [False, True])
def test_affine_weights_tosa_pipeline(operator: str, per_channel: bool) -> None:
    model, inputs = _affine_model(operator)
    pipeline = TosaPipelineINT[tuple[torch.Tensor]](
        model,
        inputs,
        aten_op=f"torch.ops.aten.{operator}.{'input' if operator == 'conv_transpose2d' else 'default'}",
        per_channel_quantization=per_channel,
    )
    pipeline.quantizer.set_global(_affine_config(per_channel))
    pipeline.add_stage_after("quantize", _check_affine_qparams, pipeline, per_channel)
    pipeline.run()
    assert pipeline.has_stage("run_method_and_compare_outputs")


def test_conflicting_affine_weight_zero_points_rejected_by_pipeline() -> None:
    model, inputs = _affine_model("conv2d", conflicting_zero_points=True)
    pipeline = TosaPipelineINT[tuple[torch.Tensor]](
        model, inputs, aten_op="torch.ops.aten.conv2d.default"
    )
    pipeline.quantizer.set_global(_affine_config(True))
    with pytest.raises(ValueError, match="common weight zero point"):
        pipeline.run()
