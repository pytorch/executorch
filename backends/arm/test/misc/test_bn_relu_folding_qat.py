# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import TosaPipelineINT
from executorch.backends.arm.tosa import TosaSpecification

from executorch.backends.test.harness.tester import Quantize
from torch import nn
from torch.fx import Node
from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e


input_t1 = Tuple[torch.Tensor]  # Input x


class Conv2dModule(torch.nn.Module):
    input_shape = (1, 28, 28)
    batch_size = 64
    test_data: input_t1 = (torch.randn(batch_size, *input_shape),)

    def __init__(self, batch_norm: bool = True, inplace: bool = False) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 16, 3, stride=2)
        self.bn = nn.BatchNorm2d(num_features=16) if batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class Conv1dModule(torch.nn.Module):
    input_shape = (3, 10)
    batch_size = 2
    test_data: input_t1 = (torch.randn(batch_size, *input_shape),)

    def __init__(self, batch_norm: bool = True, inplace: bool = False) -> None:
        super().__init__()
        self.conv = torch.nn.Conv1d(3, 8, 5, padding=2)
        self.bn = nn.BatchNorm1d(num_features=8) if batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


models = {
    # name : (model, is_per_channel)
    "conv1d_bn_relu_per_channel": (Conv1dModule(batch_norm=True), True),
    "conv1d_relu_per_channel": (Conv1dModule(batch_norm=False), True),
    "conv1d_bn_relu_per_tensor": (Conv1dModule(batch_norm=True), False),
    "conv1d_relu_per_tensor": (Conv1dModule(batch_norm=False), False),
    "conv2d_bn_relu_per_channel": (Conv2dModule(batch_norm=True), True),
    "conv2d_relu_per_channel": (Conv2dModule(batch_norm=False), True),
    "conv2d_bn_relu_per_tensor": (Conv2dModule(batch_norm=True), False),
    "conv2d_relu_per_tensor": (Conv2dModule(batch_norm=False), False),
    "conv1d_bn_relu_inplace_per_channel": (
        Conv1dModule(batch_norm=True, inplace=True),
        True,
    ),
    "conv1d_relu_inplace_per_channel": (
        Conv1dModule(batch_norm=False, inplace=True),
        True,
    ),
    "conv1d_bn_relu_inplace_per_tensor": (
        Conv1dModule(batch_norm=True, inplace=True),
        False,
    ),
    "conv1d_relu_inplace_per_tensor": (
        Conv1dModule(batch_norm=False, inplace=True),
        False,
    ),
    "conv2d_bn_relu_inplace_per_channel": (
        Conv2dModule(batch_norm=True, inplace=True),
        True,
    ),
    "conv2d_relu_inplace_per_channel": (
        Conv2dModule(batch_norm=False, inplace=True),
        True,
    ),
    "conv2d_bn_relu_inplace_per_tensor": (
        Conv2dModule(batch_norm=True, inplace=True),
        False,
    ),
    "conv2d_relu_inplace_per_tensor": (
        Conv2dModule(batch_norm=False, inplace=True),
        False,
    ),
}


@common.parametrize(
    "test_data",
    models,
)
def test_bn_relu_folding_qat_tosa_INT(test_data):
    model, per_channel = test_data
    pipeline = TosaPipelineINT[input_t1](model, model.test_data, [], [], qtol=1)
    quantizer = TOSAQuantizer(TosaSpecification.create_from_string("TOSA-1.0+INT"))
    pipeline.change_args(
        "quantize",
        Quantize(
            quantizer=quantizer,
            quantization_config=get_symmetric_quantization_config(
                is_qat=True, is_per_channel=per_channel
            ),
        ),
    )
    pipeline.run()


def test_qat_batch_norm_updates_registered_running_stats():
    class ConvBatchNorm(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv1d(2, 2, 1, bias=False)
            self.bn = nn.BatchNorm1d(2)
            with torch.no_grad():
                self.conv.weight.copy_(torch.eye(2).unsqueeze(-1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.bn(self.conv(x))

    sample = torch.stack((torch.full((4, 8), 3.0), torch.full((4, 8), -2.0)), dim=1)
    model = torch.export.export(
        ConvBatchNorm().train(), (sample,), strict=True
    ).module()
    quantizer = TOSAQuantizer(
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
        use_composable_quantizer=True,
    ).set_global(get_symmetric_quantization_config(is_qat=True, is_per_channel=False))

    prepared = prepare_qat_pt2e(model, quantizer)
    batch_norm_nodes = [
        node
        for node in prepared.graph.nodes
        if node.target == torch.ops.aten.batch_norm.default
    ]
    assert len(batch_norm_nodes) == 1
    batch_norm_node = batch_norm_nodes[0]

    for state_node in batch_norm_node.args[1:5]:
        assert isinstance(state_node, Node)
        assert state_node.op == "get_attr"

    running_mean_node = batch_norm_node.args[3]
    running_var_node = batch_norm_node.args[4]
    assert isinstance(running_mean_node, Node)
    assert isinstance(running_var_node, Node)
    running_mean = prepared.get_buffer(str(running_mean_node.target))
    running_var = prepared.get_buffer(str(running_var_node.target))
    initial_running_mean = running_mean.clone()
    initial_running_var = running_var.clone()

    prepared(sample)

    assert not torch.equal(running_mean, initial_running_mean)
    assert not torch.equal(running_var, initial_running_var)
