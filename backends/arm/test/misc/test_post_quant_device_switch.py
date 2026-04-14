# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from dataclasses import dataclass
from typing import Callable

import pytest
import torch
import torch.nn.functional as F
from executorch.backends.arm.quantizer import (
    get_symmetric_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.exir.passes import ToDevicePass
from torch._subclasses.fake_tensor import FakeTensor
from torchao.quantization.pt2e import move_exported_model_to_eval
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_qat_pt2e


class AddAlpha(torch.nn.Module):
    def forward(self, x, y):
        return torch.add(x, y, alpha=2.0)


class SubAlpha(torch.nn.Module):
    def forward(self, x, y):
        return torch.sub(x, y, alpha=2.0)


class SliceScatter(torch.nn.Module):
    def forward(self, x, src):
        return torch.slice_scatter(x, src, dim=1, start=0, end=4, step=2)


class MeanDim(torch.nn.Module):
    def forward(self, x):
        return torch.mean(x, dim=(1,), keepdim=True)


class MeanDefault(torch.nn.Module):
    def forward(self, x):
        return torch.mean(x)


class VarCorrection(torch.nn.Module):
    def forward(self, x):
        return torch.var(x, dim=(2, 3), correction=1, keepdim=True)


class VarDim(torch.nn.Module):
    def forward(self, x):
        return torch.ops.aten.var.dim(x, [2, 3], 1, True)


class DivTensorMode(torch.nn.Module):
    def forward(self, x, y):
        return torch.div(x, y, rounding_mode="trunc")


class LeakyRelu(torch.nn.Module):
    def forward(self, x):
        return F.leaky_relu(x, negative_slope=0.2)


class AvgPool2d(torch.nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=2, stride=1, padding=1)


class LayerNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(4, elementwise_affine=False)

    def forward(self, x):
        return self.layer_norm(x)


class GroupNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.group_norm = torch.nn.GroupNorm(2, 4, affine=False)

    def forward(self, x):
        return self.group_norm(x)


@dataclass(frozen=True)
class MetaRetraceCase:
    name: str
    module_factory: Callable[[], torch.nn.Module]
    inputs_factory: Callable[[], tuple[torch.Tensor, ...]]
    aten_op: str


_TEST_CASES = [
    MetaRetraceCase(
        "add_alpha",
        AddAlpha,
        lambda: (torch.randn(2, 3), torch.randn(2, 3)),
        "aten.add.Tensor",
    ),
    MetaRetraceCase(
        "sub_alpha",
        SubAlpha,
        lambda: (torch.randn(2, 3), torch.randn(2, 3)),
        "aten.sub.Tensor",
    ),
    MetaRetraceCase(
        "slice_scatter",
        SliceScatter,
        lambda: (torch.randn(2, 4), torch.randn(2, 2)),
        "aten.slice_scatter.default",
    ),
    MetaRetraceCase(
        "mean_dim",
        MeanDim,
        lambda: (torch.randn(2, 3, 4),),
        "aten.mean.dim",
    ),
    MetaRetraceCase(
        "mean_default",
        MeanDefault,
        lambda: (torch.randn(2, 3, 4),),
        "aten.mean.default",
    ),
    MetaRetraceCase(
        "var_correction",
        VarCorrection,
        lambda: (torch.randn(2, 3, 4, 4),),
        "aten.var.correction",
    ),
    MetaRetraceCase(
        "var_dim",
        VarDim,
        lambda: (torch.randn(2, 3, 4, 4),),
        "aten.var.dim",
    ),
    MetaRetraceCase(
        "div_tensor_mode",
        DivTensorMode,
        lambda: (torch.randn(2, 3), torch.randn(2, 3) + 1.0),
        "aten.div.Tensor_mode",
    ),
    MetaRetraceCase(
        "leaky_relu",
        LeakyRelu,
        lambda: (torch.randn(2, 3),),
        "aten.leaky_relu.default",
    ),
    MetaRetraceCase(
        "avg_pool2d",
        AvgPool2d,
        lambda: (torch.randn(1, 3, 4, 4),),
        "aten.avg_pool2d.default",
    ),
    MetaRetraceCase(
        "layer_norm",
        LayerNorm,
        lambda: (torch.randn(2, 3, 4),),
        "aten.layer_norm.default",
    ),
    MetaRetraceCase(
        "group_norm",
        GroupNorm,
        lambda: (torch.randn(2, 4, 3, 3),),
        "aten.group_norm.default",
    ),
]


def _make_quantizer() -> TOSAQuantizer:
    quantizer = TOSAQuantizer(TosaSpecification.create_from_string("TOSA-1.0+INT"))
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=False))
    return quantizer


def _iter_fake_tensors(meta_val):
    if isinstance(meta_val, FakeTensor):
        yield meta_val
        return

    if isinstance(meta_val, (list, tuple)):
        for item in meta_val:
            yield from _iter_fake_tensors(item)


def _to_meta_inputs(
    example_inputs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    return tuple(inp.to(device="meta") for inp in example_inputs)


@pytest.mark.parametrize("case", _TEST_CASES, ids=[case.name for case in _TEST_CASES])
def test_post_quant_device_switch(case: MetaRetraceCase) -> None:
    """This test tests that moving a model to another device after quantiation
    works.
    """
    module = case.module_factory().train()
    example_inputs = case.inputs_factory()

    # Quantize module
    exported = torch.export.export(module, example_inputs, strict=True)
    prepared = prepare_qat_pt2e(copy.deepcopy(exported.graph_module), _make_quantizer())
    prepared(*example_inputs)
    prepared = move_exported_model_to_eval(prepared)
    quantized_module = convert_pt2e(prepared)

    # Move and test running the model with other device.
    meta_inputs = _to_meta_inputs(example_inputs)
    meta_module = ToDevicePass("meta")(quantized_module).graph_module
    meta_module(*meta_inputs)

    # Retrace module using meta device to check all fake tensors are moved.
    meta_module = torch.export.export(meta_module, meta_inputs, strict=True)

    # Validate transformation.
    fake_tensor_devices = [
        (str(fake_tensor.device), str(node))
        for node in meta_module.graph.nodes
        for fake_tensor in _iter_fake_tensors(node.meta.get("val"))
    ]

    assert fake_tensor_devices, "Expected traced graph to contain FakeTensor metadata"
    assert all(device == "meta" for device, _ in fake_tensor_devices), (
        "Expected all traced FakeTensors to use the meta device, got "
        f"{fake_tensor_devices}"
    )
