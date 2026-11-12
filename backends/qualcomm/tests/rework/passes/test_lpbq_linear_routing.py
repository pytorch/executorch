# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# HTP lowers per-block (LPBQ) linears as 1x1 conv2d to avoid the FullyConnected
# block-scale over-read (pytorch/executorch#23746). Host only, no QNN run.

import pytest
import torch
from executorch.backends.qualcomm.quantizer.observers.per_block_param_observer import (
    PerBlockParamObserver,
)
from executorch.backends.qualcomm.quantizer.quant_recipe import (
    QuantGranularity,
    QuantRecipe,
)
from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer, QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import (
    _soc_info_table,
    HtpArch,
    QcomChipset,
)
from executorch.backends.qualcomm.tests.rework.conftest import calibrate
from executorch.backends.qualcomm.utils.constants import QCOM_BLOCK_SIZE
from torchao.quantization.pt2e import MinMaxObserver
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e


class TwoLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(512, 32)
        self.b = torch.nn.Linear(32, 16)

    def forward(self, x):
        return self.b(self.a(x))


class OneLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(512, 32)

    def forward(self, x):
        return self.a(x)


class ConvConvLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_a = torch.nn.Conv2d(32, 64, 1)
        self.conv_b = torch.nn.Conv2d(64, 64, 1)
        self.linear = torch.nn.Linear(64, 16)

    def forward(self, x):
        x = self.conv_b(torch.relu(self.conv_a(x)))
        return self.linear(x.mean(dim=(2, 3)))


def _weight_dq(gm):
    """Map each linear/conv node name to the target of its weight dequantize."""
    return {
        n.name: str(n.args[1].target)
        for n in gm.graph.nodes
        if n.op == "call_function"
        and n.target in (torch.ops.aten.linear.default, torch.ops.aten.conv2d.default)
    }


def _observed_block_sizes(prepared):
    return sorted(
        tuple(m.block_size)
        for m in prepared.modules()
        if isinstance(m, PerBlockParamObserver)
    )


def _padded(block_sizes):
    """V69+ lowers a per-block linear as a 1x1 conv2d, padding its block size to 4D."""
    return sorted(tuple(size) + (1,) * (4 - len(size)) for size in block_sizes)


def _block_quantizer(block_size_map, soc_model=QcomChipset.SM8850, strict=True):
    quantizer = QnnQuantizer(soc_model=soc_model, strict=strict)
    quantizer.set_default_quant_config(QuantDtype.use_16a4w_block)
    quantizer.set_block_size_map(block_size_map)
    return quantizer


@pytest.mark.parametrize("shape", [(3, 512), (3, 3, 512), (3, 3, 3, 512)])
def test_block_size_map_routes_linear_to_conv(shape):
    user_map = {"linear": (1, 32)}
    quantizer = _block_quantizer(user_map)
    with calibrate(TwoLinear().eval(), [(torch.randn(shape),)], quantizer) as gm:
        weight_dq = _weight_dq(gm)
    assert set(weight_dq) == {"conv2d_default", "linear_1"}, weight_dq
    assert "dequantize_affine" in weight_dq["conv2d_default"]
    assert "dequantize_affine" not in weight_dq["linear_1"]
    assert quantizer.block_size_map == {"linear": (1, 32)}


def test_rank5_linear_is_not_converted():
    quantizer = _block_quantizer({"linear": (1, 32)})
    inputs = (torch.randn(2, 3, 3, 3, 512),)
    with calibrate(TwoLinear().eval(), [inputs], quantizer) as gm:
        weight_dq = _weight_dq(gm)
    assert set(weight_dq) == {"linear", "linear_1"}, weight_dq
    assert "dequantize_affine" in weight_dq["linear"]


def test_v68_linear_is_not_converted():
    v68_soc = next(
        soc
        for soc, info in _soc_info_table.items()
        if info.htp_info.htp_arch == HtpArch.V68
    )
    # V68 rejects per-block FullyConnected in validation; only routing is checked.
    quantizer = _block_quantizer({"linear": (1, 32)}, soc_model=v68_soc, strict=False)
    with calibrate(TwoLinear().eval(), [(torch.randn(3, 512),)], quantizer) as gm:
        weight_dq = _weight_dq(gm)
    assert set(weight_dq) == {"linear", "linear_1"}, weight_dq
    assert "dequantize_affine" in weight_dq["linear"]


def test_converted_block_sizes_reset_on_reuse():
    quantizer = _block_quantizer({"linear": (1, 32)})
    with calibrate(TwoLinear().eval(), [(torch.randn(3, 512),)], quantizer):
        pass
    assert "conv2d_default" in quantizer._converted_block_sizes

    # The second model's linear is converted by the global flag and gets the
    # same conv name, so a stale entry would give it per-block quantization.
    quantizer.set_block_size_map({})
    quantizer.set_convert_linear_to_conv2d(True)
    with calibrate(OneLinear().eval(), [(torch.randn(3, 512),)], quantizer) as gm:
        weight_dq = _weight_dq(gm)
    assert set(weight_dq) == {"conv2d_default"}, weight_dq
    assert "dequantize_affine" not in weight_dq["conv2d_default"]
    assert quantizer._converted_block_sizes == {}


def test_recipe_routes_per_block_linear_to_conv():
    recipe = QuantRecipe(
        QuantDtype.use_16a4w,
        False,
        act_observer=MinMaxObserver,
        granularity=QuantGranularity.PER_TENSOR,
    ).add_regex(
        {r"^a$"},
        QuantDtype.use_16a4w_block,
        False,
        act_observer=MinMaxObserver,
        granularity=QuantGranularity.PER_BLOCK,
        extra_kwargs={"block_size": (1, 32)},
    )
    quantizer = QnnQuantizer(soc_model=QcomChipset.SM8850)
    quantizer.set_recipe(recipe)
    with calibrate(TwoLinear().eval(), [(torch.randn(3, 512),)], quantizer) as gm:
        weight_dq = _weight_dq(gm)
    assert set(weight_dq) == {"conv2d_default", "linear_1"}, weight_dq
    assert "dequantize_affine" in weight_dq["conv2d_default"]
    assert "dequantize_affine" not in weight_dq["linear_1"]


def test_block_size_is_per_node():
    # The annotators used to write each block size into the weight observer
    # constructor that every node shares, so the last one written won.
    sample_input = (torch.randn(1, 32, 8, 8),)
    gm = torch.export.export(ConvConvLinear().eval(), sample_input).module()
    names = [
        n.name
        for n in gm.graph.nodes
        if n.target in (torch.ops.aten.conv2d.default, torch.ops.aten.linear.default)
    ]
    block_size_map = dict(zip(names, [(1, 16, 1, 1), (1, 32, 1, 1), (1, 8)]))
    quantizer = _block_quantizer(block_size_map)
    configs = quantizer.default_quant_config.per_block_quant_config_list
    before = [
        (c.block_size, dict(c.weight.observer_or_fake_quant_ctr.p.keywords))
        for c in configs
    ]

    prepared = prepare_pt2e(gm, quantizer)
    prepared(*sample_input)

    assert _observed_block_sizes(prepared) == _padded(block_size_map.values())
    after = [
        (c.block_size, dict(c.weight.observer_or_fake_quant_ctr.p.keywords))
        for c in configs
    ]
    assert after == before
    assert all(QCOM_BLOCK_SIZE not in keywords for _, keywords in after)


def test_recipe_keeps_each_per_block_strategy_block_size():
    recipe = QuantRecipe(
        QuantDtype.use_16a4w,
        False,
        act_observer=MinMaxObserver,
        granularity=QuantGranularity.PER_TENSOR,
    )
    for regex, block_size in ((r"^a$", (1, 32)), (r"^b$", (1, 16))):
        recipe.add_regex(
            {regex},
            QuantDtype.use_16a4w_block,
            False,
            act_observer=MinMaxObserver,
            granularity=QuantGranularity.PER_BLOCK,
            extra_kwargs={"block_size": block_size},
        )
    quantizer = QnnQuantizer(soc_model=QcomChipset.SM8850)
    quantizer.set_recipe(recipe)
    sample_input = (torch.randn(3, 512),)
    gm = torch.export.export(TwoLinear().eval(), sample_input).module()
    prepared = prepare_pt2e(gm, quantizer)
    prepared(*sample_input)
    assert _observed_block_sizes(prepared) == _padded([(1, 32), (1, 16)])
