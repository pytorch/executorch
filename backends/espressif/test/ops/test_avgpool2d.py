# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import pytest
import torch
from executorch.backends.espressif.test.tester import EspTestCase, EspTester
from executorch.backends.test.suite.operators.test_avgpool2d import Model
from executorch.backends.transforms.fused_quant.graph_utils import (
    get_qparams_from_node,
    get_scale,
)
from executorch.backends.transforms.utils import get_constant
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
)
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.frontends.fused_quant.quantizer.quantizer import (
    FusedQuantQuantizer,
    OpQuantizer,
)

_POOL = exir_ops.edge.aten.avg_pool2d.default
_FQ_POOL = exir_ops.edge.fused_quant.avg_pool2d.default
_ESP_POOL = exir_ops.edge.esp_nn.quantized_avg_pool2d_int8.default
_ESP_ADD = exir_ops.edge.esp_nn.quantized_add_int8.default
_PERMUTE = exir_ops.edge.aten.permute_copy.default


class EspAvgPool2d(Model):
    def __init__(self, *, supported=True, divisor_override=None, **kwargs):
        super().__init__(**kwargs)
        self.avgpool.divisor_override = divisor_override
        self.ops_before_transforms = {_FQ_POOL: 1}
        self.ops_after_transforms = {
            _POOL: 0 if supported else 1,
            _ESP_POOL: 1 if supported else 0,
            _FQ_POOL: 0,
        }


class EspAvgPoolChain(EspAvgPool2d):
    def __init__(self):
        super().__init__(kernel_size=2)
        self.ops_before_transforms[_FQ_POOL] = 2
        self.ops_after_transforms[_ESP_POOL] = 2

    def forward(self, x):
        return self.avgpool(self.avgpool(x))


class EspAvgPoolWithFloatUsers(EspAvgPool2d):
    def forward(self, x):
        pooled = self.avgpool(x)
        return pooled, torch.sin(pooled), torch.sin(x)


class EspAvgPoolAddChain(EspAvgPoolChain):
    def __init__(self):
        super().__init__()
        self.ops_after_transforms[_ESP_ADD] = 1

    def forward(self, x):
        pooled = self.avgpool(x)
        return self.avgpool(pooled + pooled)


class EspAvgPoolAndAdd(EspAvgPool2d):
    def __init__(self):
        super().__init__(kernel_size=2)
        self.ops_after_transforms[_ESP_ADD] = 1

    def forward(self, x):
        return self.avgpool(x + x)


class EspAvgPoolWithBuffer(EspAvgPoolChain):
    def __init__(self):
        super().__init__()
        self.register_buffer("offset", torch.linspace(-1, 2, 16).reshape(1, 4, 2, 2))
        self.ops_after_transforms[_ESP_ADD] = 1

    def forward(self, x):
        return self.avgpool(self.avgpool(x) + self.offset)


def _input(shape, *, negative=False):
    values = torch.arange(math.prod(shape)).reshape(shape)
    return (values * 0.7).sin() * 3 - (4 if negative else 0)


test_cases = {
    "channels_4": EspTestCase(EspAvgPool2d(kernel_size=2), (_input((1, 4, 5, 5)),)),
    "channels_8": EspTestCase(EspAvgPool2d(), (_input((1, 8, 9, 11)),)),
    "channels_12": EspTestCase(EspAvgPool2d(), (_input((1, 12, 9, 11)),)),
    "batches": EspTestCase(EspAvgPool2d(), (_input((3, 4, 7, 8)),)),
    "rectangular": EspTestCase(
        EspAvgPool2d(
            kernel_size=(3, 2), stride=(2, 1), padding=(1, 0), count_include_pad=False
        ),
        (_input((1, 8, 7, 5)),),
    ),
    "negative_padding": EspTestCase(
        EspAvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False),
        (_input((1, 4, 7, 7), negative=True),),
    ),
    "identity": EspTestCase(EspAvgPool2d(kernel_size=1), (_input((1, 4, 3, 3)),)),
    "global": EspTestCase(EspAvgPool2d(kernel_size=(5, 7)), (_input((1, 4, 5, 7)),)),
    "single_row": EspTestCase(EspAvgPool2d(), (_input((1, 8, 3, 9)),)),
    "large_window": EspTestCase(
        EspAvgPool2d(kernel_size=(32, 16)), (torch.full((1, 4, 32, 16), -7.0),)
    ),
    "signed_rounding": EspTestCase(
        EspAvgPool2d(kernel_size=(1, 2)),
        (
            torch.tensor([0.0, 1.0, -1.0, 0.0, 2.0, 1.0, -2.0, -1.0])
            .repeat(4)
            .reshape(1, 4, 1, 8),
        ),
    ),
    "chain": EspTestCase(EspAvgPoolChain(), (_input((1, 4, 9, 9)),)),
    "add_between_pools": EspTestCase(EspAvgPoolAddChain(), (_input((1, 4, 9, 9)),)),
    "add_producer": EspTestCase(EspAvgPoolAndAdd(), (_input((1, 4, 7, 7)),)),
    "constant_between_pools": EspTestCase(
        EspAvgPoolWithBuffer(), (torch.linspace(-2, 4, 64).reshape(1, 4, 4, 4),)
    ),
    "float_users": EspTestCase(EspAvgPoolWithFloatUsers(), (_input((1, 4, 7, 8)),)),
    "wide_output_channels_4_fallback": EspTestCase(
        EspAvgPool2d(kernel_size=2, supported=False), (_input((1, 4, 5, 7)),)
    ),
    "tall_output_channels_4_fallback": EspTestCase(
        EspAvgPool2d(kernel_size=2, supported=False), (_input((1, 4, 7, 5)),)
    ),
    "wide_output_channels_8_fallback": EspTestCase(
        EspAvgPool2d(kernel_size=2, supported=False), (_input((1, 8, 5, 7)),)
    ),
    "tall_output_channels_8_fallback": EspTestCase(
        EspAvgPool2d(kernel_size=2, supported=False), (_input((1, 8, 7, 5)),)
    ),
    "channel_tail_fallback": EspTestCase(
        EspAvgPool2d(supported=False), (_input((1, 5, 7, 8)),)
    ),
    "ceil_fallback": EspTestCase(
        EspAvgPool2d(supported=False, ceil_mode=True), (_input((1, 4, 7, 7)),)
    ),
    "padding_divisor_fallback": EspTestCase(
        EspAvgPool2d(supported=False, padding=1), (_input((1, 4, 7, 7)),)
    ),
    "divisor_override_fallback": EspTestCase(
        EspAvgPool2d(supported=False, divisor_override=2), (_input((1, 4, 7, 8)),)
    ),
    "unbatched": EspTestCase(
        EspAvgPool2d(supported=False), (_input((4, 7, 9)),)
    ),
}


@pytest.mark.parametrize("test_case", test_cases.values(), ids=test_cases.keys())
def test_dialect_avgpool2d(test_case):
    tester = EspTester(test_case.model, test_case.example_inputs)
    tester.test_dialect(
        test_case.model.ops_before_transforms,
        test_case.model.ops_after_transforms,
        qtol=test_case.model.ops_after_transforms[_ESP_POOL],
        atol=1e-6 if test_case.model.ops_after_transforms[_ESP_POOL] else 0,
        rtol=0,
    )
    program = tester.get_artifact().exported_program()
    for node in program.graph.nodes:
        if node.target == _ESP_POOL:
            assert node.args[0].meta["val"].dtype == torch.int8
            assert node.args[0].meta["val"].is_contiguous()
            assert node.meta["val"].is_contiguous()
        if node.target == _PERMUTE:
            assert node.meta["val"].dtype == torch.int8
    tester.to_executorch().serialize()
    serialized = deserialize_pte_binary(tester.get_artifact()).program
    operators = {
        (op.name, op.overload) for op in serialized.execution_plan[0].operators
    }
    assert not serialized.execution_plan[0].delegates
    assert not any(name.startswith("fused_quant::") for name, _ in operators)
    assert (("esp_nn::quantized_avg_pool2d_int8", "out") in operators) == (
        test_case.model.ops_after_transforms[_ESP_POOL] > 0
    )


@pytest.mark.parametrize("test_case", test_cases.values(), ids=test_cases.keys())
def test_implementation_avgpool2d(test_case, pytestconfig):
    EspTester(test_case.model, test_case.example_inputs).test_implementation(
        pytestconfig.rootpath,
        rtol=(
            torch.finfo(torch.float32).eps
            if isinstance(test_case.model, EspAvgPoolWithFloatUsers)
            else 0
        ),
    )


def test_avgpool_qparams_share_input_observer():
    inputs = (torch.linspace(-3, 5, 100).reshape(1, 4, 5, 5),)
    tester = EspTester(torch.nn.AvgPool2d(2), inputs).export_fused_quant().to_edge()
    program = tester.get_artifact().exported_program()
    pool = next(node for node in program.graph.nodes if node.target == _FQ_POOL)
    inp = get_qparams_from_node(pool, "inp")
    out = get_qparams_from_node(pool, "out")
    torch.testing.assert_close(
        get_constant(program, inp.scale),
        get_constant(program, out.scale),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        get_constant(program, inp.zero_point),
        get_constant(program, out.zero_point),
        atol=0,
        rtol=0,
    )


def test_divisor_override_keeps_independent_observers():
    inputs = (torch.linspace(0.25, 1.25, 196).reshape(1, 4, 7, 7),)
    tester = (
        EspTester(torch.nn.AvgPool2d(3, divisor_override=1), inputs)
        .export_fused_quant()
        .to_edge()
    )
    program = tester.get_artifact().exported_program()
    pool = next(node for node in program.graph.nodes if node.target == _FQ_POOL)
    assert get_scale(program, get_qparams_from_node(pool, "inp")) < get_scale(
        program, get_qparams_from_node(pool, "out")
    )
    expected = program.module()(*inputs)
    tester.run_passes().check_node_count({_FQ_POOL: 0, _POOL: 1, _ESP_POOL: 0})
    torch.testing.assert_close(
        tester.get_artifact().exported_program().module()(*inputs),
        expected,
        atol=0,
        rtol=0,
    )
    tester.to_executorch().serialize()


def test_different_output_qparams_stay_portable():
    inputs = (torch.linspace(-3, 5, 100).reshape(1, 4, 5, 5),)
    quantizer = FusedQuantQuantizer(
        [
            OpQuantizer(
                torch.ops.aten.avg_pool2d.default,
                torch.ops.fused_quant.avg_pool2d.default,
                get_symmetric_quantization_config(),
            )
        ]
    )
    tester = (
        EspTester(torch.nn.AvgPool2d(2), inputs).export_fused_quant(quantizer).to_edge()
    )
    reference = tester.get_artifact().exported_program()
    pool = next(node for node in reference.graph.nodes if node.target == _FQ_POOL)
    assert get_scale(reference, get_qparams_from_node(pool, "inp")) != get_scale(
        reference, get_qparams_from_node(pool, "out")
    )
    expected = reference.module()(*inputs)
    tester.run_passes().check_node_count({_POOL: 1, _ESP_POOL: 0})
    torch.testing.assert_close(
        tester.get_artifact().exported_program().module()(*inputs),
        expected,
        atol=0,
        rtol=0,
    )
    tester.to_executorch().serialize()


@pytest.mark.parametrize(
    "values, expected",
    [
        ([0, 1], 1),
        ([0, -1], -1),
        ([1, 2], 2),
        ([-1, -2], -2),
        ([0, 0, 1], 0),
        ([0, 0, -1], 0),
        ([0, 1, 1], 1),
        ([0, -1, -1], -1),
        ([-128, 127], -1),
        ([-128, -128], -128),
        ([127, 127], 127),
    ],
)
def test_integer_reference_rounding(values, expected):
    inputs = (
        torch.tensor(values, dtype=torch.int8)
        .reshape(1, 1, -1, 1)
        .expand(-1, -1, -1, 4)
        .contiguous()
    )
    actual = torch.ops.esp_nn.quantized_avg_pool2d_int8(
        inputs, [1, len(values)], [1, len(values)], [0, 0]
    )
    torch.testing.assert_close(
        actual, torch.full((1, 1, 1, 4), expected, dtype=torch.int8)
    )
