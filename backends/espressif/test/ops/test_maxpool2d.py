# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import pytest
import torch
from executorch.backends.espressif.test.tester import EspTestCase, EspTester
from executorch.backends.test.suite.operators.test_maxpool2d import Model
from executorch.backends.transforms.fused_quant.graph_utils import (
    get_qparams_from_node,
    get_scale,
)
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
)
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.frontends.fused_quant.quantizer.quantizer import (
    FusedQuantQuantizer,
    OpQuantizer,
)

_POOL = exir_ops.edge.aten.max_pool2d_with_indices.default
_FQ_POOL = exir_ops.edge.fused_quant.max_pool2d_with_indices.default
_ESP_POOL = exir_ops.edge.esp_nn.quantized_max_pool2d_int8.default
_PERMUTE = exir_ops.edge.aten.permute_copy.default


class EspMaxPool2d(Model):
    def __init__(self, *, supported=True, **kwargs):
        super().__init__(**kwargs)
        self.ops_before_transforms = {_FQ_POOL: 1}
        self.ops_after_transforms = {
            _POOL: 0 if supported else 1,
            _ESP_POOL: 1 if supported else 0,
            _FQ_POOL: 0,
        }


class EspPoolChain(EspMaxPool2d):
    def __init__(self):
        super().__init__(kernel_size=2)
        self.ops_before_transforms[_FQ_POOL] = 2
        self.ops_after_transforms[_ESP_POOL] = 2

    def forward(self, x):
        return self.maxpool(self.maxpool(x))


class EspPoolWithFloatUsers(EspMaxPool2d):
    def forward(self, x):
        pooled = self.maxpool(x)
        return pooled, torch.sin(pooled), torch.sin(x)


class EspPoolDefaultStride(EspMaxPool2d):
    def forward(self, x):
        return torch.ops.aten.max_pool2d_with_indices.default(x, [2, 2])[0]


class EspPoolWithFallback(EspMaxPool2d):
    def __init__(self):
        super().__init__(kernel_size=2)
        self.portable_pool = torch.nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True)
        self.ops_before_transforms[_FQ_POOL] = 2
        self.ops_after_transforms[_POOL] = 1

    def forward(self, x):
        return self.maxpool(x), self.portable_pool(x)


class EspPoolAddChain(EspPoolChain):
    def __init__(self):
        super().__init__()
        self.ops_after_transforms[exir_ops.edge.esp_nn.quantized_add_int8.default] = 1

    def forward(self, x):
        pooled = self.maxpool(x)
        return self.maxpool(pooled + pooled)


class EspPoolAndAdd(EspMaxPool2d):
    def __init__(self):
        super().__init__(kernel_size=2)
        self.ops_after_transforms[exir_ops.edge.esp_nn.quantized_add_int8.default] = 1

    def forward(self, x):
        return self.maxpool(x + x)


class EspPoolWithBuffer(EspPoolChain):
    def __init__(self):
        super().__init__()
        self.register_buffer("offset", torch.linspace(-1, 2, 24).reshape(1, 4, 2, 3))

    def forward(self, x):
        return self.maxpool(self.maxpool(x) + self.offset)


def _input(shape, *, negative=False):
    values = torch.arange(math.prod(shape)).reshape(shape)
    return (values * 0.7).sin() * 3 - (4 if negative else 0)


test_cases = {
    "channels_4": EspTestCase(EspMaxPool2d(kernel_size=2), (_input((1, 4, 5, 7)),)),
    "channels_8": EspTestCase(EspMaxPool2d(), (_input((1, 8, 9, 11)),)),
    "channels_12": EspTestCase(EspMaxPool2d(), (_input((1, 12, 9, 11)),)),
    "padding_limit_channels_16": EspTestCase(
        EspMaxPool2d(kernel_size=2, padding=1), (_input((1, 16, 5, 7)),)
    ),
    "stride_one_padding_channels_20": EspTestCase(
        EspMaxPool2d(kernel_size=5, stride=1, padding=2), (_input((1, 20, 7, 9)),)
    ),
    "default_stride": EspTestCase(EspPoolDefaultStride(), (_input((1, 4, 5, 7)),)),
    "batches": EspTestCase(EspMaxPool2d(), (_input((3, 4, 7, 9)),)),
    "rectangular": EspTestCase(
        EspMaxPool2d(kernel_size=(3, 2), stride=(2, 1), padding=(1, 0)),
        (_input((1, 8, 7, 9)),),
    ),
    "negative_padding": EspTestCase(
        EspMaxPool2d(kernel_size=3, stride=2, padding=1),
        (_input((1, 4, 7, 9), negative=True),),
    ),
    "identity": EspTestCase(EspMaxPool2d(kernel_size=1), (_input((1, 4, 3, 5)),)),
    "global": EspTestCase(EspMaxPool2d(kernel_size=(5, 7)), (_input((1, 4, 5, 7)),)),
    "chain": EspTestCase(EspPoolChain(), (_input((1, 4, 9, 11)),)),
    "add_between_pools": EspTestCase(EspPoolAddChain(), (_input((1, 4, 9, 11)),)),
    "add_producer": EspTestCase(EspPoolAndAdd(), (_input((1, 4, 7, 9)),)),
    "constant_between_pools": EspTestCase(
        EspPoolWithBuffer(), (torch.linspace(-2, 4, 96).reshape(1, 4, 4, 6),)
    ),
    "float_users": EspTestCase(EspPoolWithFloatUsers(), (_input((1, 4, 7, 9)),)),
    "shared_input_with_fallback": EspTestCase(
        EspPoolWithFallback(), (_input((1, 4, 7, 9)),)
    ),
    "channel_tail_fallback": EspTestCase(
        EspMaxPool2d(supported=False), (_input((1, 5, 7, 9)),)
    ),
    "ceil_fallback": EspTestCase(
        EspMaxPool2d(supported=False, ceil_mode=True), (_input((1, 4, 7, 9)),)
    ),
    "dilation_fallback": EspTestCase(
        EspMaxPool2d(supported=False, dilation=2), (_input((1, 4, 7, 9)),)
    ),
    "indices_fallback": EspTestCase(
        EspMaxPool2d(supported=False, return_indices=True), (_input((1, 4, 7, 9)),)
    ),
    "unbatched": EspTestCase(EspMaxPool2d(), (_input((4, 7, 9)),)),
}


@pytest.mark.parametrize("test_case", test_cases.values(), ids=test_cases.keys())
def test_dialect_maxpool2d(test_case):
    tester = EspTester(test_case.model, test_case.example_inputs)
    tester.test_dialect(
        test_case.model.ops_before_transforms,
        test_case.model.ops_after_transforms,
        atol=0,
        rtol=0,
    )
    program = tester.get_artifact().exported_program()
    for node in program.graph.nodes:
        if node.target == _ESP_POOL:
            assert node.args[0].meta["val"].dtype == torch.int8
            assert node.args[0].meta["val"].is_contiguous()
            assert node.meta["val"].is_contiguous()
        if node.target == _PERMUTE:
            assert node.meta["val"].dtype in (torch.int8, torch.int64)
    tester.to_executorch().serialize()
    serialized = deserialize_pte_binary(tester.get_artifact()).program
    operators = {
        (op.name, op.overload) for op in serialized.execution_plan[0].operators
    }
    assert not serialized.execution_plan[0].delegates
    assert not any(name.startswith("fused_quant::") for name, _ in operators)
    assert (("esp_nn::quantized_max_pool2d_int8", "out") in operators) == (
        test_case.model.ops_after_transforms[_ESP_POOL] > 0
    )


@pytest.mark.parametrize("test_case", test_cases.values(), ids=test_cases.keys())
def test_implementation_maxpool2d(test_case, pytestconfig):
    EspTester(test_case.model, test_case.example_inputs).test_implementation(
        pytestconfig.rootpath
    )


def test_different_output_qparams_stay_portable():
    inputs = (torch.linspace(-3, 5, 140).reshape(1, 4, 5, 7),)
    quantizer = FusedQuantQuantizer(
        [
            OpQuantizer(
                torch.ops.aten.max_pool2d_with_indices.default,
                torch.ops.fused_quant.max_pool2d_with_indices.default,
                get_symmetric_quantization_config(),
            )
        ]
    )
    tester = (
        EspTester(torch.nn.MaxPool2d(2), inputs).export_fused_quant(quantizer).to_edge()
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
