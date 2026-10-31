# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from executorch.backends.espressif.test.tester import (
    EspTestCase,
    EspTester,
    ramp_tensor,
)
from executorch.backends.test.suite.operators.test_mul import Model
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.exir.dialects._ops import ops as exir_ops

_MUL = exir_ops.edge.aten.mul.Tensor
_ESP_MUL = exir_ops.edge.esp_nn.quantized_mul_int8.default


class EspMul(Model):
    ops_before_transforms = {_MUL: 1}
    ops_after_transforms = {_MUL: 0, _ESP_MUL: 1}


class EspSelfMul(EspMul):
    def forward(self, x):
        return x * x


class EspConstantMul(EspMul):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.linspace(3, -2, 17))

    def forward(self, x):
        return x * self.weight


class EspMulChain(EspMul):
    ops_before_transforms = {_MUL: 2}
    ops_after_transforms = {_MUL: 0, _ESP_MUL: 2}

    def forward(self, x, y):
        product = x * y
        return product * x, torch.sin(product), torch.sin(x)


class EspMulWithFloatUsers(EspMul):
    def forward(self, x, y):
        product = x * y
        return product, torch.sin(x), torch.sin(product)


class EspBroadcastMul(EspMulWithFloatUsers):
    ops_after_transforms = {_MUL: 1, _ESP_MUL: 0}


class EspMulAdd(EspMul):
    ops_after_transforms = {
        _MUL: 0,
        _ESP_MUL: 1,
        exir_ops.edge.aten.add.Tensor: 1,
    }

    def forward(self, x, y):
        return x * y + y


test_cases = {
    "scalar": EspTestCase(
        EspMul(),
        (ramp_tensor(-1.5, 2.5, ()), ramp_tensor(3, -4, ())),
    ),
    "tensor_rank_1": EspTestCase(
        EspMul(),
        (ramp_tensor(-1.5, 2.5, (17,)), ramp_tensor(3, -4, (17,))),
    ),
    "tensor_rank_3": EspTestCase(
        EspMul(),
        (ramp_tensor(-1.5, 2.5, (2, 3, 5)), ramp_tensor(3, -4, (2, 3, 5))),
    ),
    "tensor_rank_4": EspTestCase(
        EspMul(),
        (ramp_tensor(-1.5, 2.5, (1, 2, 3, 4)), ramp_tensor(3, -4, (1, 2, 3, 4))),
    ),
    "self": EspTestCase(EspSelfMul(), (ramp_tensor(-1, 2, (17,)),)),
    "constant": EspTestCase(EspConstantMul(), (ramp_tensor(-1, 2, (17,)),)),
    "chain": EspTestCase(
        EspMulChain(),
        (ramp_tensor(-1, 1, (17,)), ramp_tensor(2, -2, (17,))),
    ),
    "float_users": EspTestCase(
        EspMulWithFloatUsers(),
        (ramp_tensor(-1, 1, (2, 3)), ramp_tensor(-2, 2, (2, 3))),
    ),
    "broadcast": EspTestCase(
        EspBroadcastMul(),
        (ramp_tensor(-1, 1, (2, 3)), ramp_tensor(-2, 2, (3,))),
    ),
}


@pytest.mark.parametrize("test_case", test_cases.values(), ids=test_cases.keys())
def test_dialect_mul(test_case):
    tester = EspTester(test_case.model, test_case.example_inputs)
    tester.test_dialect(
        test_case.model.ops_before_transforms,
        test_case.model.ops_after_transforms,
        qtol=1,
        atol=0,
        rtol=torch.finfo(torch.float32).eps,
    )
    tester.to_executorch().serialize()
    program = deserialize_pte_binary(tester.get_artifact()).program
    assert not program.execution_plan[0].delegates
    operators = {(op.name, op.overload) for op in program.execution_plan[0].operators}
    assert (("esp_nn::quantized_mul_int8", "out") in operators) == (
        test_case.model.ops_after_transforms[_ESP_MUL] > 0
    )


def test_mul_keeps_dequantization_for_portable_add():
    model = EspMulAdd()
    inputs = (ramp_tensor(-1, 1, (17,)), ramp_tensor(2, -2, (17,)))
    tester = EspTester(model, inputs)
    tester.test_dialect(
        model.ops_before_transforms, model.ops_after_transforms, qtol=1, atol=0, rtol=0
    )
    add = next(
        node
        for node in tester.get_artifact().exported_program().graph.nodes
        if node.target == exir_ops.edge.aten.add.Tensor
    )
    assert all(
        arg.target == exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
        for arg in add.args[:2]
    )
    tester.to_executorch().serialize()
