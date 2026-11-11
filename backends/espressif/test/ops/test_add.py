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
from executorch.backends.test.suite.operators.test_add import Model, ModelAlpha
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.exir.dialects._ops import ops as exir_ops

_ADD = exir_ops.edge.aten.add.Tensor
_ESP_ADD = exir_ops.edge.esp_nn.quantized_add_int8.default


class EspAdd(Model):
    ops_before_transforms = {_ADD: 1}
    ops_after_transforms = {_ADD: 0, _ESP_ADD: 1}


class EspSelfAdd(EspAdd):
    def forward(self, x):
        return x + x


class EspConstantAdd(EspAdd):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.linspace(3, -2, 17))

    def forward(self, x):
        return x + self.weight


class EspAddChain(EspAdd):
    ops_before_transforms = {_ADD: 2}
    ops_after_transforms = {_ADD: 0, _ESP_ADD: 2}

    def forward(self, x, y):
        total = x + y
        return total + x, torch.sin(total), torch.sin(x)


class EspAddWithFloatUsers(EspAdd):
    def forward(self, x, y):
        total = x + y
        return total, torch.sin(x), torch.sin(total)


class EspBroadcastAdd(EspAddWithFloatUsers):
    ops_after_transforms = {_ADD: 1, _ESP_ADD: 0}


class EspAlphaAdd(ModelAlpha):
    ops_before_transforms = {_ADD: 1}
    ops_after_transforms = {_ADD: 1, _ESP_ADD: 0}


class EspAddRelu(EspAdd):
    ops_after_transforms = {_ADD: 1, _ESP_ADD: 0}

    def forward(self, x, y):
        return torch.relu(x + y)


test_cases = {
    "scalar": EspTestCase(
        EspAdd(), (ramp_tensor(-1.5, 2.5, ()), ramp_tensor(3, -4, ()))
    ),
    "tensor_rank_1": EspTestCase(
        EspAdd(), (ramp_tensor(-1.5, 2.5, (17,)), ramp_tensor(3, -4, (17,)))
    ),
    "tensor_rank_4": EspTestCase(
        EspAdd(),
        (ramp_tensor(-1.5, 2.5, (1, 2, 3, 4)), ramp_tensor(3, -4, (1, 2, 3, 4))),
    ),
    "unequal_scales": EspTestCase(
        EspAdd(), (ramp_tensor(-0.01, 0.02, (17,)), ramp_tensor(30, -40, (17,)))
    ),
    "self": EspTestCase(EspSelfAdd(), (ramp_tensor(-1, 2, (17,)),)),
    "constant": EspTestCase(EspConstantAdd(), (ramp_tensor(-1, 2, (17,)),)),
    "chain": EspTestCase(
        EspAddChain(), (ramp_tensor(-1, 1, (17,)), ramp_tensor(2, -2, (17,)))
    ),
    "float_users": EspTestCase(
        EspAddWithFloatUsers(),
        (ramp_tensor(-1, 1, (2, 3)), ramp_tensor(-2, 2, (2, 3))),
    ),
    "broadcast": EspTestCase(
        EspBroadcastAdd(), (ramp_tensor(-1, 1, (2, 3)), ramp_tensor(-2, 2, (3,)))
    ),
    "alpha": EspTestCase(
        EspAlphaAdd(alpha=2),
        (ramp_tensor(-1, 1, (17,)), ramp_tensor(2, -2, (17,))),
    ),
    "relu": EspTestCase(
        EspAddRelu(), (ramp_tensor(-1, 1, (17,)), ramp_tensor(2, -2, (17,)))
    ),
    "saturation": EspTestCase(
        EspAdd(),
        (torch.tensor([-2.0, -2.0, 2.0, 2.0]), torch.tensor([-2.0, 2.0, -2.0, 2.0])),
        calibration_samples=[
            (torch.tensor([-2.0, 2.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, -2.0, 2.0]))
        ],
    ),
}


@pytest.mark.parametrize("test_case", test_cases.values(), ids=test_cases.keys())
def test_dialect_add(test_case):
    tester = EspTester(
        test_case.model,
        test_case.example_inputs,
        calibration_samples=test_case.calibration_samples,
    )
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
    assert (("esp_nn::quantized_add_int8", "out") in operators) == (
        test_case.model.ops_after_transforms[_ESP_ADD] > 0
    )


@pytest.mark.parametrize("test_case", test_cases.values(), ids=test_cases.keys())
def test_implementation_add(test_case, pytestconfig):
    tester = EspTester(
        test_case.model,
        test_case.example_inputs,
        calibration_samples=test_case.calibration_samples,
    )
    tester.test_implementation(
        pytestconfig.rootpath, rtol=torch.finfo(torch.float32).eps
    )
