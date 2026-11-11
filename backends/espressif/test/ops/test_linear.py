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
from executorch.backends.test.suite.operators.test_linear import Model
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.exir.dialects._ops import ops as exir_ops

_ADDMM = exir_ops.edge.aten.addmm.default
_MM = exir_ops.edge.aten.mm.default
_ESP_LINEAR = exir_ops.edge.esp_nn.quantized_linear_int8.default
_RELU6 = exir_ops.edge.aten.hardtanh.default


class EspLinear(Model):
    def __init__(self, in_features=17, out_features=5, bias=True, activation=None):
        super().__init__(in_features, out_features, bias)
        self.activation = activation or torch.nn.Identity()
        with torch.no_grad():
            values = torch.arange(in_features * out_features).reshape(
                out_features, in_features
            )
            self.linear.weight.copy_(
                values.sin() * torch.arange(1, out_features + 1).reshape(-1, 1) * 0.02
            )
            if bias:
                self.linear.bias.copy_(torch.linspace(-0.5, 0.7, out_features))
        self.ops_before_transforms = {_ADDMM if bias else _MM: 1}
        self.ops_after_transforms = {_ADDMM: 0, _MM: 0, _ESP_LINEAR: 1}
        if activation is not None:
            target = (
                exir_ops.edge.aten.relu.default
                if isinstance(activation, torch.nn.ReLU)
                else _RELU6
            )
            self.ops_before_transforms[target] = 1
            self.ops_after_transforms[target] = 0

    def forward(self, x):
        return self.activation(self.linear(x))


class EspLinearWithFloatUsers(EspLinear):
    def forward(self, x):
        result = self.linear(x)
        return result, torch.sin(result), torch.sin(x)


class EspLinearWithSharedQParams(EspLinear):
    def forward(self, x):
        return torch.cat((super().forward(x), x), dim=-1)


class EspLinearActivationWithUsers(EspLinear):
    def __init__(self):
        super().__init__(activation=torch.nn.ReLU6())
        self.ops_after_transforms[_RELU6] = 1

    def forward(self, x):
        result = self.linear(x)
        return result, self.activation(result)


class EspLinearChain(EspLinear):
    def __init__(self):
        super().__init__(17, 17)
        self.ops_before_transforms = {_ADDMM: 2}
        self.ops_after_transforms[_ESP_LINEAR] = 2

    def forward(self, x):
        return self.linear(self.linear(x))


class EspLinearWithCoarseBias(EspLinear):
    def __init__(self):
        super().__init__(in_features=2, out_features=1)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[1.0, 0.0]]))
            self.linear.bias.fill_(0.1)


class EspUnsupportedLinearRank(EspLinear):
    def __init__(self):
        super().__init__()
        self.ops_after_transforms = {_ADDMM: 1, _ESP_LINEAR: 0}


test_cases = {
    "bias": EspTestCase(EspLinear(), (ramp_tensor(-3, 5, (2, 17)),)),
    "no_bias": EspTestCase(EspLinear(bias=False), (ramp_tensor(-3, 5, (2, 17)),)),
    "single_row": EspTestCase(EspLinear(), (ramp_tensor(-3, 5, (1, 17)),)),
    "single_channel": EspTestCase(
        EspLinear(out_features=1), (ramp_tensor(-3, 5, (3, 17)),)
    ),
    "single_feature": EspTestCase(
        EspLinear(in_features=1), (ramp_tensor(-3, 5, (4, 1)),)
    ),
    "aligned": EspTestCase(EspLinear(in_features=256), (ramp_tensor(-3, 5, (3, 256)),)),
    "tail": EspTestCase(EspLinear(in_features=273), (ramp_tensor(-3, 5, (3, 273)),)),
    "relu": EspTestCase(
        EspLinear(activation=torch.nn.ReLU()), (ramp_tensor(-3, 5, (2, 17)),)
    ),
    "relu6": EspTestCase(
        EspLinear(activation=torch.nn.ReLU6()), (ramp_tensor(-30, 70, (2, 17)),)
    ),
    "relu6_shared_qparams": EspTestCase(
        EspLinearWithSharedQParams(activation=torch.nn.ReLU6()),
        (ramp_tensor(-40, 80, (2, 17)),),
    ),
    "relu6_with_users": EspTestCase(
        EspLinearActivationWithUsers(), (ramp_tensor(-30, 70, (2, 17)),)
    ),
    "float_users": EspTestCase(
        EspLinearWithFloatUsers(), (ramp_tensor(-3, 5, (2, 17)),)
    ),
    "chain": EspTestCase(EspLinearChain(), (ramp_tensor(-3, 5, (2, 17)),)),
    "bias_rounding": EspTestCase(
        EspLinearWithCoarseBias(),
        (torch.tensor([[0.0, 100000.0], [0.0, -100000.0]]),),
    ),
    "vector_fallback": EspTestCase(
        EspUnsupportedLinearRank(), (ramp_tensor(-3, 5, (17,)),)
    ),
    "rank_3_fallback": EspTestCase(
        EspUnsupportedLinearRank(), (ramp_tensor(-3, 5, (2, 3, 17)),)
    ),
}


@pytest.mark.parametrize("test_case", test_cases.values(), ids=test_cases.keys())
def test_dialect_linear(test_case):
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
    assert (("esp_nn::quantized_linear_int8", "out") in operators) == (
        test_case.model.ops_after_transforms[_ESP_LINEAR] > 0
    )


@pytest.mark.parametrize("test_case", test_cases.values(), ids=test_cases.keys())
def test_implementation_linear(test_case, pytestconfig):
    tester = EspTester(test_case.model, test_case.example_inputs)
    tester.test_implementation(
        pytestconfig.rootpath, rtol=torch.finfo(torch.float32).eps
    )
