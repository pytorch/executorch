# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

import pytest
import torch
from executorch.backends.espressif.test.tester import (
    EspTestCase,
    EspTester,
    ramp_tensor,
)
from executorch.backends.test.suite.operators.test_mul import Model
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
    "portable_add": EspTestCase(
        EspMulAdd(),
        (ramp_tensor(-1, 1, (17,)), ramp_tensor(2, -2, (17,))),
    ),
    "saturation": EspTestCase(
        EspMul(),
        (torch.tensor([-2.0, -2.0, 2.0, 2.0]), torch.tensor([-2.0, 2.0, -2.0, 2.0])),
        calibration_samples=[
            (
                torch.tensor([-2.0, 2.0, 0.0, 0.0]),
                torch.tensor([0.25, 0.25, -2.0, 2.0]),
            )
        ],
    ),
}


@pytest.mark.parametrize("test_case", test_cases.values(), ids=test_cases.keys())
def test_dialect_mul(test_case):
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
    for node in tester.get_artifact().exported_program().graph.nodes:
        if node.target == exir_ops.edge.aten.add.Tensor:
            assert all(
                arg.target
                == exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
                for arg in node.args[:2]
            )
    tester.to_executorch().serialize()
    program = deserialize_pte_binary(tester.get_artifact()).program
    assert not program.execution_plan[0].delegates
    operators = {(op.name, op.overload) for op in program.execution_plan[0].operators}
    assert (("esp_nn::quantized_mul_int8", "out") in operators) == (
        test_case.model.ops_after_transforms[_ESP_MUL] > 0
    )


@pytest.mark.parametrize("test_case", test_cases.values(), ids=test_cases.keys())
def test_implementation_mul(test_case, pytestconfig):
    tester = EspTester(
        test_case.model,
        test_case.example_inputs,
        calibration_samples=test_case.calibration_samples,
    )
    tester.test_implementation(
        pytestconfig.rootpath, rtol=torch.finfo(torch.float32).eps
    )


_FUSED_MUL = exir_ops.edge.fused_quant.mul.default


def _fused_quant_mul_tester(test_case):
    tester = EspTester(
        test_case.model,
        test_case.example_inputs,
        calibration_samples=test_case.calibration_samples,
    ).compile_fused_quant()
    tester.check_node_count(
        {_FUSED_MUL: test_case.model.ops_before_transforms[_MUL], _MUL: 0}
    )
    reference = tester.get_artifact().exported_program()
    graph_before = str(reference.graph)
    constants_before = reference.constants.copy()
    expected = reference.module()(*test_case.example_inputs)
    unfused = (
        EspTester(
            test_case.model,
            test_case.example_inputs,
            calibration_samples=test_case.calibration_samples,
        )
        .quantize()
        .get_artifact()
    )
    torch.testing.assert_close(
        expected, unfused(*test_case.example_inputs), atol=0, rtol=0
    )
    scale = max(
        get_scale(reference, get_qparams_from_node(node, "out"))
        for node in reference.graph.nodes
        if node.target == _FUSED_MUL
    )
    tester.run_fused_quant_passes().check_node_count(
        {_FUSED_MUL: 0, **test_case.model.ops_after_transforms}
    )
    lowered = tester.get_artifact().exported_program()
    lowered.validate()
    actual = lowered.module()(*test_case.example_inputs)
    accelerated = test_case.model.ops_after_transforms[_ESP_MUL] > 0
    torch.testing.assert_close(
        actual,
        expected,
        atol=scale if accelerated else 0,
        rtol=torch.finfo(torch.float32).eps if accelerated else 0,
    )
    if isinstance(test_case.model, EspMulWithFloatUsers):
        torch.testing.assert_close(
            actual[1], torch.sin(test_case.example_inputs[0]), atol=0, rtol=0
        )
    assert str(reference.graph) == graph_before
    assert reference.constants.keys() == constants_before.keys()
    assert all(
        reference.constants[key] is value for key, value in constants_before.items()
    )
    reference.validate()
    torch.testing.assert_close(
        reference.module()(*test_case.example_inputs), expected, atol=0, rtol=0
    )
    return tester


@pytest.mark.parametrize("test_case", test_cases.values(), ids=test_cases.keys())
def test_dialect_fused_quant_mul(test_case):
    tester = _fused_quant_mul_tester(test_case).to_executorch().serialize()
    plan = deserialize_pte_binary(tester.get_artifact()).program.execution_plan[0]
    assert not plan.delegates
    assert not any(op.name.startswith("fused_quant::") for op in plan.operators)
    operators = {(op.name, op.overload) for op in plan.operators}
    assert (("esp_nn::quantized_mul_int8", "out") in operators) == (
        test_case.model.ops_after_transforms[_ESP_MUL] > 0
    )


@pytest.mark.parametrize("test_case", test_cases.values(), ids=test_cases.keys())
def test_implementation_fused_quant_mul(test_case, pytestconfig):
    _fused_quant_mul_tester(test_case).test_implementation(
        pytestconfig.rootpath, rtol=torch.finfo(torch.float32).eps
    )


@pytest.mark.parametrize("float_edge", ["input_activation", "output_activation"])
def test_dialect_partially_quantized_mul_falls_back(float_edge):
    config = replace(get_symmetric_quantization_config(), **{float_edge: None})
    quantizer = FusedQuantQuantizer(
        [
            OpQuantizer(
                torch.ops.aten.mul.Tensor,
                torch.ops.fused_quant.mul.default,
                config,
                activation_names=("input", "other"),
            )
        ]
    )
    case = test_cases["float_users"]
    tester = EspTester(case.model, case.example_inputs).compile_fused_quant(quantizer)
    tester.check_node_count({_FUSED_MUL: 1})
    expected = tester.get_artifact().exported_program().module()(*case.example_inputs)
    tester.run_fused_quant_passes().check_node_count(
        {_FUSED_MUL: 0, _ESP_MUL: 0, _MUL: 1}
    )
    torch.testing.assert_close(
        tester.get_artifact().exported_program().module()(*case.example_inputs),
        expected,
        atol=0,
        rtol=0,
    )
    tester.to_executorch().serialize()
