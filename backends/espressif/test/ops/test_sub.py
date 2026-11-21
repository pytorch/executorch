# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

import pytest
import torch
from executorch.backends.espressif import (
    get_esp_fused_quant_recipe,
    get_operator_library_report,
)
from executorch.backends.espressif.test.tester import (
    EspTestCase,
    EspTester,
    ramp_tensor,
)
from executorch.backends.test.suite.operators.test_sub import Model, ModelAlpha
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
)
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.export import export
from executorch.frontends.fused_quant.quantizer.quantizer import (
    FusedQuantQuantizer,
    OpQuantizer,
)

_SUB = exir_ops.edge.aten.sub.Tensor
_FQ_SUB = exir_ops.edge.fused_quant.sub.default
_ESP_ADD = exir_ops.edge.esp_nn.quantized_add_int8.default


class EspSub(Model):
    ops_before_transforms = {_FQ_SUB: 1}
    ops_after_transforms = {_SUB: 0, _ESP_ADD: 1}


class EspSelfSub(EspSub):
    def forward(self, x):
        return x - x


class EspConstantSub(EspSub):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.linspace(3, -2, 17))

    def forward(self, x):
        return x - self.weight


class EspSubChain(EspSub):
    ops_before_transforms = {_FQ_SUB: 2}
    ops_after_transforms = {_SUB: 0, _ESP_ADD: 2}

    def forward(self, x, y):
        difference = x - y
        return difference - x, torch.sin(difference), torch.sin(x)


class EspSubWithFloatUsers(EspSub):
    def forward(self, x, y):
        difference = x - y
        return difference, torch.sin(x), torch.sin(difference)


class EspPortableSub(EspSubWithFloatUsers):
    ops_after_transforms = {_SUB: 1, _ESP_ADD: 0}


class EspAlphaSub(ModelAlpha):
    ops_before_transforms = {_FQ_SUB: 1}
    ops_after_transforms = {_SUB: 1, _ESP_ADD: 0}


class EspSubRelu(EspSub):
    ops_before_transforms = {_FQ_SUB: 1, exir_ops.edge.fused_quant.relu.default: 1}
    ops_after_transforms = {
        _SUB: 0,
        _ESP_ADD: 1,
        exir_ops.edge.esp_nn.quantized_lut_int8.default: 0,
        exir_ops.edge.aten.relu.default: 0,
    }

    def forward(self, x, y):
        return torch.relu(x - y)


test_cases = {
    "scalar": EspTestCase(EspSub(), (torch.tensor(-1.5), torch.tensor(3.0))),
    "short_vector": EspTestCase(
        EspSub(), (ramp_tensor(-1.5, 2.5, (7,)), ramp_tensor(3, -4, (7,)))
    ),
    "vector_tail": EspTestCase(
        EspSub(), (ramp_tensor(-1.5, 2.5, (17,)), ramp_tensor(3, -4, (17,)))
    ),
    "aligned_vector": EspTestCase(
        EspSub(), (ramp_tensor(-1.5, 2.5, (32,)), ramp_tensor(3, -4, (32,)))
    ),
    "rank_four": EspTestCase(
        EspSub(),
        (ramp_tensor(-1.5, 2.5, (1, 2, 3, 4)), ramp_tensor(3, -4, (1, 2, 3, 4))),
    ),
    "unequal_scales": EspTestCase(
        EspSub(), (ramp_tensor(-0.01, 0.02, (17,)), ramp_tensor(30, -40, (17,)))
    ),
    "self": EspTestCase(EspSelfSub(), (ramp_tensor(-1, 2, (17,)),)),
    "cancellation": EspTestCase(
        EspSub(),
        (ramp_tensor(-20, 30, (33,)), ramp_tensor(-20, 30, (33,)) + 0.0001),
    ),
    "constant": EspTestCase(EspConstantSub(), (ramp_tensor(-1, 2, (17,)),)),
    "chain": EspTestCase(
        EspSubChain(), (ramp_tensor(-1, 1, (17,)), ramp_tensor(2, -2, (17,)))
    ),
    "float_users": EspTestCase(
        EspSubWithFloatUsers(),
        (ramp_tensor(-1, 1, (2, 3)), ramp_tensor(-2, 2, (2, 3))),
    ),
    "broadcast": EspTestCase(
        EspPortableSub(), (ramp_tensor(-1, 1, (2, 3)), ramp_tensor(-2, 2, (3,)))
    ),
    "rescale_fallback": EspTestCase(
        EspPortableSub(), (ramp_tensor(-1e8, 1e8, (17,)), ramp_tensor(-1e8, 1e8, (17,)))
    ),
    "alpha": EspTestCase(
        EspAlphaSub(alpha=2),
        (ramp_tensor(-1, 1, (17,)), ramp_tensor(2, -2, (17,))),
    ),
    "relu": EspTestCase(
        EspSubRelu(), (ramp_tensor(-1, 1, (17,)), ramp_tensor(2, -2, (17,)))
    ),
    "saturation": EspTestCase(
        EspSub(),
        (torch.tensor([-2.0, -2.0, 2.0, 2.0]), torch.tensor([-2.0, 2.0, -2.0, 2.0])),
        calibration_samples=[
            (torch.tensor([-2.0, 2.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, -2.0, 2.0]))
        ],
    ),
}


@pytest.mark.parametrize("case", test_cases.values(), ids=test_cases.keys())
def test_dialect_sub(case):
    accelerated = case.model.ops_after_transforms[_ESP_ADD] > 0
    tester = EspTester(case.model, case.example_inputs, case.calibration_samples)
    tester.test_dialect(
        case.model.ops_before_transforms,
        {
            _FQ_SUB: 0,
            exir_ops.edge.aten.neg.default: 0,
            **case.model.ops_after_transforms,
        },
        qtol=1 if accelerated else 0,
        atol=0,
        rtol=torch.finfo(torch.float32).eps if accelerated else 0,
    )
    if isinstance(case.model, EspSubWithFloatUsers):
        actual = tester.get_artifact().exported_program().module()(*case.example_inputs)
        torch.testing.assert_close(
            actual[1], torch.sin(case.example_inputs[0]), atol=0, rtol=0
        )
    tester.to_executorch().serialize()
    plan = deserialize_pte_binary(tester.get_artifact()).program.execution_plan[0]
    assert not plan.delegates
    assert all(not op.name.startswith("fused_quant::") for op in plan.operators)
    assert (
        any(op.name == "esp_nn::quantized_add_int8" for op in plan.operators)
        == accelerated
    )


@pytest.mark.parametrize("case", test_cases.values(), ids=test_cases.keys())
def test_implementation_sub(case, pytestconfig):
    EspTester(
        case.model, case.example_inputs, case.calibration_samples
    ).test_implementation(
        pytestconfig.rootpath, rtol=torch.finfo(torch.float32).eps, num_inferences=2
    )


@pytest.mark.parametrize("float_edge", ["input_activation", "output_activation"])
def test_partially_quantized_sub_falls_back(float_edge):
    config = replace(get_symmetric_quantization_config(), **{float_edge: None})
    quantizer = FusedQuantQuantizer(
        [
            OpQuantizer(
                torch.ops.aten.sub.Tensor,
                torch.ops.fused_quant.sub.default,
                config,
                activation_names=("input", "other"),
            )
        ]
    )
    case = test_cases["float_users"]
    tester = (
        EspTester(case.model, case.example_inputs)
        .export_fused_quant(quantizer)
        .to_edge()
    )
    tester.check_node_count({_FQ_SUB: 1})
    expected = tester.get_artifact().exported_program().module()(*case.example_inputs)
    tester.run_passes().check_node_count({_FQ_SUB: 0, _ESP_ADD: 0, _SUB: 1})
    torch.testing.assert_close(
        tester.get_artifact().exported_program().module()(*case.example_inputs),
        expected,
        atol=0,
        rtol=0,
    )
    tester.to_executorch().serialize()


def test_public_recipe_sub():
    case = test_cases["vector_tail"]
    session = export(
        case.model,
        example_inputs=[case.example_inputs],
        export_recipe=get_esp_fused_quant_recipe(allow_cpu_fallback=False),
    )
    report = get_operator_library_report(session.get_edge_program_manager())
    assert report["esp_nn_ops"] == 1
    assert report["cpu_ops"] == []
    assert report["layout_ops"] == []
    assert report["quantization_ops"]
    assert session.get_executorch_program_manager().buffer
