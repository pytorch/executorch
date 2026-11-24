# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

import pytest
import torch
from executorch.backends.espressif.lowering import (
    get_esp_fused_quant_recipe,
    get_operator_library_report,
)
from executorch.backends.espressif.test.tester import (
    EspTestCase,
    EspTester,
    ramp_tensor,
)
from executorch.backends.test.suite.operators.test_linear import Model
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.backends.transforms.utils import get_constant
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

_FQ_LINEAR = exir_ops.edge.fused_quant.linear.default
_ESP_LINEAR = exir_ops.edge.esp_nn.quantized_linear_int8.default


class EspLinear(Model):
    ops_before_transforms = {_FQ_LINEAR: 1}
    ops_after_transforms = {_ESP_LINEAR: 1}

    def __init__(self, in_features=17, out_features=5, bias=True):
        super().__init__(in_features, out_features, bias)
        with torch.no_grad():
            values = torch.arange(in_features * out_features).reshape(
                out_features, in_features
            )
            self.linear.weight.copy_(
                values.sin() * torch.arange(1, out_features + 1).reshape(-1, 1) * 0.02
            )
            if bias:
                self.linear.bias.copy_(torch.linspace(-0.5, 0.7, out_features))


class EspLinearWithFloatUsers(EspLinear):
    def forward(self, x):
        result = self.linear(x)
        return result, torch.sin(result), torch.sin(x), self.linear.bias


class EspLinearRelu(EspLinear):
    ops_before_transforms = {_FQ_LINEAR: 1, exir_ops.edge.fused_quant.relu.default: 1}
    ops_after_transforms = {_ESP_LINEAR: 1, exir_ops.edge.aten.relu.default: 1}

    def forward(self, x):
        return torch.relu(self.linear(x))


class EspSharedLinear(EspLinear):
    ops_before_transforms = {_FQ_LINEAR: 2}
    ops_after_transforms = {_ESP_LINEAR: 2}

    def forward(self, x, y):
        return self.linear(x), self.linear(y), self.linear.bias


class EspLinearChain(EspLinear):
    ops_before_transforms = {_FQ_LINEAR: 2}
    ops_after_transforms = {_ESP_LINEAR: 2}

    def __init__(self):
        super().__init__(17, 17)

    def forward(self, x):
        return self.linear(self.linear(x))


class EspLinearLargeBias(EspLinear):
    def __init__(self):
        super().__init__(2, 3)
        with torch.no_grad():
            self.linear.bias.copy_(torch.tensor([500.0, 100.0, 0.1]))


class EspRuntimeBias(EspLinear):
    ops_after_transforms = {
        _ESP_LINEAR: 0,
        exir_ops.edge.aten.addmm.default: 1,
    }

    def forward(self, x, bias):
        return torch.nn.functional.linear(x, self.linear.weight, bias)


class EspRuntimeWeight(torch.nn.Module):
    ops_before_transforms = {_FQ_LINEAR: 0, exir_ops.edge.aten.mm.default: 1}
    ops_after_transforms = {_ESP_LINEAR: 0, exir_ops.edge.aten.mm.default: 1}

    def forward(self, x, weight):
        return torch.nn.functional.linear(x, weight)


test_cases = {
    "bias": EspTestCase(EspLinear(), (ramp_tensor(-3, 5, (2, 17)),)),
    "no_bias": EspTestCase(EspLinear(bias=False), (ramp_tensor(-3, 5, (2, 17)),)),
    "vector": EspTestCase(EspLinear(), (ramp_tensor(-3, 5, (17,)),)),
    "rank_3": EspTestCase(EspLinear(), (ramp_tensor(-3, 5, (2, 3, 17)),)),
    "rank_4": EspTestCase(EspLinear(), (ramp_tensor(-3, 5, (2, 1, 3, 17)),)),
    "single_row": EspTestCase(EspLinear(), (ramp_tensor(-3, 5, (1, 17)),)),
    "single_output": EspTestCase(
        EspLinear(out_features=1), (ramp_tensor(-3, 5, (3, 17)),)
    ),
    "single_feature": EspTestCase(
        EspLinear(in_features=1), (ramp_tensor(-3, 5, (4, 1)),)
    ),
    "aligned": EspTestCase(EspLinear(in_features=256), (ramp_tensor(-3, 5, (3, 256)),)),
    "tail": EspTestCase(EspLinear(in_features=273), (ramp_tensor(-3, 5, (3, 273)),)),
    "float_users": EspTestCase(
        EspLinearWithFloatUsers(), (ramp_tensor(-3, 5, (2, 17)),)
    ),
    "relu": EspTestCase(EspLinearRelu(), (ramp_tensor(-3, 5, (2, 17)),)),
    "shared_bias": EspTestCase(
        EspSharedLinear(), (ramp_tensor(-3, 5, (2, 17)), ramp_tensor(-24, 40, (2, 17)))
    ),
    "chain": EspTestCase(EspLinearChain(), (ramp_tensor(-3, 5, (2, 17)),)),
    "large_bias": EspTestCase(EspLinearLargeBias(), (ramp_tensor(-3, 5, (2, 2)),)),
    "runtime_bias": EspTestCase(
        EspRuntimeBias(),
        (ramp_tensor(-3, 5, (2, 17)), ramp_tensor(-0.5, 0.7, (5,))),
    ),
    "runtime_weight": EspTestCase(
        EspRuntimeWeight(),
        (ramp_tensor(-3, 5, (2, 17)), ramp_tensor(-0.2, 0.3, (5, 17))),
    ),
    "saturation": EspTestCase(
        EspLinear(),
        (ramp_tensor(-30, 50, (2, 17)),),
        calibration_samples=[(ramp_tensor(-3, 5, (2, 17)),)],
    ),
}


@pytest.mark.parametrize("test_case", test_cases.values(), ids=test_cases.keys())
def test_dialect_linear(test_case):
    tester = EspTester(
        test_case.model, test_case.example_inputs, test_case.calibration_samples
    )
    tester.test_dialect(
        test_case.model.ops_before_transforms,
        {_FQ_LINEAR: 0, **test_case.model.ops_after_transforms},
        qtol=1,
        atol=0,
        rtol=torch.finfo(torch.float32).eps,
    )
    tester.to_executorch().serialize()
    program = deserialize_pte_binary(tester.get_artifact()).program
    plan = program.execution_plan[0]
    assert not plan.delegates
    assert any(op.name == "esp_nn::quantized_linear_int8" for op in plan.operators) == (
        test_case.model.ops_after_transforms[_ESP_LINEAR] > 0
    )
    assert all(
        not op.name.startswith("fused_quant::") and op.name != "aten::linear"
        for op in plan.operators
    )


@pytest.mark.parametrize("test_case", test_cases.values(), ids=test_cases.keys())
def test_implementation_linear(test_case, pytestconfig):
    tester = EspTester(
        test_case.model, test_case.example_inputs, test_case.calibration_samples
    )
    tester.test_implementation(
        pytestconfig.rootpath, rtol=torch.finfo(torch.float32).eps
    )


def test_public_recipe_linear():
    case = test_cases["rank_3"]
    session = export(
        case.model,
        example_inputs=[case.example_inputs],
        export_recipe=get_esp_fused_quant_recipe(allow_cpu_fallback=False),
    )
    assert get_operator_library_report(session.get_edge_program_manager()) == {
        "esp_nn_ops": 1,
        "cpu_ops": [],
        "layout_ops": [],
    }
    assert session.get_executorch_program_manager().buffer


@pytest.mark.parametrize("case_name", ["shared_bias", "large_bias"])
def test_lowering_reuses_quantized_bias_and_preserves_source_program(case_name):
    case = test_cases[case_name]
    tester = EspTester(case.model, case.example_inputs).export_fused_quant().to_edge()
    source = tester.get_artifact().exported_program()
    original_graph = str(source.graph)
    biases = [
        get_constant(source, get_arg(node, "bias")).clone()
        for node in source.graph.find_nodes(op="call_function", target=_FQ_LINEAR)
    ]
    tester.run_passes()
    lowered = tester.get_artifact().exported_program()
    nodes = list(lowered.graph.find_nodes(op="call_function", target=_ESP_LINEAR))
    assert len(nodes) == len(biases) == case.model.ops_after_transforms[_ESP_LINEAR]
    for node, expected in zip(nodes, biases):
        torch.testing.assert_close(
            get_constant(lowered, get_arg(node, "bias")), expected, atol=0, rtol=0
        )
    assert str(source.graph) == original_graph
    source.validate()


@pytest.mark.parametrize(
    "model, inputs",
    [
        (EspRuntimeBias(), (torch.ones(2, 17), torch.ones(5))),
        (EspRuntimeWeight(), (torch.ones(2, 17), torch.ones(5, 17))),
        (EspLinear(in_features=65536), (torch.ones(1, 65536),)),
    ],
    ids=["runtime_bias", "runtime_weight", "oversized_dimension"],
)
def test_public_recipe_unsupported_linear_falls_back(model, inputs):
    session = export(
        model, example_inputs=[inputs], export_recipe=get_esp_fused_quant_recipe()
    )
    report = get_operator_library_report(session.get_edge_program_manager())
    assert report["esp_nn_ops"] == 0
    assert {op["target"] for op in report["cpu_ops"]} <= {
        "aten::addmm.out",
        "aten::mm.out",
        "aten::permute_copy.out",
    }
    buffer = session.get_executorch_program_manager().buffer
    operators = deserialize_pte_binary(buffer).program.execution_plan[0].operators
    assert any(op.name in {"aten::addmm", "aten::mm"} for op in operators)
    assert all(
        op.name != "aten::linear" and not op.name.startswith("fused_quant::")
        for op in operators
    )
    portable = pytest.importorskip("executorch.extension.pybindings.portable_lib")
    runtime = portable._load_for_executorch_from_buffer(buffer)
    torch.testing.assert_close(
        runtime.forward(inputs)[0],
        session.get_exported_program().module()(*inputs),
        atol=1e-6,
        rtol=1e-6,
    )


def test_strict_export_rejects_portable_linear():
    case = test_cases["runtime_bias"]
    with pytest.raises(RuntimeError, match="aten::addmm"):
        export(
            case.model,
            example_inputs=[case.example_inputs],
            export_recipe=get_esp_fused_quant_recipe(allow_cpu_fallback=False),
        )


@pytest.mark.parametrize("float_edge", ["input_activation", "output_activation"])
def test_partially_quantized_linear_falls_back(float_edge):
    config = replace(
        get_symmetric_quantization_config(is_per_channel=True), **{float_edge: None}
    )
    quantizer = FusedQuantQuantizer(
        [
            OpQuantizer(
                torch.ops.aten.linear.default,
                torch.ops.fused_quant.linear.default,
                config,
                weight_names=("weight",),
                other_names=("bias",),
            )
        ]
    )
    case = test_cases["no_bias"]
    tester = (
        EspTester(case.model, case.example_inputs)
        .export_fused_quant(quantizer)
        .to_edge()
    )
    expected = tester.get_artifact().exported_program().module()(*case.example_inputs)
    tester.run_passes().check_node_count(
        {_FQ_LINEAR: 0, _ESP_LINEAR: 0, exir_ops.edge.aten.mm.default: 1}
    )
    torch.testing.assert_close(
        tester.get_artifact().exported_program().module()(*case.example_inputs),
        expected,
        atol=1e-6,
        rtol=1e-6,
    )
    tester.to_executorch().serialize()


def test_accumulator_overflow_falls_back():
    model = torch.nn.Linear(2, 3).eval()
    with torch.no_grad():
        model.weight.zero_()
        model.bias.copy_(torch.tensor([127.0, 128.0, -128.0]))
    tester = EspTester(model, (torch.zeros(1, 2),)).export_fused_quant().to_edge()
    expected = tester.get_artifact().exported_program().module()(*tester.example_inputs)
    tester.run_passes().check_node_count(
        {_FQ_LINEAR: 0, _ESP_LINEAR: 0, exir_ops.edge.aten.addmm.default: 1}
    )
    torch.testing.assert_close(
        tester.get_artifact().exported_program().module()(*tester.example_inputs),
        expected,
        atol=1e-6,
        rtol=1e-6,
    )
    tester.to_executorch().serialize()
