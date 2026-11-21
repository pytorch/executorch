# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import pytest
import torch
from executorch.backends.espressif.passes.fold_activations import FoldActivations
from executorch.backends.espressif.test.tester import (
    EspTestCase,
    EspTester,
    ramp_tensor,
)
from executorch.backends.transforms.fused_quant.graph_utils import get_qparams_from_node
from executorch.backends.transforms.permute_pass_utils import get_arg, set_arg
from executorch.backends.transforms.utils import get_constant
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
)
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.frontends.fused_quant.quantizer.quantizer import (
    FusedQuantQuantizer,
    get_core_aten_quantizers,
)

_OPERATORS = {
    "add": (
        torch.add,
        exir_ops.edge.fused_quant.add.default,
        exir_ops.edge.esp_nn.quantized_add_int8.default,
    ),
    "sub": (
        torch.sub,
        exir_ops.edge.fused_quant.sub.default,
        exir_ops.edge.esp_nn.quantized_add_int8.default,
    ),
    "mul": (
        torch.mul,
        exir_ops.edge.fused_quant.mul.default,
        exir_ops.edge.esp_nn.quantized_mul_int8.default,
    ),
}
_ESP_MUL_BROADCAST = exir_ops.edge.esp_nn.quantized_mul_broadcast_int8.default
_CLAMP_CASES = [pytest.param(binary, False, id=binary) for binary in _OPERATORS] + [
    pytest.param("mul", True, id="mul_broadcast")
]
_ACTIVATIONS = {
    "relu": (torch.nn.ReLU, exir_ops.edge.fused_quant.relu.default),
    "relu6": (torch.nn.ReLU6, exir_ops.edge.fused_quant.hard_tanh.default),
    "hardtanh": (torch.nn.Hardtanh, exir_ops.edge.fused_quant.hard_tanh.default),
    "hardtanh_asymmetric": (
        partial(torch.nn.Hardtanh, min_val=-2.0625, max_val=3.1875),
        exir_ops.edge.fused_quant.hard_tanh.default,
    ),
    "hardtanh_positive": (
        partial(torch.nn.Hardtanh, min_val=1, max_val=4),
        exir_ops.edge.fused_quant.hard_tanh.default,
    ),
    "hardtanh_negative": (
        partial(torch.nn.Hardtanh, min_val=-5, max_val=-2),
        exir_ops.edge.fused_quant.hard_tanh.default,
    ),
}
_ESP_LUT = exir_ops.edge.esp_nn.quantized_lut_int8.default


class BinaryActivation(torch.nn.Module):
    def __init__(
        self, binary, activation, *, fork=False, accelerated=True, broadcast=False
    ):
        super().__init__()
        self.binary, self.fused_binary, self.esp_binary = _OPERATORS[binary]
        if broadcast:
            self.esp_binary = _ESP_MUL_BROADCAST
        activation_cls, self.fused_activation = _ACTIVATIONS[activation]
        self.activation = activation_cls()
        self.fork = fork
        self.accelerated = accelerated
        self.ops_before_transforms = {self.fused_binary: 1, self.fused_activation: 1}
        self.ops_after_transforms = {
            self.fused_binary: 0,
            self.fused_activation: 0,
            self.esp_binary: int(accelerated),
            _ESP_LUT: int(fork),
        }
        if binary == "mul":
            other_mul = _OPERATORS["mul"][2] if broadcast else _ESP_MUL_BROADCAST
            self.ops_after_transforms[other_mul] = 0

    def forward(self, x, y):
        result = self.binary(x, y)
        activated = self.activation(result)
        return (activated, result, torch.sin(result)) if self.fork else activated


test_cases = {
    f"{binary}_{activation}{'_fork' if fork else ''}": EspTestCase(
        BinaryActivation(binary, activation, fork=fork),
        (ramp_tensor(-8, 9, (257,)), torch.sin(torch.linspace(0, 9, 257)) * 3),
    )
    for binary in _OPERATORS
    for activation in _ACTIVATIONS
    for fork in (False, True)
}
test_cases.update(
    {
        "mul_relu6_broadcast": EspTestCase(
            BinaryActivation("mul", "relu6", broadcast=True),
            (ramp_tensor(-5, 5, (3, 17)), ramp_tensor(-3, 4, (17,))),
        ),
        "mul_relu6_broadcast_reversed": EspTestCase(
            BinaryActivation("mul", "relu6", broadcast=True),
            (ramp_tensor(-3, 4, (17,)), ramp_tensor(-5, 5, (3, 17))),
        ),
        "mul_hardtanh_broadcast_reversed": EspTestCase(
            BinaryActivation("mul", "hardtanh_asymmetric", broadcast=True),
            (ramp_tensor(-3, 4, (17,)), ramp_tensor(-5, 5, (3, 17))),
        ),
        "add_hardtanh_broadcast_fallback": EspTestCase(
            BinaryActivation("add", "hardtanh_asymmetric", accelerated=False),
            (ramp_tensor(-5, 5, (3, 17)), ramp_tensor(-3, 4, (17,))),
        ),
        "add_relu6_broadcast_fallback": EspTestCase(
            BinaryActivation("add", "relu6", accelerated=False),
            (ramp_tensor(-5, 5, (3, 17)), ramp_tensor(-3, 4, (17,))),
        ),
        "sub_relu_broadcast_fallback": EspTestCase(
            BinaryActivation("sub", "relu", accelerated=False),
            (ramp_tensor(-5, 5, (3, 17)), ramp_tensor(-3, 4, (17,))),
        ),
        "mul_relu6_broadcast_fallback": EspTestCase(
            BinaryActivation("mul", "relu6", accelerated=False),
            (ramp_tensor(-5, 5, (3, 17)), ramp_tensor(-3, 4, (3, 1))),
        ),
        "sub_relu6_rescale_fallback": EspTestCase(
            BinaryActivation("sub", "relu6", accelerated=False),
            (ramp_tensor(-1e8, 1e8, (17,)), ramp_tensor(-1e8, 1e8, (17,))),
        ),
        "add_relu6_saturation": EspTestCase(
            BinaryActivation("add", "relu6"),
            (
                torch.tensor([-16.0, 0.0, 3.0, 16.0]),
                torch.tensor([-8.0, 0.0, 1.0, 8.0]),
            ),
            calibration_samples=[
                (
                    torch.tensor([-8.0, 0.0, 2.0, 8.0]),
                    torch.tensor([-8.0, 0.0, 2.0, 8.0]),
                )
            ],
        ),
    }
)


@pytest.mark.parametrize("case", test_cases.values(), ids=test_cases.keys())
def test_dialect_binary_activation(case):
    tester = EspTester(case.model, case.example_inputs, case.calibration_samples)
    tester.test_dialect(
        case.model.ops_before_transforms,
        case.model.ops_after_transforms,
        qtol=1 if case.model.accelerated else 0,
        atol=0,
        rtol=torch.finfo(torch.float32).eps if case.model.accelerated else 0,
    )
    tester.to_executorch().serialize()
    plan = deserialize_pte_binary(tester.get_artifact()).program.execution_plan[0]
    assert not plan.delegates
    assert not any(op.name.startswith("fused_quant::") for op in plan.operators)
    if case.model.accelerated and not case.model.fork:
        assert sum(op.name.startswith("esp_nn::") for op in plan.operators) == 1
        assert not any(op.name.startswith("aten::") for op in plan.operators)


@pytest.mark.parametrize("case", test_cases.values(), ids=test_cases.keys())
def test_implementation_binary_activation(case, pytestconfig):
    EspTester(
        case.model, case.example_inputs, case.calibration_samples
    ).test_implementation(
        pytestconfig.rootpath, rtol=torch.finfo(torch.float32).eps, num_inferences=2
    )


@pytest.mark.parametrize("binary", _OPERATORS)
@pytest.mark.parametrize("activation", _ACTIVATIONS)
@pytest.mark.parametrize("fork", [False, True])
def test_quantization_boundary(binary, activation, fork):
    model = BinaryActivation(binary, activation, fork=fork)
    tester = EspTester(model, (torch.linspace(-8, 8, 17), torch.ones(17)))
    tester.export_fused_quant().to_edge()
    graph = tester.get_artifact().exported_program().graph
    (producer,) = graph.find_nodes(op="call_function", target=model.fused_binary)
    (activation_node,) = graph.find_nodes(
        op="call_function", target=model.fused_activation
    )
    assert (get_qparams_from_node(producer, "out") is not None) == fork
    assert (get_qparams_from_node(activation_node, "inp") is not None) == fork
    assert get_qparams_from_node(activation_node, "out") is not None


@pytest.mark.parametrize("binary", _OPERATORS)
@pytest.mark.parametrize("activation", _ACTIVATIONS)
def test_separately_quantized_activation_keeps_rounding(binary, activation):
    model = BinaryActivation(binary, activation)
    config = get_symmetric_quantization_config()
    tester = EspTester(model, (torch.linspace(-8, 8, 17), torch.linspace(-1, 4, 17)))
    tester.export_fused_quant(
        quantizer=FusedQuantQuantizer(get_core_aten_quantizers(config, config))
    ).to_edge()
    assert not FoldActivations()(tester.get_artifact().exported_program()).modified
    tester.run_passes().check_node_count({model.esp_binary: 1, _ESP_LUT: 1})


def _set_output_grid(program, activation_target):
    (node,) = program.graph.find_nodes(op="call_function", target=activation_target)
    params = get_qparams_from_node(node, "out")
    get_constant(program, params.scale).fill_(0.125)
    get_constant(program, params.zero_point).fill_(-7)
    (output,) = program.graph.find_nodes(op="output")
    dequant = output.args[0][0]
    set_arg(dequant, "scale", 0.125)
    set_arg(dequant, "zero_point", -7)
    program.graph_module.recompile()


@pytest.mark.parametrize("binary, broadcast", _CLAMP_CASES)
@pytest.mark.parametrize(
    "activation, bounds",
    [
        ("relu", (-7, 127)),
        ("relu6", (-7, 41)),
        ("hardtanh", (-15, 1)),
        ("hardtanh_asymmetric", (-23, 19)),
        ("hardtanh_positive", (1, 25)),
        ("hardtanh_negative", (-47, -23)),
    ],
)
def test_fold_uses_final_qparams_and_clamp(binary, broadcast, activation, bounds):
    model = BinaryActivation(binary, activation, broadcast=broadcast)
    inputs = (
        ramp_tensor(-20, 20, (3, 257) if broadcast else (257,)),
        torch.linspace(-2, 3, 257),
    )
    tester = EspTester(model, inputs).export_fused_quant().to_edge()
    program = tester.get_artifact().exported_program()
    _set_output_grid(program, model.fused_activation)
    expected = program.module()(*inputs)
    assert FoldActivations()(program).modified
    torch.testing.assert_close(program.module()(*inputs), expected, atol=0, rtol=0)
    tester.run_passes().check_node_count({model.esp_binary: 1, _ESP_LUT: 0})
    lowered = tester.get_artifact().exported_program()
    (node,) = lowered.graph.find_nodes(op="call_function", target=model.esp_binary)
    assert get_arg(node, "output_offset") == -7
    assert (get_arg(node, "activation_min"), get_arg(node, "activation_max")) == bounds
    torch.testing.assert_close(lowered.module()(*inputs), expected, atol=0.125, rtol=0)


@pytest.mark.parametrize(
    "dtype, min_val",
    [(torch.bfloat16, 0.3333), (torch.float16, 0.33167)],
    ids=["bfloat16", "float16"],
)
def test_fold_preserves_activation_dtype(dtype, min_val):
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return torch.nn.functional.hardtanh(x + y, min_val, 0.3349)

    inputs = (
        torch.linspace(-1, 1, 17, dtype=dtype),
        torch.zeros(17, dtype=dtype),
    )
    tester = EspTester(Model(), inputs).export_fused_quant().to_edge()
    program = tester.get_artifact().exported_program()
    expected = program.module()(*inputs)
    assert FoldActivations()(program).modified
    torch.testing.assert_close(program.module()(*inputs), expected, atol=0, rtol=0)
    tester.run_passes().check_node_count(
        {
            exir_ops.edge.fused_quant.add.default: 0,
            exir_ops.edge.fused_quant.hard_tanh.default: 0,
            exir_ops.edge.esp_nn.quantized_add_int8.default: 0,
            _ESP_LUT: 0,
        }
    )
    lowered = tester.get_artifact().exported_program()
    torch.testing.assert_close(lowered.module()(*inputs), expected, atol=0, rtol=0)


@pytest.mark.parametrize("binary, broadcast", _CLAMP_CASES)
@pytest.mark.parametrize("activation", _ACTIVATIONS)
def test_implementation_explicit_clamp(binary, broadcast, activation, pytestconfig):
    # Calibration can map activation limits to the int8 endpoints.
    # Exercise the kernel's narrower clamp with another valid output grid.
    model = BinaryActivation(binary, activation, broadcast=broadcast)
    tester = EspTester(
        model,
        (
            ramp_tensor(-20, 20, (3, 257) if broadcast else (257,)),
            torch.ones(257),
        ),
    )
    tester.export_fused_quant().to_edge()
    program = tester.get_artifact().exported_program()
    _set_output_grid(program, model.fused_activation)
    tester.run_passes()
    tester.test_implementation(pytestconfig.rootpath, num_inferences=2)
