# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math

import pytest
import torch
from executorch.backends.espressif.lowering import get_esp_fused_quant_recipe
from executorch.backends.espressif.quantizer import get_esp_fused_quant_quantizer
from executorch.backends.espressif.test.tester import EspTester
from executorch.backends.transforms.fused_quant.graph_utils import get_qparams_from_node
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.export import export
from torchao.quantization.pt2e.quantizer import QuantizationAnnotation
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY


@pytest.fixture
def linear():
    model = torch.nn.Linear(2, 3).eval()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.2, -0.1], [0.03, 0.05], [-0.4, 0.1]]))
        model.bias.copy_(torch.tensor([0.125, -0.3, 0.6]))
    return model


def _quantize(model, inputs):
    program = EspTester(model, inputs).export_fused_quant().get_artifact()
    interpreter = torch.fx.Interpreter(program.module(), garbage_collect_values=False)
    interpreter.run(*inputs)
    return interpreter


@pytest.mark.parametrize(
    "activation, target",
    [
        (torch.nn.Sigmoid, torch.ops.fused_quant.sigmoid.default),
        (torch.nn.SiLU, torch.ops.fused_quant.silu.default),
        (torch.nn.Hardswish, torch.ops.fused_quant.hardswish.default),
        (torch.nn.Hardsigmoid, torch.ops.fused_quant.hardsigmoid.default),
    ],
    ids=["sigmoid", "silu", "hardswish", "hardsigmoid"],
)
def test_lut_output_qparams_are_calibrated(activation, target):
    scales = []
    for low, high in [(-16, 16), (-3, -1)]:
        interpreter = _quantize(activation(), (torch.linspace(low, high, 257),))
        (node,) = interpreter.module.graph.find_nodes(op="call_function", target=target)
        params = get_qparams_from_node(node, "out")
        assert params.dtype == torch.int8
        assert (params.quant_min, params.quant_max) == (-128, 127)
        scales.append(interpreter.env[params.scale].item())
        assert interpreter.env[get_arg(node, "inp")].dtype == torch.int8
    assert scales[0] > scales[1]


def _check_bias(interpreter, node, original_bias):
    values = interpreter.env
    inp = get_qparams_from_node(node, "inp")
    weight = get_qparams_from_node(node, "weight")
    bias = get_qparams_from_node(node, "bias")
    scales = values[bias.scale]
    torch.testing.assert_close(
        scales, values[inp.scale] * values[weight.scale].flatten(), atol=0, rtol=0
    )
    assert torch.count_nonzero(values[bias.zero_point]) == 0
    stored = values[get_arg(node, "bias")]
    assert stored.dtype == torch.int32
    expected = torch.round(original_bias / scales).clamp(bias.quant_min, bias.quant_max)
    torch.testing.assert_close(stored, expected.to(torch.int32), atol=0, rtol=0)


@pytest.mark.parametrize("shape", [(2,), (3, 2), (2, 3, 2)])
def test_linear_preservation_and_int32_bias(linear, shape):
    inputs = (torch.linspace(-2, 3, math.prod(shape)).reshape(shape),)
    interpreter = _quantize(linear, inputs)
    (node,) = interpreter.module.graph.find_nodes(
        op="call_function", target=torch.ops.fused_quant.linear.default
    )
    weight = interpreter.env[get_arg(node, "weight")]
    weight_qparams = get_qparams_from_node(node, "weight")
    assert weight.dtype == torch.int8
    assert interpreter.env[weight_qparams.scale].shape == (3, 1)
    assert torch.count_nonzero(interpreter.env[weight_qparams.zero_point]) == 0
    _check_bias(interpreter, node, linear.bias)
    assert interpreter.env[node].shape == (*shape[:-1], 3)


def test_linear_without_bias():
    interpreter = _quantize(torch.nn.Linear(2, 3, bias=False), (torch.ones(1, 2),))
    (node,) = interpreter.module.graph.find_nodes(
        op="call_function", target=torch.ops.fused_quant.linear.default
    )
    assert get_arg(node, "bias") is None
    assert get_qparams_from_node(node, "bias") is None


@pytest.mark.parametrize("out_features", [1, 3])
def test_scalar_linear_bias_stays_float(out_features):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "weight", torch.arange(out_features * 2).reshape(out_features, 2) * 0.05
            )
            self.register_buffer("bias", torch.tensor(0.125))

        def forward(self, x):
            return torch.nn.functional.linear(x, self.weight, self.bias)

    inputs = (torch.tensor([[1.0, -2.0], [0.5, 3.0]]),)
    session = export(
        Model(), example_inputs=[inputs], export_recipe=get_esp_fused_quant_recipe()
    )
    reference = session.get_exported_program()
    (node,) = reference.graph.find_nodes(
        op="call_function", target=torch.ops.fused_quant.linear.default
    )
    assert get_qparams_from_node(node, "bias") is None
    buffer = session.get_executorch_program_manager().buffer
    operators = deserialize_pte_binary(buffer).program.execution_plan[0].operators
    assert any(op.name == "aten::addmm" for op in operators)
    assert all(
        not op.name.startswith(("esp_nn::", "fused_quant::")) for op in operators
    )
    portable = pytest.importorskip("executorch.extension.pybindings.portable_lib")
    runtime = portable._load_for_executorch_from_buffer(buffer)
    torch.testing.assert_close(
        runtime.forward(inputs)[0], reference.module()(*inputs), atol=1e-6, rtol=1e-6
    )


def test_runtime_linear_weights_stay_float():
    class Model(torch.nn.Module):
        def forward(self, x, weight):
            return torch.nn.functional.linear(x, weight)

    interpreter = _quantize(Model(), (torch.ones(2, 4), torch.ones(3, 4)))
    (node,) = interpreter.module.graph.find_nodes(
        op="call_function", target=torch.ops.aten.linear.default
    )
    weight = get_arg(node, "weight")
    assert weight.op == "placeholder"
    assert interpreter.env[weight].dtype == torch.float32
    torch.testing.assert_close(interpreter.env[node], torch.full((2, 3), 4.0))


@pytest.mark.parametrize("shared_weight", [False, True])
def test_shared_bias_has_independent_scales_and_preserves_float_user(
    linear, shared_weight
):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = linear
            self.second = copy.deepcopy(linear)
            self.second.bias = linear.bias
            if shared_weight:
                self.second.weight = linear.weight

        def forward(self, x, y):
            return self.first(x), self.second(y), self.first.bias

    x = torch.tensor([[-2.0, 3.0]])
    interpreter = _quantize(Model().eval(), (x, x * 8))
    nodes = list(
        interpreter.module.graph.find_nodes(
            op="call_function", target=torch.ops.fused_quant.linear.default
        )
    )
    assert len(nodes) == 2
    for node in nodes:
        _check_bias(interpreter, node, linear.bias)
    assert get_arg(nodes[0], "bias") is not get_arg(nodes[1], "bias")
    scales = [interpreter.env[get_qparams_from_node(n, "bias").scale] for n in nodes]
    torch.testing.assert_close(scales[1], scales[0] * 8, atol=0, rtol=0)
    assert not list(
        interpreter.module.graph.find_nodes(
            op="call_function", target=torch.ops.aten.clone.default
        )
    )
    output = next(n for n in interpreter.module.graph.nodes if n.op == "output")
    torch.testing.assert_close(interpreter.env[output][2], linear.bias, atol=0, rtol=0)


def test_shared_runtime_bias_stays_float(linear):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = linear.weight

        def forward(self, x, y, bias):
            return (
                torch.nn.functional.linear(x, self.weight, bias),
                torch.nn.functional.linear(y, self.weight, bias),
                bias,
            )

    x = torch.tensor([[-2.0, 3.0]])
    interpreter = _quantize(Model().eval(), (x, x * 8, linear.bias.detach()))
    nodes = list(
        interpreter.module.graph.find_nodes(
            op="call_function", target=torch.ops.fused_quant.linear.default
        )
    )
    assert len(nodes) == 2
    for node in nodes:
        bias = get_arg(node, "bias")
        assert bias.op == "placeholder"
        assert interpreter.env[bias].dtype == torch.float32
        assert get_qparams_from_node(node, "bias") is None
    assert get_arg(nodes[0], "bias") is get_arg(nodes[1], "bias")
    assert not list(
        interpreter.module.graph.find_nodes(
            op="call_function", target=torch.ops.aten.clone.default
        )
    )


def test_bias_int32_boundaries(linear):
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.copy_(torch.tensor([127.0, 128.0, -128.0]))
    interpreter = _quantize(linear, (torch.zeros(1, 2),))
    (node,) = interpreter.module.graph.find_nodes(
        op="call_function", target=torch.ops.fused_quant.linear.default
    )
    torch.testing.assert_close(
        interpreter.env[get_arg(node, "bias")],
        torch.tensor([2130706432, 2147483520, -2147483648], dtype=torch.int32),
        atol=0,
        rtol=0,
    )


def test_bias_rounding_is_visible_in_fused_reference(linear):
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[1.0, 0.0]]).expand(3, -1))
        linear.bias.fill_(0.1)
    inputs = (torch.tensor([[0.0, 100000.0], [0.0, -100000.0]]),)
    interpreter = _quantize(linear, inputs)
    torch.testing.assert_close(linear(*inputs), torch.full((2, 3), 0.1), atol=0, rtol=0)
    torch.testing.assert_close(
        interpreter.module(*inputs), torch.zeros(2, 3), atol=0, rtol=0
    )


def test_bias_annotation_preserves_other_quantizers(linear):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = linear

        def forward(self, x):
            return self.linear(x), self.linear.bias

    module = torch.export.export(Model().eval(), (torch.ones(1, 2),)).module()
    (node,) = module.graph.find_nodes(
        op="call_function", target=torch.ops.aten.linear.default
    )
    annotation = QuantizationAnnotation(_annotated=True)
    node.meta[Q_ANNOTATION_KEY] = annotation
    bias = get_arg(node, "bias")
    get_esp_fused_quant_quantizer().annotate(module)
    assert node.meta[Q_ANNOTATION_KEY] is annotation
    assert not annotation.input_qspec_map
    assert get_arg(node, "bias") is bias
    assert not list(
        module.graph.find_nodes(op="call_function", target=torch.ops.aten.clone.default)
    )


@pytest.mark.parametrize("outputs,groups", [(3, 1), (2, 2), (6, 2)])
def test_convolution_int32_bias(outputs, groups):
    model = torch.nn.Conv2d(2, outputs, 1, groups=groups)
    interpreter = _quantize(model, (torch.ones(1, 2, 2, 2),))
    (node,) = interpreter.module.graph.find_nodes(
        op="call_function", target=torch.ops.fused_quant.convolution.default
    )
    _check_bias(interpreter, node, model.bias)
