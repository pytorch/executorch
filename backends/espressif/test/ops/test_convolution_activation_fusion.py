# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from functools import partial

import pytest
import torch
from executorch.backends.espressif.lowering import (
    get_esp_fused_quant_recipe,
    get_operator_library_report,
)
from executorch.backends.espressif.passes.fold_activations import FoldActivations
from executorch.backends.espressif.quantizer import get_esp_fused_quant_quantizer
from executorch.backends.espressif.scratch import get_scratch_queries
from executorch.backends.espressif.target import EspKernelMode, EspTarget
from executorch.backends.espressif.test.ops.test_conv2d import EspConv2d, EspRuntimeBias
from executorch.backends.espressif.test.ops.test_depthwise_conv2d import (
    EspDepthwiseConv2d,
    EspDepthwiseRuntimeBias,
)
from executorch.backends.espressif.test.tester import (
    EspTestCase,
    EspTester,
    ramp_tensor,
)
from executorch.backends.transforms.fused_quant.graph_utils import get_qparams_from_node
from executorch.backends.transforms.fused_quant.optimization_passes.hoist_activations import (
    HoistActivations,
)
from executorch.backends.transforms.fused_quant.optimization_passes.to_channels_last import (
    ToChannelsLast,
)
from executorch.backends.transforms.permute_pass_utils import get_arg, set_arg
from executorch.backends.transforms.utils import get_constant
from executorch.exir import memory
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from executorch.exir.pass_manager import ExportedProgramPassManager
from executorch.exir.scalar_type import ScalarType
from executorch.exir.schema import KernelCall
from executorch.export import export
from executorch.frontends.fused_quant.quantizer.quantizer import FusedQuantQuantizer

_FQ_CONV = exir_ops.edge.fused_quant.convolution.default
_FQ_NHWC = exir_ops.edge.fused_quant.convolution_channels_last.default
_ESP_CONV = exir_ops.edge.esp_nn.quantized_conv2d_int8.default
_ESP_DW = exir_ops.edge.esp_nn.quantized_depthwise_conv2d_int8.default
_ESP_LUT = exir_ops.edge.esp_nn.quantized_lut_int8.default
_ACTIVATIONS = {
    "relu": (torch.nn.ReLU, (-7, 127)),
    "relu6": (torch.nn.ReLU6, (-7, 41)),
    "hardtanh": (torch.nn.Hardtanh, (-15, 1)),
    "hardtanh_asymmetric": (
        partial(torch.nn.Hardtanh, min_val=-2.0625, max_val=3.1875),
        (-23, 19),
    ),
    "hardtanh_positive": (partial(torch.nn.Hardtanh, min_val=1, max_val=4), (1, 25)),
    "hardtanh_negative": (
        partial(torch.nn.Hardtanh, min_val=-5, max_val=-2),
        (-47, -23),
    ),
}
_PRODUCERS = {"conv2d": EspConv2d, "depthwise": EspDepthwiseConv2d}


@pytest.fixture
def s3_queries():
    if get_scratch_queries(EspTarget.ESP32S3, EspKernelMode.OPTIMIZED) is None:
        pytest.skip("Build with EXECUTORCH_BUILD_ESP_NN_PYBINDS=ON")


class ConvolutionActivation(torch.nn.Module):
    def __init__(self, producer, activation, *, fork=False):
        super().__init__()
        self.producer = producer
        self.activation = _ACTIVATIONS[activation][0]()
        self.fused_activation = (
            exir_ops.edge.fused_quant.relu.default
            if activation == "relu"
            else exir_ops.edge.fused_quant.hard_tanh.default
        )
        self.fork = fork
        self.ops_before_transforms = {
            **producer.ops_before_transforms,
            self.fused_activation: 1,
        }
        self.ops_after_transforms = {
            **producer.ops_after_transforms,
            _FQ_NHWC: 0,
            self.fused_activation: 0,
            _ESP_LUT: int(fork),
        }

    def forward(self, *inputs):
        result = self.producer(*inputs)
        activated = self.activation(result)
        return (activated, result, torch.sin(result)) if self.fork else activated


test_cases = {
    f"{kind}_{activation}{'_no_bias' if not bias else ''}": EspTestCase(
        ConvolutionActivation(producer(channels=8, bias=bias), activation),
        (ramp_tensor(-30, 50, (1, 8, 7, 9)),),
    )
    for kind, producer in _PRODUCERS.items()
    for activation in _ACTIVATIONS
    for bias in (True, False)
}
test_cases.update(
    {
        f"{kind}_{activation}_fork": EspTestCase(
            ConvolutionActivation(producer(channels=8), activation, fork=True),
            (ramp_tensor(-3, 5, (1, 8, 7, 9)),),
        )
        for kind, producer in _PRODUCERS.items()
        for activation in ("relu", "hardtanh")
    }
)
test_cases.update(
    {
        f"{kind}_dilation_fallback": EspTestCase(
            ConvolutionActivation(
                producer(channels=8, dilation=2, supported=False), "relu6"
            ),
            (ramp_tensor(-3, 5, (1, 8, 7, 9)),),
        )
        for kind, producer in _PRODUCERS.items()
    }
)
test_cases.update(
    {
        f"{kind}_saturation": EspTestCase(
            ConvolutionActivation(producer(channels=8), "relu6"),
            (ramp_tensor(-30, 50, (1, 8, 7, 9)),),
            [(ramp_tensor(-3, 5, (1, 8, 7, 9)),)],
        )
        for kind, producer in _PRODUCERS.items()
    }
)
test_cases.update(
    {
        "conv2d_pointwise": EspTestCase(
            ConvolutionActivation(EspConv2d(16, 32, 1), "hardtanh"),
            (ramp_tensor(-30, 50, (1, 16, 3, 5)),),
        ),
        "conv2d_staged": EspTestCase(
            ConvolutionActivation(EspConv2d(16, 32, padding=1), "relu6"),
            (ramp_tensor(-30, 50, (1, 16, 5, 7)),),
        ),
        "conv2d_rgb": EspTestCase(
            ConvolutionActivation(EspConv2d(padding=1), "hardtanh_asymmetric"),
            (ramp_tensor(-30, 50, (2, 3, 5, 7)),),
        ),
        "depthwise_s8_padded": EspTestCase(
            ConvolutionActivation(EspDepthwiseConv2d(16, padding=1), "relu6"),
            (ramp_tensor(-30, 50, (1, 16, 5, 7)),),
        ),
        "depthwise_vendor_c": EspTestCase(
            ConvolutionActivation(EspDepthwiseConv2d(3), "hardtanh_asymmetric"),
            (ramp_tensor(-30, 50, (1, 3, 5, 7)),),
        ),
        "depthwise_multiplier_2": EspTestCase(
            ConvolutionActivation(EspDepthwiseConv2d(4, 2), "hardtanh"),
            (ramp_tensor(-30, 50, (2, 4, 5, 7)),),
        ),
        "depthwise_multiplier_8": EspTestCase(
            ConvolutionActivation(EspDepthwiseConv2d(4, 8), "relu6"),
            (ramp_tensor(-30, 50, (1, 4, 5, 7)),),
        ),
        "groups_fallback": EspTestCase(
            ConvolutionActivation(
                EspConv2d(4, 6, groups=2, supported=False), "hardtanh"
            ),
            (ramp_tensor(-3, 5, (1, 4, 5, 7)),),
        ),
        "transpose_fallback": EspTestCase(
            ConvolutionActivation(
                EspConv2d(transposed=True, supported=False), "hardtanh"
            ),
            (ramp_tensor(-3, 5, (1, 3, 5, 7)),),
        ),
        "conv2d_runtime_bias_fallback": EspTestCase(
            ConvolutionActivation(EspRuntimeBias(), "relu6"),
            (ramp_tensor(-3, 5, (1, 3, 5, 7)), torch.linspace(-0.4, 0.7, 8)),
        ),
        "depthwise_runtime_bias_fallback": EspTestCase(
            ConvolutionActivation(EspDepthwiseRuntimeBias(), "hardtanh"),
            (ramp_tensor(-3, 5, (1, 8, 5, 7)), torch.linspace(-0.4, 0.7, 8)),
        ),
    }
)


@pytest.mark.parametrize("case", test_cases.values(), ids=test_cases.keys())
def test_dialect_convolution_activation(case, s3_queries):
    tester = EspTester(case.model, case.example_inputs, case.calibration_samples)
    tester.test_dialect(
        case.model.ops_before_transforms,
        case.model.ops_after_transforms,
        qtol=2 if case.model.fork else 1,
        atol=0,
        rtol=torch.finfo(torch.float32).eps,
    )
    tester.to_executorch().serialize()
    plan = deserialize_pte_binary(tester.get_artifact()).program.execution_plan[0]
    assert not plan.delegates
    assert not any(op.name.startswith("fused_quant::") for op in plan.operators)
    for chain in plan.chains:
        for instruction in chain.instructions:
            call = instruction.instr_args
            if not isinstance(call, KernelCall):
                continue
            name = plan.operators[call.op_index].name
            if name not in {
                "esp_nn::quantized_conv2d_int8",
                "esp_nn::quantized_depthwise_conv2d_int8",
            }:
                continue
            target = _ESP_CONV if name == "esp_nn::quantized_conv2d_int8" else _ESP_DW
            scratch_index = next(
                i
                for i, arg in enumerate(target._schema.arguments)
                if arg.name == "scratch"
            )
            scratch = plan.values[call.args[scratch_index]].val
            assert scratch.scalar_type == ScalarType.BYTE
            assert scratch.data_buffer_idx == 0
            assert scratch.allocation_info.memory_id > 0
            assert scratch.allocation_info.memory_offset % 16 == 0


@pytest.mark.parametrize("case", test_cases.values(), ids=test_cases.keys())
def test_implementation_convolution_activation(case, pytestconfig, s3_queries):
    EspTester(
        case.model, case.example_inputs, case.calibration_samples
    ).test_implementation(
        pytestconfig.rootpath, rtol=torch.finfo(torch.float32).eps, num_inferences=2
    )


@pytest.mark.parametrize("kind", _PRODUCERS)
@pytest.mark.parametrize("activation", ["relu", "relu6", "hardtanh"])
@pytest.mark.parametrize("fork", [False, True])
def test_hoisting_preserves_convolution_quantization_boundary(kind, activation, fork):
    model = ConvolutionActivation(_PRODUCERS[kind](channels=8), activation, fork=fork)
    inputs = (ramp_tensor(-3, 5, (1, 8, 7, 9)),)
    tester = EspTester(model, inputs).export_fused_quant().to_edge()
    source = tester.get_artifact().exported_program()
    expected = source.module()(*inputs)
    result = ExportedProgramPassManager(
        [ToChannelsLast(), HoistActivations(), ExportPass()]
    )(copy.deepcopy(source))
    program = result.exported_program
    (producer,) = program.graph.find_nodes(op="call_function", target=_FQ_NHWC)
    (activation_node,) = program.graph.find_nodes(
        op="call_function", target=model.fused_activation
    )
    assert (get_qparams_from_node(producer, "out") is not None) == fork
    assert (get_qparams_from_node(activation_node, "inp") is not None) == fork
    assert (get_arg(activation_node, "inp") is producer) == (not fork)
    assert FoldActivations()(program).modified == (not fork)
    torch.testing.assert_close(program.module()(*inputs), expected, atol=0, rtol=0)


@pytest.mark.parametrize("kind", _PRODUCERS)
def test_separately_quantized_convolution_activation_keeps_rounding(kind, s3_queries):
    model = ConvolutionActivation(_PRODUCERS[kind](channels=8), "hardtanh")
    tester = EspTester(model, (ramp_tensor(-3, 5, (1, 8, 7, 9)),))
    tester.export_fused_quant(
        FusedQuantQuantizer(get_esp_fused_quant_quantizer().child_quantizers)
    ).to_edge()
    program = ExportedProgramPassManager(
        [ToChannelsLast(), HoistActivations(), ExportPass()]
    )(copy.deepcopy(tester.get_artifact().exported_program())).exported_program
    assert not FoldActivations()(program).modified
    target = _ESP_CONV if kind == "conv2d" else _ESP_DW
    tester.run_passes().check_node_count({target: 1, _ESP_LUT: 1})


def _convolution_clamp_tester(kind, activation):
    model = ConvolutionActivation(_PRODUCERS[kind](channels=8), activation)
    tester = EspTester(model, (ramp_tensor(-80, 80, (1, 8, 7, 9)),))
    tester.export_fused_quant().to_edge()
    program = tester.get_artifact().exported_program()
    (node,) = program.graph.find_nodes(
        op="call_function", target=model.fused_activation
    )
    params = get_qparams_from_node(node, "out")
    get_constant(program, params.scale).fill_(0.125)
    get_constant(program, params.zero_point).fill_(-7)
    (output,) = program.graph.find_nodes(op="output")
    dequant = output.args[0][0]
    set_arg(dequant, "scale", 0.125)
    set_arg(dequant, "zero_point", -7)
    program.graph_module.recompile()
    return tester


@pytest.mark.parametrize("kind", _PRODUCERS)
@pytest.mark.parametrize("activation", _ACTIVATIONS)
def test_convolution_uses_final_qparams_and_clamp(kind, activation, s3_queries):
    tester = _convolution_clamp_tester(kind, activation)
    program = tester.get_artifact().exported_program()
    expected = program.module()(*tester.example_inputs)
    (source_conv,) = program.graph.find_nodes(op="call_function", target=_FQ_CONV)
    bias = get_constant(program, get_arg(source_conv, "bias")).clone()
    target = _ESP_CONV if kind == "conv2d" else _ESP_DW
    tester.run_passes().check_node_count({target: 1, _ESP_LUT: 0})
    lowered = tester.get_artifact().exported_program()
    (node,) = lowered.graph.find_nodes(op="call_function", target=target)
    assert get_arg(node, "output_offset") == -7
    assert (
        get_arg(node, "activation_min"),
        get_arg(node, "activation_max"),
    ) == _ACTIVATIONS[activation][1]
    assert get_arg(node, "scratch").target == memory.alloc
    torch.testing.assert_close(
        get_constant(lowered, get_arg(node, "bias")), bias, atol=0, rtol=0
    )
    torch.testing.assert_close(
        lowered.module()(*tester.example_inputs), expected, atol=0.125, rtol=0
    )


@pytest.mark.parametrize("kind", _PRODUCERS)
@pytest.mark.parametrize("activation", _ACTIVATIONS)
def test_implementation_convolution_clamp(kind, activation, pytestconfig, s3_queries):
    tester = _convolution_clamp_tester(kind, activation)
    tester.run_passes()
    tester.test_implementation(pytestconfig.rootpath, num_inferences=2)


@pytest.mark.parametrize("kind", _PRODUCERS)
def test_strict_convolution_activation_recipe(kind, s3_queries):
    case = test_cases[f"{kind}_relu6"]
    session = export(
        case.model,
        example_inputs=[case.example_inputs],
        export_recipe=get_esp_fused_quant_recipe(allow_cpu_fallback=False),
    )
    report = get_operator_library_report(session.get_edge_program_manager())
    assert report["esp_nn_ops"] == 1
    assert report["cpu_ops"] == []
    assert len(report["layout_ops"]) == 2
    assert session.get_executorch_program_manager().buffer


@pytest.mark.parametrize("kind", _PRODUCERS)
def test_convolution_activation_without_scratch_binding_falls_back(kind, monkeypatch):
    case = test_cases[f"{kind}_hardtanh"]
    monkeypatch.setattr(
        "executorch.backends.espressif.passes.lower_to_esp.get_scratch_queries",
        lambda *_: None,
    )
    tester = EspTester(case.model, case.example_inputs)
    tester.test_dialect(
        case.model.ops_before_transforms,
        {
            _ESP_CONV: 0,
            _ESP_DW: 0,
            _ESP_LUT: 0,
            exir_ops.edge.aten.convolution.default: 1,
        },
        qtol=1,
        atol=0,
        rtol=torch.finfo(torch.float32).eps,
    )
