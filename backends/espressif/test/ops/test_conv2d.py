# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import sys
from pathlib import Path

import pytest
import torch
from executorch.backends.espressif.lowering import get_esp_fused_quant_recipe
from executorch.backends.espressif.passes.lower_to_esp import (
    _QParams,
    _weighted_multipliers,
)
from executorch.backends.espressif.scratch import get_scratch_queries
from executorch.backends.espressif.target import EspKernelMode, EspTarget
from executorch.backends.espressif.test.tester import (
    EspTestCase,
    EspTester,
    ramp_tensor,
)
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.exir import memory
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.scalar_type import ScalarType
from executorch.exir.schema import KernelCall, Tensor
from executorch.export import export

_FQ_CONV = exir_ops.edge.fused_quant.convolution.default
_ESP_CONV = exir_ops.edge.esp_nn.quantized_conv2d_int8.default
_CONV = exir_ops.edge.aten.convolution.default


@pytest.fixture
def s3_queries():
    queries = get_scratch_queries(EspTarget.ESP32S3, EspKernelMode.OPTIMIZED)
    if queries is None:
        pytest.skip("Build with EXECUTORCH_BUILD_ESP_NN_PYBINDS=ON")
    return queries


class EspConv2d(torch.nn.Module):
    def __init__(
        self,
        channels=3,
        outputs=8,
        kernel=3,
        *,
        supported=True,
        transposed=False,
        **kwargs,
    ):
        super().__init__()
        cls = torch.nn.ConvTranspose2d if transposed else torch.nn.Conv2d
        self.conv = cls(channels, outputs, kernel, **kwargs)
        with torch.no_grad():
            self.conv.weight.copy_(
                torch.arange(self.conv.weight.numel())
                .reshape(self.conv.weight.shape)
                .sin()
                * 0.15
            )
            if self.conv.bias is not None:
                self.conv.bias.copy_(torch.linspace(-0.4, 0.7, outputs))
        self.ops_before_transforms = {_FQ_CONV: 1}
        self.ops_after_transforms = {
            _FQ_CONV: 0,
            _ESP_CONV: int(supported),
            _CONV: int(not supported),
        }

    def forward(self, x):
        return self.conv(x)


class EspConvFloatUsers(EspConv2d):
    def forward(self, x):
        y = self.conv(x)
        return y, torch.sin(y), torch.sin(x), self.conv.bias


class EspConvChain(EspConv2d):
    def __init__(self):
        super().__init__(4, 4, padding=1)
        self.ops_before_transforms[_FQ_CONV] = 2
        self.ops_after_transforms[_ESP_CONV] = 2

    def forward(self, x):
        return self.conv(self.conv(x))


class EspSharedConv(EspConvChain):
    def forward(self, x, y):
        return self.conv(x), self.conv(y), self.conv.bias


class EspRuntimeBias(EspConv2d):
    def __init__(self):
        super().__init__(supported=False)

    def forward(self, x, bias):
        return torch.nn.functional.conv2d(x, self.conv.weight, bias)


test_cases = {
    "rgb_im2col": EspTestCase(
        EspConv2d(padding=1), (ramp_tensor(-3, 5, (1, 3, 8, 9)),)
    ),
    "positive_input": EspTestCase(
        EspConv2d(16, 16, padding=1), (ramp_tensor(1, 5, (1, 16, 8, 8)),)
    ),
    "negative_input": EspTestCase(
        EspConv2d(padding=1), (ramp_tensor(-5, -1, (1, 3, 7, 9)),)
    ),
    "pointwise": EspTestCase(
        EspConv2d(16, 32, 1), (ramp_tensor(-3, 5, (1, 16, 8, 8)),)
    ),
    "pointwise_tails": EspTestCase(
        EspConv2d(5, 7, 1), (ramp_tensor(-3, 5, (1, 5, 3, 5)),)
    ),
    "pointwise_panel": EspTestCase(
        EspConv2d(256, 104, 1), (ramp_tensor(-3, 5, (1, 256, 2, 5)),)
    ),
    "3x3_staged": EspTestCase(
        EspConv2d(16, 32, padding=1), (ramp_tensor(-3, 5, (1, 16, 8, 8)),)
    ),
    "3x3_small_output": EspTestCase(
        EspConv2d(16, 32), (ramp_tensor(-3, 5, (1, 16, 3, 3)),)
    ),
    "rectangular": EspTestCase(
        EspConv2d(5, 7, (2, 5)), (ramp_tensor(-3, 5, (1, 5, 8, 8)),)
    ),
    "stride_padding": EspTestCase(
        EspConv2d(4, 7, (3, 2), stride=(2, 1), padding=(1, 0)),
        (ramp_tensor(-3, 5, (1, 4, 7, 9)),),
    ),
    "batches": EspTestCase(EspConv2d(padding=1), (ramp_tensor(-3, 5, (2, 3, 6, 7)),)),
    "batches_staged": EspTestCase(
        EspConv2d(16, 16, padding=1), (ramp_tensor(-3, 5, (2, 16, 5, 7)),)
    ),
    "single_output": EspTestCase(EspConv2d(1, 1), (ramp_tensor(-3, 5, (1, 1, 5, 7)),)),
    "no_bias": EspTestCase(EspConv2d(bias=False), (ramp_tensor(-3, 5, (1, 3, 7, 7)),)),
    "no_bias_staged": EspTestCase(
        EspConv2d(16, 16, bias=False), (ramp_tensor(-3, 5, (1, 16, 5, 7)),)
    ),
    "stride_remainder": EspTestCase(
        EspConv2d(16, 16, stride=3), (ramp_tensor(-3, 5, (1, 16, 8, 8)),)
    ),
    "float_users": EspTestCase(
        EspConvFloatUsers(), (ramp_tensor(-3, 5, (1, 3, 7, 7)),)
    ),
    "chain": EspTestCase(EspConvChain(), (ramp_tensor(-3, 5, (1, 4, 5, 5)),)),
    "shared_bias": EspTestCase(
        EspSharedConv(),
        (ramp_tensor(-3, 5, (1, 4, 5, 5)), ramp_tensor(-24, 40, (1, 4, 5, 5))),
    ),
    "saturation": EspTestCase(
        EspConv2d(),
        (ramp_tensor(-30, 50, (1, 3, 7, 7)),),
        [(ramp_tensor(-3, 5, (1, 3, 7, 7)),)],
    ),
    "groups_fallback": EspTestCase(
        EspConv2d(4, 6, groups=2, supported=False), (ramp_tensor(-3, 5, (1, 4, 5, 7)),)
    ),
    "dilation_fallback": EspTestCase(
        EspConv2d(dilation=2, supported=False), (ramp_tensor(-3, 5, (1, 3, 7, 9)),)
    ),
    "transpose_fallback": EspTestCase(
        EspConv2d(transposed=True, supported=False), (ramp_tensor(-3, 5, (1, 3, 5, 7)),)
    ),
    "runtime_bias_fallback": EspTestCase(
        EspRuntimeBias(),
        (ramp_tensor(-3, 5, (1, 3, 5, 7)), torch.linspace(-0.4, 0.7, 8)),
    ),
}


@pytest.mark.parametrize("case", test_cases.values(), ids=test_cases.keys())
def test_dialect_conv2d(case):
    if (
        case.model.ops_after_transforms[_ESP_CONV]
        and get_scratch_queries(EspTarget.ESP32S3, EspKernelMode.OPTIMIZED) is None
    ):
        pytest.skip("Build with EXECUTORCH_BUILD_ESP_NN_PYBINDS=ON")
    tester = EspTester(case.model, case.example_inputs, case.calibration_samples)
    tester.test_dialect(
        case.model.ops_before_transforms,
        case.model.ops_after_transforms,
        qtol=1,
        atol=0,
        rtol=torch.finfo(torch.float32).eps,
    )
    if type(case.model) is EspConvChain:
        tester.check_node_count({exir_ops.edge.aten.permute_copy.default: 2})
    graph = tester.get_artifact().exported_program().graph
    for node in graph.find_nodes(op="call_function", target=_ESP_CONV):
        scratch = get_arg(node, "scratch")
        assert scratch.target == memory.alloc
        assert scratch.meta["val"].dtype == torch.uint8
        assert scratch.meta["val"].dim() == 1
        assert scratch.meta["val"].numel() > 0
        assert get_arg(node, "weight").op == "placeholder"
    tester.to_executorch().serialize()
    program = deserialize_pte_binary(tester.get_artifact()).program
    plan = program.execution_plan[0]
    assert not plan.delegates
    assert all(not op.name.startswith("fused_quant::") for op in plan.operators)
    for chain in plan.chains:
        for instruction in chain.instructions:
            call = instruction.instr_args
            if (
                not isinstance(call, KernelCall)
                or plan.operators[call.op_index].name != "esp_nn::quantized_conv2d_int8"
            ):
                continue
            scratch_index = next(
                i
                for i, arg in enumerate(_ESP_CONV._schema.arguments)
                if arg.name == "scratch"
            )
            scratch = plan.values[call.args[scratch_index]].val
            assert isinstance(scratch, Tensor)
            assert scratch.scalar_type == ScalarType.BYTE
            assert scratch.data_buffer_idx == 0
            assert scratch.allocation_info.memory_id > 0
            assert scratch.allocation_info.memory_offset % 16 == 0
            start = scratch.allocation_info.memory_offset
            end = start + math.prod(scratch.sizes)
            for index in (0, -1):
                value = plan.values[call.args[index]].val
                assert isinstance(value, Tensor)
                other = value.allocation_info
                if (
                    other is not None
                    and other.memory_id == scratch.allocation_info.memory_id
                ):
                    assert (
                        other.memory_offset >= end
                        or other.memory_offset + math.prod(value.sizes) <= start
                    )


@pytest.mark.parametrize("case", test_cases.values(), ids=test_cases.keys())
def test_implementation_conv2d(case, pytestconfig, s3_queries):
    EspTester(case.model, case.example_inputs, case.calibration_samples).test_dialect(
        case.model.ops_before_transforms,
        case.model.ops_after_transforms,
        qtol=1,
        atol=0,
        rtol=torch.finfo(torch.float32).eps,
    ).test_implementation(
        pytestconfig.rootpath, rtol=torch.finfo(torch.float32).eps, num_inferences=2
    )


@pytest.mark.parametrize(
    "target,kernel_mode,expected_bytes",
    [
        (EspTarget.ESP32S3, EspKernelMode.OPTIMIZED, 22240),
        (EspTarget.ESP32P4, EspKernelMode.OPTIMIZED, 2624),
        (EspTarget.ESP32S31, EspKernelMode.OPTIMIZED, 2624),
        (EspTarget.ESP32, EspKernelMode.OPTIMIZED, 0),
        (EspTarget.ESP32C3, EspKernelMode.OPTIMIZED, 0),
        (EspTarget.ESP32S3, EspKernelMode.ANSI, 0),
    ],
)
def test_public_recipe_conv2d(target, kernel_mode, expected_bytes):
    if get_scratch_queries(target, kernel_mode) is None:
        pytest.skip("Build with EXECUTORCH_BUILD_ESP_NN_PYBINDS=ON")
    case = test_cases["3x3_staged"]
    session = export(
        case.model,
        example_inputs=[case.example_inputs],
        export_recipe=get_esp_fused_quant_recipe(
            allow_cpu_fallback=False, target=target, kernel_mode=kernel_mode
        ),
    )
    plan = deserialize_pte_binary(
        session.get_executorch_program_manager().buffer
    ).program.execution_plan[0]
    calls = [
        instruction.instr_args
        for chain in plan.chains
        for instruction in chain.instructions
        if isinstance(instruction.instr_args, KernelCall)
        and plan.operators[instruction.instr_args.op_index].name
        == "esp_nn::quantized_conv2d_int8"
    ]
    (call,) = calls
    scratch_index = next(
        i for i, arg in enumerate(_ESP_CONV._schema.arguments) if arg.name == "scratch"
    )
    assert plan.values[call.args[scratch_index]].val.sizes == [expected_bytes]


@pytest.mark.parametrize(
    "target,expected_bytes", [(None, 2624), (EspTarget.ESP32S3, 22240)]
)
def test_tester_export_target_defaults_to_environment(
    monkeypatch, target, expected_bytes
):
    monkeypatch.setenv("ESPRESSIF_TARGET", "esp32p4")
    selected_target = target if target is not None else EspTarget.ESP32P4
    if get_scratch_queries(selected_target, EspKernelMode.OPTIMIZED) is None:
        pytest.skip("Build with EXECUTORCH_BUILD_ESP_NN_PYBINDS=ON")
    case = test_cases["3x3_staged"]
    tester = (
        EspTester(case.model, case.example_inputs, target=target)
        .export_fused_quant()
        .to_edge()
        .run_passes()
    )
    (node,) = (
        tester.get_artifact()
        .exported_program()
        .graph.find_nodes(op="call_function", target=_ESP_CONV)
    )
    assert get_arg(node, "scratch").meta["val"].numel() == expected_bytes


@pytest.mark.parametrize("missing", ["binding", "query"])
def test_convolution_falls_back_without_sizing(monkeypatch, missing):
    if missing == "binding":
        monkeypatch.setitem(
            sys.modules, "executorch.backends.espressif._esp_nn_scratch_esp32s3", None
        )
    else:
        queries = get_scratch_queries(EspTarget.ESP32S3, EspKernelMode.OPTIMIZED)
        if queries is None:
            pytest.skip("Build with EXECUTORCH_BUILD_ESP_NN_PYBINDS=ON")
        monkeypatch.delattr(queries, "conv2d_scratch_size")
    EspTester(torch.nn.Linear(2, 3), (torch.ones(1, 2),)).test_dialect(
        {exir_ops.edge.fused_quant.linear.default: 1},
        {exir_ops.edge.esp_nn.quantized_linear_int8.default: 1},
        qtol=1,
    ).to_executorch()
    fallback = test_cases["groups_fallback"]
    EspTester(fallback.model, fallback.example_inputs).test_dialect(
        fallback.model.ops_before_transforms,
        fallback.model.ops_after_transforms,
        atol=0,
        rtol=0,
    ).to_executorch()
    case = test_cases["rgb_im2col"]
    EspTester(case.model, case.example_inputs, target=EspTarget.ESP32S3).test_dialect(
        {_FQ_CONV: 1}, {_ESP_CONV: 0, _CONV: 1}, atol=0, rtol=0
    ).to_executorch().serialize()
    with pytest.raises(RuntimeError, match="aten::convolution"):
        export(
            case.model,
            example_inputs=[case.example_inputs],
            export_recipe=get_esp_fused_quant_recipe(allow_cpu_fallback=False),
        )


def test_convolution_falls_back_when_sizing_rejects_bounds(s3_queries):
    with pytest.raises(ValueError, match="scratch query intermediates"):
        s3_queries.conv2d_scratch_size(
            input_dims=(1, 1, 32768),
            filter_dims=(1, 1, 32768),
            output_dims=(1, 1, 2),
            stride=(1, 1),
            padding=(0, 0),
        )
    model = EspConv2d(32768, 2, 1, supported=False)
    inputs = (ramp_tensor(-3, 5, (1, 32768, 1, 1)),)
    EspTester(model, inputs, target=EspTarget.ESP32S3).test_dialect(
        model.ops_before_transforms, model.ops_after_transforms, atol=0, rtol=0
    ).to_executorch().serialize()
    with pytest.raises(RuntimeError, match="aten::convolution"):
        export(
            model,
            example_inputs=[inputs],
            export_recipe=get_esp_fused_quant_recipe(allow_cpu_fallback=False),
        )


def test_implementation_conv2d_undersized_scratch(pytestconfig, s3_queries):
    if not os.environ.get("ESPRESSIF_BUILD_DIR"):
        pytest.skip("Set ESPRESSIF_BUILD_DIR to the ESP32-S3 build directory")
    case = test_cases["rgb_im2col"]
    tester = (
        EspTester(case.model, case.example_inputs)
        .export_fused_quant()
        .to_edge()
        .run_passes()
    )
    module = tester.get_artifact().exported_program().graph_module
    (node,) = module.graph.find_nodes(op="call_function", target=_ESP_CONV)
    scratch = get_arg(node, "scratch")
    scratch.args = (((1,), torch.uint8),)
    scratch.meta["val"] = scratch.meta["val"].new_empty((1,))
    module.recompile()
    with pytest.raises(AssertionError):
        tester.test_implementation(
            pytestconfig.rootpath, rtol=torch.finfo(torch.float32).eps, num_inferences=2
        )
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    log = (
        Path(os.environ["ESPRESSIF_BUILD_DIR"])
        / f"operator-library-{worker}-{tester.target.value}-{tester.kernel_mode.value}"
        / "emulator.log"
    )
    text = log.read_text()
    assert "op_quantized_conv2d.cpp" in text
    assert "scratch.nbytes()" in text


def test_long_reduction_rejects_float32_overflow():
    weight = torch.zeros((1, 257 * 257 * 128), dtype=torch.int8)
    weight[0, 0] = 1
    scales = torch.tensor([[1e20]], dtype=torch.float32)
    assert torch.isinf(torch.tensor(127 * 1e20, dtype=torch.float32) * scales[0, 0])
    with pytest.raises(ValueError, match="float32 reference may overflow"):
        _weighted_multipliers(
            _QParams(1e20, 0), _QParams(1e37, 0), weight, scales, None
        )


def test_implementation_conv2d_zero_scratch(pytestconfig):
    if get_scratch_queries(EspTarget.ESP32S3, EspKernelMode.ANSI) is None:
        pytest.skip("Build with EXECUTORCH_BUILD_ESP_NN_PYBINDS=ON")
    case = test_cases["rgb_im2col"]
    tester = EspTester(case.model, case.example_inputs, kernel_mode=EspKernelMode.ANSI)
    tester.test_dialect(
        case.model.ops_before_transforms, case.model.ops_after_transforms, qtol=1
    )
    (node,) = (
        tester.get_artifact()
        .exported_program()
        .graph.find_nodes(op="call_function", target=_ESP_CONV)
    )
    assert get_arg(node, "scratch").meta["val"].numel() == 0
    tester.test_implementation(pytestconfig.rootpath, num_inferences=2)


def test_implementation_conv2d_without_sizing(pytestconfig, monkeypatch):
    monkeypatch.setattr(
        "executorch.backends.espressif.passes.lower_to_esp.get_scratch_queries",
        lambda *_: None,
    )
    case = test_cases["rgb_im2col"]
    tester = EspTester(case.model, case.example_inputs)
    tester.test_dialect({_FQ_CONV: 1}, {_ESP_CONV: 0, _CONV: 1}, atol=0, rtol=0)
    tester.test_implementation(
        pytestconfig.rootpath, rtol=torch.finfo(torch.float32).eps, num_inferences=2
    )
