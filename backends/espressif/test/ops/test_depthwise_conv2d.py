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
from executorch.backends.espressif.lowering import (
    get_esp_fused_quant_recipe,
    get_operator_library_report,
)
from executorch.backends.espressif.scratch import get_scratch_queries
from executorch.backends.espressif.target import EspKernelMode, EspTarget
from executorch.backends.espressif.test.tester import (
    EspTestCase,
    EspTester,
    ramp_tensor,
)
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.devtools.bundled_program.core import BundledProgram
from executorch.exir import memory
from executorch.exir._serialize._program import (
    deserialize_pte_binary,
    serialize_pte_binary,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.scalar_type import ScalarType
from executorch.exir.schema import EValue, Int, IntList, KernelCall
from executorch.export import export

_FQ_CONV = exir_ops.edge.fused_quant.convolution.default
_ESP_DW = exir_ops.edge.esp_nn.quantized_depthwise_conv2d_int8.default
_ESP_CONV = exir_ops.edge.esp_nn.quantized_conv2d_int8.default
_CONV = exir_ops.edge.aten.convolution.default


@pytest.fixture
def s3_queries():
    queries = get_scratch_queries(EspTarget.ESP32S3, EspKernelMode.OPTIMIZED)
    if queries is None:
        pytest.skip("Build with EXECUTORCH_BUILD_ESP_NN_PYBINDS=ON")
    return queries


class EspDepthwiseConv2d(torch.nn.Module):
    def __init__(self, channels=8, multiplier=1, kernel=3, *, supported=True, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            channels, channels * multiplier, kernel, groups=channels, **kwargs
        )
        with torch.no_grad():
            self.conv.weight.copy_(
                torch.arange(self.conv.weight.numel())
                .reshape(self.conv.weight.shape)
                .sin()
                * 0.15
            )
            if self.conv.bias is not None:
                self.conv.bias.copy_(torch.linspace(-0.4, 0.7, channels * multiplier))
        self.ops_before_transforms = {_FQ_CONV: 1}
        self.ops_after_transforms = {
            _FQ_CONV: 0,
            _ESP_DW: int(supported),
            _ESP_CONV: 0,
            _CONV: int(not supported),
        }

    def forward(self, x):
        return self.conv(x)


class EspDepthwiseFloatUsers(EspDepthwiseConv2d):
    def forward(self, x):
        y = self.conv(x)
        return y, torch.sin(y), torch.sin(x), self.conv.bias


class EspDepthwiseChain(EspDepthwiseConv2d):
    def __init__(self):
        super().__init__(padding=1)
        self.ops_before_transforms[_FQ_CONV] = 2
        self.ops_after_transforms[_ESP_DW] = 2

    def forward(self, x):
        return self.conv(self.conv(x))


class EspSharedDepthwise(EspDepthwiseChain):
    def forward(self, x, y):
        return self.conv(x), self.conv(y), self.conv.bias


class EspDepthwiseRuntimeBias(EspDepthwiseConv2d):
    def __init__(self):
        super().__init__(supported=False)

    def forward(self, x, bias):
        return torch.nn.functional.conv2d(x, self.conv.weight, bias, groups=8)


test_cases = {
    "s8_padded": EspTestCase(
        EspDepthwiseConv2d(16, padding=1), (ramp_tensor(-3, 5, (1, 16, 7, 9)),)
    ),
    "s8_unpadded": EspTestCase(
        EspDepthwiseConv2d(16), (ramp_tensor(-3, 5, (1, 16, 7, 9)),)
    ),
    "s16": EspTestCase(EspDepthwiseConv2d(), (ramp_tensor(-3, 5, (1, 8, 7, 9)),)),
    "pad_channels_to_16": EspTestCase(
        EspDepthwiseConv2d(24, padding=1), (ramp_tensor(-3, 5, (1, 24, 7, 9)),)
    ),
    "pad_channels_to_8": EspTestCase(
        EspDepthwiseConv2d(5, padding=1), (ramp_tensor(-3, 5, (1, 5, 7, 9)),)
    ),
    "rgb_vendor_c": EspTestCase(
        EspDepthwiseConv2d(3), (ramp_tensor(-3, 5, (1, 3, 7, 9)),)
    ),
    "multiplier_2": EspTestCase(
        EspDepthwiseConv2d(3, 2, padding=1), (ramp_tensor(-3, 5, (1, 3, 7, 9)),)
    ),
    "multiplier_3": EspTestCase(
        EspDepthwiseConv2d(4, 3, padding=1), (ramp_tensor(-3, 5, (1, 4, 7, 9)),)
    ),
    "multiplier_4": EspTestCase(
        EspDepthwiseConv2d(4, 4, padding=1), (ramp_tensor(-3, 5, (1, 4, 8, 8)),)
    ),
    "multiplier_stride_padding": EspTestCase(
        EspDepthwiseConv2d(4, 4, (3, 2), stride=(2, 1), padding=(1, 0)),
        (ramp_tensor(-3, 5, (1, 4, 7, 9)),),
    ),
    "multiplier_8_3x3": EspTestCase(
        EspDepthwiseConv2d(4, 8, padding=1), (ramp_tensor(-3, 5, (1, 4, 8, 8)),)
    ),
    "multiplier_8_rectangular": EspTestCase(
        EspDepthwiseConv2d(4, 8, (2, 5)), (ramp_tensor(-3, 5, (1, 4, 8, 8)),)
    ),
    "multiplier_12": EspTestCase(
        EspDepthwiseConv2d(4, 12), (ramp_tensor(-3, 5, (1, 4, 8, 8)),)
    ),
    "width_one": EspTestCase(
        EspDepthwiseConv2d(16, kernel=(5, 1), padding=(2, 0)),
        (ramp_tensor(-3, 5, (1, 16, 19, 1)),),
    ),
    "rectangular": EspTestCase(
        EspDepthwiseConv2d(kernel=(2, 5)), (ramp_tensor(-3, 5, (1, 8, 7, 9)),)
    ),
    "stride_padding": EspTestCase(
        EspDepthwiseConv2d(5, kernel=(3, 2), stride=(2, 1), padding=(1, 0)),
        (ramp_tensor(-3, 5, (1, 5, 7, 9)),),
    ),
    "batches_unaligned": EspTestCase(
        EspDepthwiseConv2d(4, 4), (ramp_tensor(-3, 5, (2, 4, 5, 7)),)
    ),
    "batches_s16_unaligned_output": EspTestCase(
        EspDepthwiseConv2d(8), (ramp_tensor(-3, 5, (2, 8, 5, 7)),)
    ),
    "stride_s8_unpadded": EspTestCase(
        EspDepthwiseConv2d(16, stride=2), (ramp_tensor(-3, 5, (1, 16, 9, 9)),)
    ),
    "positive_input": EspTestCase(
        EspDepthwiseConv2d(), (ramp_tensor(1, 5, (1, 8, 7, 9)),)
    ),
    "negative_input": EspTestCase(
        EspDepthwiseConv2d(), (ramp_tensor(-5, -1, (1, 8, 7, 9)),)
    ),
    "no_bias": EspTestCase(
        EspDepthwiseConv2d(bias=False), (ramp_tensor(-3, 5, (1, 8, 7, 9)),)
    ),
    "no_bias_s8": EspTestCase(
        EspDepthwiseConv2d(16, bias=False), (ramp_tensor(-3, 5, (1, 16, 7, 9)),)
    ),
    "float_users": EspTestCase(
        EspDepthwiseFloatUsers(), (ramp_tensor(-3, 5, (1, 8, 7, 9)),)
    ),
    "chain": EspTestCase(EspDepthwiseChain(), (ramp_tensor(-3, 5, (1, 8, 7, 9)),)),
    "shared_bias": EspTestCase(
        EspSharedDepthwise(),
        (ramp_tensor(-3, 5, (1, 8, 7, 9)), ramp_tensor(-24, 40, (1, 8, 7, 9))),
    ),
    "saturation": EspTestCase(
        EspDepthwiseConv2d(),
        (ramp_tensor(-30, 50, (1, 8, 7, 9)),),
        [(ramp_tensor(-3, 5, (1, 8, 7, 9)),)],
    ),
    "dilation_fallback": EspTestCase(
        EspDepthwiseConv2d(dilation=2, supported=False),
        (ramp_tensor(-3, 5, (1, 8, 7, 9)),),
    ),
    "runtime_bias_fallback": EspTestCase(
        EspDepthwiseRuntimeBias(),
        (ramp_tensor(-3, 5, (1, 8, 7, 9)), torch.linspace(-0.4, 0.7, 8)),
    ),
}


@pytest.mark.parametrize("case", test_cases.values(), ids=test_cases.keys())
def test_dialect_depthwise_conv2d(case):
    if (
        case.model.ops_after_transforms[_ESP_DW]
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
    if type(case.model) is EspDepthwiseChain:
        tester.check_node_count({exir_ops.edge.aten.permute_copy.default: 2})
    graph = tester.get_artifact().exported_program().graph
    for node in graph.find_nodes(op="call_function", target=_ESP_DW):
        scratch = get_arg(node, "scratch")
        assert scratch.target == memory.alloc
        assert scratch.meta["val"].dtype == torch.uint8
        weight = get_arg(node, "weight")
        assert weight.op == "placeholder"
        assert weight.meta["val"].shape[0] == 1
        assert weight.meta["val"].shape[-1] == node.meta["val"].shape[-1]
    tester.to_executorch().serialize()
    plan = deserialize_pte_binary(tester.get_artifact()).program.execution_plan[0]
    assert not plan.delegates
    assert all(not op.name.startswith("fused_quant::") for op in plan.operators)
    scratch_index = next(
        i for i, arg in enumerate(_ESP_DW._schema.arguments) if arg.name == "scratch"
    )
    for chain in plan.chains:
        for instruction in chain.instructions:
            call = instruction.instr_args
            if (
                isinstance(call, KernelCall)
                and plan.operators[call.op_index].name
                == "esp_nn::quantized_depthwise_conv2d_int8"
            ):
                scratch = plan.values[call.args[scratch_index]].val
                assert scratch.scalar_type == ScalarType.BYTE
                assert scratch.data_buffer_idx == 0
                assert scratch.allocation_info.memory_id > 0
                assert scratch.allocation_info.memory_offset % 16 == 0
                start = scratch.allocation_info.memory_offset
                end = start + math.prod(scratch.sizes)
                for index in (0, -1):
                    value = plan.values[call.args[index]].val
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
def test_implementation_depthwise_conv2d(case, pytestconfig, s3_queries):
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
        (EspTarget.ESP32S3, EspKernelMode.OPTIMIZED, 728),
        (EspTarget.ESP32P4, EspKernelMode.OPTIMIZED, 0),
        (EspTarget.ESP32S31, EspKernelMode.OPTIMIZED, 0),
        (EspTarget.ESP32, EspKernelMode.OPTIMIZED, 0),
        (EspTarget.ESP32C3, EspKernelMode.OPTIMIZED, 0),
        (EspTarget.ESP32S3, EspKernelMode.ANSI, 0),
    ],
)
def test_depthwise_scratch_selection(target, kernel_mode, expected_bytes):
    queries = get_scratch_queries(target, kernel_mode)
    if queries is None:
        pytest.skip("Build with EXECUTORCH_BUILD_ESP_NN_PYBINDS=ON")
    case = test_cases["multiplier_stride_padding"]
    tester = EspTester(
        case.model, case.example_inputs, target=target, kernel_mode=kernel_mode
    )
    tester.test_dialect(
        case.model.ops_before_transforms, case.model.ops_after_transforms, qtol=1
    )
    (node,) = (
        tester.get_artifact()
        .exported_program()
        .graph.find_nodes(op="call_function", target=_ESP_DW)
    )
    assert get_arg(node, "scratch").meta["val"].numel() == expected_bytes


def test_public_recipe_depthwise(s3_queries):
    case = test_cases["multiplier_3"]
    session = export(
        case.model,
        example_inputs=[case.example_inputs],
        export_recipe=get_esp_fused_quant_recipe(),
    )
    report = get_operator_library_report(session.get_edge_program_manager())
    assert report["esp_nn_ops"] == 1
    assert not report["cpu_ops"]
    assert len(report["layout_ops"]) == 2
    assert session.get_executorch_program_manager().buffer
    with pytest.raises(RuntimeError, match="aten::permute_copy"):
        export(
            case.model,
            example_inputs=[case.example_inputs],
            export_recipe=get_esp_fused_quant_recipe(allow_cpu_fallback=False),
        )

    case = test_cases["dilation_fallback"]
    with pytest.raises(RuntimeError, match="aten::convolution"):
        export(
            case.model,
            example_inputs=[case.example_inputs],
            export_recipe=get_esp_fused_quant_recipe(allow_cpu_fallback=False),
        )


@pytest.mark.parametrize("missing", ["binding", "query"])
def test_depthwise_without_sizing(monkeypatch, missing):
    if missing == "binding":
        monkeypatch.setitem(
            sys.modules, "executorch.backends.espressif._esp_nn_scratch_esp32s3", None
        )
    else:
        queries = get_scratch_queries(EspTarget.ESP32S3, EspKernelMode.OPTIMIZED)
        if queries is None:
            pytest.skip("Build with EXECUTORCH_BUILD_ESP_NN_PYBINDS=ON")
        monkeypatch.delattr(queries, "depthwise_conv2d_scratch_size")
    case = test_cases["s16"]
    EspTester(case.model, case.example_inputs, target=EspTarget.ESP32S3).test_dialect(
        {_FQ_CONV: 1}, {_ESP_DW: 0, _CONV: 1}, atol=0, rtol=0
    ).to_executorch().serialize()


def test_depthwise_falls_back_when_sizing_rejects_bounds(s3_queries):
    with pytest.raises(ValueError, match="scratch query intermediates"):
        s3_queries.depthwise_conv2d_scratch_size(
            input_dims=(7, 7, 4800),
            filter_dims=(7, 7, 4800),
            output_dims=(7, 7, 4800),
            stride=(1, 1),
            padding=(3, 3),
            channel_multiplier=1,
        )
    model = EspDepthwiseConv2d(4800, kernel=7, padding=3, supported=False)
    inputs = (ramp_tensor(-3, 5, (1, 4800, 7, 7)),)
    EspTester(model, inputs, target=EspTarget.ESP32S3).test_dialect(
        model.ops_before_transforms, model.ops_after_transforms, atol=0, rtol=0
    ).to_executorch().serialize()


@pytest.mark.parametrize("extent,supported", [(32768, True), (32769, False)])
@pytest.mark.parametrize("axis", [2, 3])
def test_depthwise_spatial_index_bounds(
    s3_queries, monkeypatch, extent, supported, axis
):
    def unexpected_query(**kwargs):
        pytest.fail("Unsafe spatial indices must fall back before sizing")

    if not supported:
        monkeypatch.setattr(
            s3_queries, "depthwise_conv2d_scratch_size", unexpected_query
        )
    model = EspDepthwiseConv2d(2, kernel=1, supported=supported)
    shape = [1, 2, 1, 1]
    shape[axis] = extent
    EspTester(
        model, (ramp_tensor(-3, 5, shape),), target=EspTarget.ESP32S3
    ).test_dialect(
        model.ops_before_transforms,
        model.ops_after_transforms,
        qtol=int(supported),
        atol=0,
        rtol=0,
    ).to_executorch().serialize()


def test_implementation_depthwise_undersized_scratch(pytestconfig, s3_queries):
    if not os.environ.get("ESPRESSIF_BUILD_DIR"):
        pytest.skip("Set ESPRESSIF_BUILD_DIR to the ESP32-S3 build directory")
    case = test_cases["s8_padded"]
    tester = (
        EspTester(case.model, case.example_inputs)
        .export_fused_quant()
        .to_edge()
        .run_passes()
    )
    module = tester.get_artifact().exported_program().graph_module
    (node,) = module.graph.find_nodes(op="call_function", target=_ESP_DW)
    scratch = get_arg(node, "scratch")
    if scratch.meta["val"].numel() == 0:
        pytest.skip("The selected kernel does not require scratch")
    scratch.args = (((1,), torch.uint8),)
    scratch.meta["val"] = scratch.meta["val"].new_empty((1,))
    module.recompile()
    with pytest.raises(AssertionError):
        tester.test_implementation(pytestconfig.rootpath)
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    log = (
        Path(os.environ["ESPRESSIF_BUILD_DIR"])
        / f"operator-library-{worker}-{tester.target.value}-{tester.kernel_mode.value}"
        / "emulator.log"
    )
    text = log.read_text()
    assert "op_quantized_depthwise_conv2d.cpp" in text
    assert "scratch.nbytes()" in text


def test_depthwise_padding_bound(s3_queries, monkeypatch):
    def unexpected_query(**kwargs):
        pytest.fail("Unsafe spatial indices must fall back before sizing")

    monkeypatch.setattr(s3_queries, "depthwise_conv2d_scratch_size", unexpected_query)
    model = EspDepthwiseConv2d(
        2, kernel=1, stride=(65535, 1), padding=(32769, 0), supported=False
    )
    EspTester(
        model, (ramp_tensor(-3, 5, (1, 2, 1, 1)),), target=EspTarget.ESP32S3
    ).test_dialect(
        model.ops_before_transforms, model.ops_after_transforms, atol=0, rtol=0
    ).to_executorch().serialize()


@pytest.mark.parametrize("axis", [0, 1], ids=["height", "width"])
def test_implementation_depthwise_rejects_out_of_bounds_windows(
    pytestconfig, s3_queries, monkeypatch, axis
):
    if not os.environ.get("ESPRESSIF_BUILD_DIR"):
        pytest.skip("Set ESPRESSIF_BUILD_DIR to the ESP32-S3 build directory")
    serialize_to_schema = BundledProgram.serialize_to_schema

    def invalid_stride(bundle):
        schema = serialize_to_schema(bundle)
        pte = deserialize_pte_binary(schema.program)
        plan = pte.program.execution_plan[0]
        (call,) = (
            instruction.instr_args
            for chain in plan.chains
            for instruction in chain.instructions
            if isinstance(instruction.instr_args, KernelCall)
            and plan.operators[instruction.instr_args.op_index].name
            == "esp_nn::quantized_depthwise_conv2d_int8"
        )
        stride_index = next(
            i for i, arg in enumerate(_ESP_DW._schema.arguments) if arg.name == "stride"
        )
        stride = list(plan.values[call.args[stride_index]].val.items)
        stride[axis] = len(plan.values)
        plan.values.append(EValue(Int(2)))
        call.args[stride_index] = len(plan.values)
        plan.values.append(EValue(IntList(stride)))
        schema.program = bytes(serialize_pte_binary(pte))
        return schema

    monkeypatch.setattr(BundledProgram, "serialize_to_schema", invalid_stride)
    case = test_cases["s8_padded"]
    tester = EspTester(case.model, case.example_inputs)
    with pytest.raises(AssertionError):
        tester.test_implementation(pytestconfig.rootpath)
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    log = (
        Path(os.environ["ESPRESSIF_BUILD_DIR"])
        / f"operator-library-{worker}-{tester.target.value}-{tester.kernel_mode.value}"
        / "emulator.log"
    )
    text = log.read_text()
    assert "op_quantized_depthwise_conv2d.cpp" in text
    assert "weight.size(axis + 1)" in text


def test_implementation_depthwise_zero_scratch(pytestconfig):
    if get_scratch_queries(EspTarget.ESP32S3, EspKernelMode.ANSI) is None:
        pytest.skip("Build with EXECUTORCH_BUILD_ESP_NN_PYBINDS=ON")
    case = test_cases["multiplier_3"]
    tester = EspTester(case.model, case.example_inputs, kernel_mode=EspKernelMode.ANSI)
    tester.test_dialect(
        case.model.ops_before_transforms, case.model.ops_after_transforms, qtol=1
    )
    (node,) = (
        tester.get_artifact()
        .exported_program()
        .graph.find_nodes(op="call_function", target=_ESP_DW)
    )
    assert get_arg(node, "scratch").meta["val"].numel() == 0
    tester.test_implementation(pytestconfig.rootpath, num_inferences=2)
