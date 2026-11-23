# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from executorch.backends.espressif.lowering import get_esp_fused_quant_recipe
from executorch.backends.espressif.passes.lower_to_esp import _softmax_parameters
from executorch.backends.espressif.scratch import get_scratch_queries
from executorch.backends.espressif.target import EspKernelMode, EspTarget
from executorch.backends.espressif.test.tester import EspTestCase, EspTester
from executorch.backends.test.harness.stages import StageType
from executorch.backends.transforms.fused_quant.graph_utils import (
    get_qparams_from_node,
    get_scale,
    get_zero_point,
)
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
)
from executorch.exir import memory
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.schema import KernelCall, ScalarType, Tensor
from executorch.export import export
from executorch.frontends.fused_quant.quantizer.quantizer import (
    FusedQuantQuantizer,
    OpQuantizer,
)
from torchao.quantization.pt2e.quantizer import FixedQParamsQuantizationSpec

_SOFTMAX = exir_ops.edge.aten._softmax.default
_FQ_SOFTMAX = exir_ops.edge.fused_quant._softmax.default
_ESP_SOFTMAX = exir_ops.edge.esp_nn.quantized_softmax_int8.default
_OUTPUT_SCALE = 1 / 256


class EspSoftmax(torch.nn.Module):
    def __init__(self, dim=-1, *, supported=True):
        super().__init__()
        self.dim = dim
        self.ops_before_transforms = {_FQ_SOFTMAX: 1}
        self.ops_after_transforms = {
            _FQ_SOFTMAX: 0,
            _SOFTMAX: 0 if supported else 1,
            _ESP_SOFTMAX: 1 if supported else 0,
        }

    def forward(self, x):
        return torch.softmax(x, dim=self.dim)


class EspSoftmaxWithFloatUsers(EspSoftmax):
    def forward(self, x):
        result = super().forward(x)
        return result, torch.sin(result), torch.sin(x)


def _input(shape):
    values = torch.arange(math.prod(shape)).reshape(shape)
    return (values * 0.7).sin() * 5 + values.remainder(5) * 0.2


test_cases = {
    "vector": EspTestCase(EspSoftmax(), (_input((10,)),)),
    "rows": EspTestCase(EspSoftmax(), (_input((3, 10)),)),
    "aligned_rows": EspTestCase(EspSoftmax(), (_input((3, 64)),)),
    "unaligned_rows": EspTestCase(EspSoftmax(), (_input((3, 36)),)),
    "rank_3": EspTestCase(EspSoftmax(), (_input((2, 3, 17)),)),
    "rank_4": EspTestCase(EspSoftmax(), (_input((2, 3, 4, 5)),)),
    "positive_dim": EspTestCase(EspSoftmax(dim=1), (_input((3, 10)),)),
    "single_element": EspTestCase(EspSoftmax(), (_input((3, 1)),)),
    "uniform": EspTestCase(EspSoftmax(), (torch.zeros(2, 8),)),
    "positive": EspTestCase(EspSoftmax(), (_input((2, 10)) + 10,)),
    "negative": EspTestCase(EspSoftmax(), (_input((2, 10)) - 10,)),
    "peaky": EspTestCase(EspSoftmax(), (torch.linspace(-30, 30, 20).reshape(1, 20),)),
    "float_users": EspTestCase(EspSoftmaxWithFloatUsers(), (_input((3, 10)),)),
    "width_limit": EspTestCase(EspSoftmax(), (torch.zeros(1, 255),)),
    "width_fallback": EspTestCase(EspSoftmax(supported=False), (torch.zeros(1, 256),)),
    "wide_uniform_fallback": EspTestCase(
        EspSoftmax(supported=False), (torch.zeros(1, 4095),)
    ),
    "wide_first_axis_fallback": EspTestCase(
        EspSoftmax(dim=0, supported=False), (torch.zeros(4095, 1),)
    ),
    "first_axis_fallback": EspTestCase(
        EspSoftmax(dim=0, supported=False), (_input((3, 10)),)
    ),
    "middle_axis_fallback": EspTestCase(
        EspSoftmax(dim=1, supported=False), (_input((2, 3, 4)),)
    ),
}


@pytest.fixture(params=test_cases.values(), ids=test_cases.keys())
def softmax_case(request):
    case = request.param
    tester = EspTester(case.model, case.example_inputs)
    if (
        case.model.ops_after_transforms[_ESP_SOFTMAX]
        and get_scratch_queries(tester.target, tester.kernel_mode) is None
    ):
        pytest.skip("Build with EXECUTORCH_BUILD_ESP_NN_PYBINDS=ON")
    tester.test_dialect(
        case.model.ops_before_transforms,
        case.model.ops_after_transforms,
        atol=_OUTPUT_SCALE if case.model.ops_after_transforms[_ESP_SOFTMAX] else 0,
        rtol=torch.finfo(torch.float32).eps,
    )
    return case, tester


def test_dialect_softmax(softmax_case):
    case, tester = softmax_case
    reference = tester.get_artifact(StageType.TO_EDGE).exported_program()
    (node,) = reference.graph.find_nodes(op="call_function", target=_FQ_SOFTMAX)
    params = get_qparams_from_node(node, "out")
    if case.model.ops_after_transforms[_ESP_SOFTMAX]:
        assert get_scale(reference, params) == _OUTPUT_SCALE
        assert get_zero_point(reference, params) == -128
    for node in (
        tester.get_artifact()
        .exported_program()
        .graph.find_nodes(op="call_function", target=_ESP_SOFTMAX)
    ):
        scratch = get_arg(node, "scratch")
        assert scratch.target == memory.alloc
        assert scratch.meta["val"].dtype == torch.uint8
        assert get_arg(node, "input").meta["val"].is_contiguous()
    tester.to_executorch().serialize()
    plan = deserialize_pte_binary(tester.get_artifact()).program.execution_plan[0]
    assert not plan.delegates
    operators = {(op.name, op.overload) for op in plan.operators}
    assert (("esp_nn::quantized_softmax_int8", "out") in operators) == bool(
        case.model.ops_after_transforms[_ESP_SOFTMAX]
    )


def test_implementation_softmax(softmax_case, pytestconfig):
    case, tester = softmax_case
    tester.test_implementation(
        pytestconfig.rootpath,
        atol=_OUTPUT_SCALE if case.model.ops_after_transforms[_ESP_SOFTMAX] else 0,
        rtol=torch.finfo(torch.float32).eps,
        num_inferences=2,
    )


@pytest.mark.parametrize(
    "shape,expected_probability",
    [((2, 8), 1 / 8), ((1, 255), _OUTPUT_SCALE)],
)
def test_softmax_uniform_output_is_exact(shape, expected_probability):
    model = EspSoftmax()
    inputs = (torch.zeros(shape),)
    tester = EspTester(model, inputs)
    if get_scratch_queries(tester.target, tester.kernel_mode) is None:
        pytest.skip("Build with EXECUTORCH_BUILD_ESP_NN_PYBINDS=ON")
    tester.test_dialect(
        model.ops_before_transforms, model.ops_after_transforms, atol=0, rtol=0
    )
    actual = tester.get_artifact().exported_program().module()(*inputs)
    torch.testing.assert_close(
        actual, torch.full_like(actual, expected_probability), atol=0, rtol=0
    )


@pytest.mark.parametrize("shape", [(2, 8), (1, 255)])
def test_implementation_softmax_uniform_is_exact(shape, pytestconfig):
    model = EspSoftmax()
    tester = EspTester(model, (torch.zeros(shape),))
    if get_scratch_queries(tester.target, tester.kernel_mode) is None:
        pytest.skip("Build with EXECUTORCH_BUILD_ESP_NN_PYBINDS=ON")
    tester.test_dialect(
        model.ops_before_transforms, model.ops_after_transforms, atol=0, rtol=0
    )
    tester.test_implementation(pytestconfig.rootpath, atol=0, rtol=0, num_inferences=2)


@pytest.mark.parametrize(
    "shape,dim",
    [((1, 4095), -1), ((4095, 1), 0)],
)
def test_softmax_wide_fallback_preserves_probabilities(shape, dim):
    model = EspSoftmax(dim=dim, supported=False)
    inputs = (torch.zeros(shape),)
    tester = EspTester(model, inputs).test_dialect(
        model.ops_before_transforms, model.ops_after_transforms, atol=0, rtol=0
    )
    reference = tester.get_artifact(StageType.TO_EDGE).exported_program()
    (node,) = reference.graph.find_nodes(op="call_function", target=_FQ_SOFTMAX)
    assert get_scale(reference, get_qparams_from_node(node, "out")) < _OUTPUT_SCALE
    actual = tester.get_artifact().exported_program().module()(*inputs)
    row_sums = actual.sum(dim=dim)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-3, rtol=0)


def test_softmax_peaky_input_reaches_diff_min():
    model = EspSoftmax()
    inputs = (torch.linspace(-30, 30, 20).reshape(1, 20),)
    tester = EspTester(model, inputs)
    if get_scratch_queries(tester.target, tester.kernel_mode) is None:
        pytest.skip("Build with EXECUTORCH_BUILD_ESP_NN_PYBINDS=ON")
    tester.test_dialect(
        model.ops_before_transforms,
        model.ops_after_transforms,
        atol=_OUTPUT_SCALE,
        rtol=0,
    )
    interpreter = torch.fx.Interpreter(
        tester.get_artifact().exported_program().module(),
        garbage_collect_values=False,
    )
    interpreter.run(*inputs)
    (softmax,) = interpreter.module.graph.find_nodes(
        op="call_function", target=_ESP_SOFTMAX
    )
    inputs = interpreter.env[get_arg(softmax, "input")].to(torch.int32)
    differences = inputs - inputs.amax(-1, keepdim=True)
    assert torch.any(differences < get_arg(softmax, "diff_min"))


@pytest.mark.parametrize(
    "target,kernel_mode,expected_bytes",
    [
        (EspTarget.ESP32S3, EspKernelMode.OPTIMIZED, 2064),
        (EspTarget.ESP32P4, EspKernelMode.OPTIMIZED, 2064),
        (EspTarget.ESP32S31, EspKernelMode.OPTIMIZED, 2064),
        (EspTarget.ESP32, EspKernelMode.OPTIMIZED, 40),
        (EspTarget.ESP32C3, EspKernelMode.OPTIMIZED, 40),
        (EspTarget.ESP32S3, EspKernelMode.ANSI, 0),
    ],
)
def test_softmax_scratch_is_planned(target, kernel_mode, expected_bytes):
    if get_scratch_queries(target, kernel_mode) is None:
        pytest.skip("Build with EXECUTORCH_BUILD_ESP_NN_PYBINDS=ON")
    case = test_cases["rows"]
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
    (call,) = (
        instruction.instr_args
        for chain in plan.chains
        for instruction in chain.instructions
        if isinstance(instruction.instr_args, KernelCall)
        and plan.operators[instruction.instr_args.op_index].name
        == "esp_nn::quantized_softmax_int8"
    )
    scratch_index = next(
        i
        for i, arg in enumerate(_ESP_SOFTMAX._schema.arguments)
        if arg.name == "scratch"
    )
    scratch = plan.values[call.args[scratch_index]].val
    assert isinstance(scratch, Tensor)
    assert scratch.scalar_type == ScalarType.BYTE
    assert scratch.sizes == [expected_bytes]
    assert scratch.data_buffer_idx == 0
    if expected_bytes:
        assert scratch.allocation_info.memory_id > 0
        assert scratch.allocation_info.memory_offset % 16 == 0
        start = scratch.allocation_info.memory_offset
        end = start + expected_bytes
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


@pytest.mark.parametrize(
    "input_scale,supported",
    [
        (2**-27, True),
        (1 / 256, True),
        (0.125, True),
        (8, True),
        (1e-12, False),
        (16, False),
        (1e38, False),
    ],
)
def test_softmax_arithmetic_bounds(input_scale, supported):
    params = _softmax_parameters(input_scale)
    assert (params is not None) == supported
    if params is not None:
        multiplier, shift, diff_min = params
        assert 0 <= shift <= 30
        assert -diff_min * (1 << shift) <= torch.iinfo(torch.int32).max
        assert math.ldexp(multiplier / float(1 << 31), shift - 26) == pytest.approx(
            input_scale, rel=2**-30
        )


@pytest.mark.parametrize(
    "float_input,scale,zero_point",
    [
        (True, _OUTPUT_SCALE, -128),
        (False, None, None),
        (False, 1 / 128, -128),
        (False, _OUTPUT_SCALE, 0),
    ],
    ids=["float_input", "float_output", "output_scale", "output_zero_point"],
)
def test_softmax_unsupported_qparams_stay_portable(float_input, scale, zero_point):
    config = get_symmetric_quantization_config()
    config = replace(
        config,
        input_activation=None if float_input else config.input_activation,
        output_activation=(
            FixedQParamsQuantizationSpec(
                dtype=torch.int8,
                scale=scale,
                zero_point=zero_point,
                quant_min=-128,
                quant_max=127,
                qscheme=torch.per_tensor_affine,
            )
            if scale is not None
            else None
        ),
    )
    quantizer = FusedQuantQuantizer(
        [
            OpQuantizer(
                torch.ops.aten._softmax.default,
                torch.ops.fused_quant._softmax.default,
                config,
            )
        ]
    )
    case = test_cases["rows"]
    tester = (
        EspTester(case.model, case.example_inputs)
        .export_fused_quant(quantizer)
        .to_edge()
        .check_node_count({_FQ_SOFTMAX: 1})
    )
    expected = tester.get_artifact().exported_program().module()(*case.example_inputs)
    tester.run_passes().check_node_count({_FQ_SOFTMAX: 0, _ESP_SOFTMAX: 0, _SOFTMAX: 1})
    actual = tester.get_artifact().exported_program().module()(*case.example_inputs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    tester.to_executorch().serialize()


@pytest.mark.parametrize("missing", ["module", "getter"])
def test_softmax_falls_back_without_sizing(monkeypatch, missing):
    if missing == "module":
        monkeypatch.setattr(
            "executorch.backends.espressif.passes.lower_to_esp.get_scratch_queries",
            lambda *args: None,
        )
    else:
        queries = get_scratch_queries(EspTarget.ESP32S3, EspKernelMode.OPTIMIZED)
        if queries is None:
            pytest.skip("Build with EXECUTORCH_BUILD_ESP_NN_PYBINDS=ON")
        monkeypatch.delattr(queries, "softmax_scratch_size")
    case = test_cases["rows"]
    EspTester(case.model, case.example_inputs, target=EspTarget.ESP32S3).test_dialect(
        {_FQ_SOFTMAX: 1}, {_ESP_SOFTMAX: 0, _SOFTMAX: 1}, atol=0, rtol=0
    ).to_executorch().serialize()


def test_implementation_softmax_zero_scratch(pytestconfig):
    if get_scratch_queries(EspTarget.ESP32S3, EspKernelMode.ANSI) is None:
        pytest.skip("Build with EXECUTORCH_BUILD_ESP_NN_PYBINDS=ON")
    case = test_cases["rows"]
    tester = EspTester(case.model, case.example_inputs, kernel_mode=EspKernelMode.ANSI)
    tester.test_dialect(
        case.model.ops_before_transforms,
        case.model.ops_after_transforms,
        atol=_OUTPUT_SCALE,
        rtol=0,
    )
    (node,) = (
        tester.get_artifact()
        .exported_program()
        .graph.find_nodes(op="call_function", target=_ESP_SOFTMAX)
    )
    assert get_arg(node, "scratch").meta["val"].numel() == 0
    tester.test_implementation(
        pytestconfig.rootpath, atol=_OUTPUT_SCALE, num_inferences=2
    )


def test_implementation_softmax_undersized_scratch(pytestconfig):
    if not os.environ.get("ESPRESSIF_BUILD_DIR"):
        pytest.skip("Set ESPRESSIF_BUILD_DIR to the ESP32-S3 build directory")
    case = test_cases["rows"]
    tester = (
        EspTester(case.model, case.example_inputs)
        .export_fused_quant()
        .to_edge()
        .run_passes()
    )
    module = tester.get_artifact().exported_program().graph_module
    (node,) = module.graph.find_nodes(op="call_function", target=_ESP_SOFTMAX)
    scratch = get_arg(node, "scratch")
    scratch.args = (((1,), torch.uint8),)
    scratch.meta["val"] = scratch.meta["val"].new_empty((1,))
    module.recompile()
    with pytest.raises(AssertionError):
        tester.test_implementation(pytestconfig.rootpath, atol=_OUTPUT_SCALE)
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    log = (
        Path(os.environ["ESPRESSIF_BUILD_DIR"])
        / f"operator-library-{worker}-{tester.target.value}-{tester.kernel_mode.value}"
        / "emulator.log"
    ).read_text()
    assert "op_quantized_softmax.cpp" in log
    assert "scratch.nbytes()" in log
