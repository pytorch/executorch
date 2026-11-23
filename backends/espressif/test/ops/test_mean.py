# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import pytest
import torch
from executorch.backends.espressif.lowering import get_esp_fused_quant_recipe
from executorch.backends.espressif.passes.lower_to_esp import _mean_multiplier, _QParams
from executorch.backends.espressif.test.tester import EspTestCase, EspTester
from executorch.backends.test.suite.operators.test_mean import MeanModel
from executorch.backends.transforms.fused_quant.graph_utils import (
    get_qparams_from_node,
    get_scale,
)
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.export import export

_MEAN = exir_ops.edge.aten.mean.dim
_FQ_MEAN = exir_ops.edge.fused_quant.mean.dim
_ESP_MEAN = exir_ops.edge.esp_nn.quantized_mean_int8.default
_PERMUTE = exir_ops.edge.aten.permute_copy.default


class EspMean(MeanModel):
    def __init__(self, dim=(2, 3), *, supported=True, **kwargs):
        super().__init__(dim=dim, **kwargs)
        self.ops_before_transforms = {_FQ_MEAN: 1}
        self.ops_after_transforms = {
            _FQ_MEAN: 0,
            _MEAN: 0 if supported else 1,
            _ESP_MEAN: 1 if supported else 0,
        }


class EspMeanAfterPool(EspMean):
    def __init__(self):
        super().__init__()
        self.ops_before_transforms[
            exir_ops.edge.fused_quant.max_pool2d_with_indices.default
        ] = 1
        self.ops_after_transforms[
            exir_ops.edge.esp_nn.quantized_max_pool2d_int8.default
        ] = 1
        self.ops_after_transforms[_PERMUTE] = 1

    def forward(self, x):
        return super().forward(torch.nn.functional.max_pool2d(x, 2))


class EspMeanWithFloatUsers(EspMean):
    def forward(self, x):
        result = super().forward(x)
        return result, torch.sin(result), torch.sin(x)


def _input(shape):
    values = torch.arange(math.prod(shape)).reshape(shape)
    return (values * 0.7).sin() * 3 + values.remainder(5) * 0.2


test_cases = {
    "channels_1": EspTestCase(EspMean(), (_input((1, 1, 5, 7)),)),
    "channels_3": EspTestCase(EspMean(), (_input((1, 3, 5, 7)),)),
    "channels_4": EspTestCase(EspMean(), (_input((1, 4, 5, 7)),)),
    "channels_17": EspTestCase(EspMean(), (_input((1, 17, 3, 5)),)),
    "channels_64": EspTestCase(EspMean(), (_input((1, 64, 8, 8)),)),
    "channels_256": EspTestCase(EspMean(), (_input((1, 256, 3, 3)),)),
    "batches": EspTestCase(EspMean(), (_input((3, 7, 5, 9)),)),
    "aligned_batches": EspTestCase(EspMean(), (_input((2, 16, 5, 7)),)),
    "large_plane": EspTestCase(EspMean(), (_input((1, 16, 32, 32)),)),
    "keepdim": EspTestCase(EspMean(keepdim=True), (_input((2, 5, 7, 3)),)),
    "negative_dims": EspTestCase(EspMean(dim=(-2, -1)), (_input((1, 4, 5, 7)),)),
    "reversed_dims": EspTestCase(EspMean(dim=(3, 2)), (_input((1, 4, 5, 7)),)),
    "single_row": EspTestCase(EspMean(), (_input((1, 4, 1, 11)),)),
    "single_column": EspTestCase(EspMean(), (_input((1, 4, 11, 1)),)),
    "single_pixel": EspTestCase(EspMean(), (_input((2, 4, 1, 1)),)),
    "negative": EspTestCase(EspMean(), (_input((1, 7, 5, 9)) - 5,)),
    "zeros": EspTestCase(EspMean(), (torch.zeros(1, 4, 3, 5),)),
    "pool_producer": EspTestCase(EspMeanAfterPool(), (_input((2, 4, 8, 10)),)),
    "float_users": EspTestCase(EspMeanWithFloatUsers(), (_input((1, 4, 5, 7)),)),
    "all_dims_fallback": EspTestCase(
        EspMean(dim=None, supported=False), (_input((1, 4, 5, 7)),)
    ),
    "channel_fallback": EspTestCase(
        EspMean(dim=(1,), supported=False), (_input((1, 4, 5, 7)),)
    ),
    "batch_fallback": EspTestCase(
        EspMean(dim=(0, 2), supported=False), (_input((2, 4, 5, 7)),)
    ),
    "unbatched_fallback": EspTestCase(
        EspMean(dim=(1, 2), supported=False), (_input((4, 5, 7)),)
    ),
}


@pytest.fixture(params=test_cases.values(), ids=test_cases.keys())
def mean_case(request):
    case = request.param
    tester = EspTester(case.model, case.example_inputs).test_dialect(
        case.model.ops_before_transforms,
        case.model.ops_after_transforms,
        qtol=case.model.ops_after_transforms[_ESP_MEAN],
        atol=0,
        rtol=torch.finfo(torch.float32).eps,
    )
    return case, tester


def test_dialect_mean(mean_case):
    case, tester = mean_case
    for node in tester.get_artifact().exported_program().graph.nodes:
        if node.target == _ESP_MEAN:
            assert node.args[0].meta["val"].is_contiguous()
            assert node.args[0].meta["val"].dtype == torch.int8
            assert node.args[0].meta["val"].shape[-1] == case.example_inputs[0].shape[1]
        if node.target == _PERMUTE:
            assert node.meta["val"].dtype == torch.int8
    tester.to_executorch().serialize()
    plan = deserialize_pte_binary(tester.get_artifact()).program.execution_plan[0]
    assert not plan.delegates
    operators = {(op.name, op.overload) for op in plan.operators}
    assert (("esp_nn::quantized_mean_int8", "out") in operators) == bool(
        case.model.ops_after_transforms[_ESP_MEAN]
    )


def test_implementation_mean(mean_case, pytestconfig):
    case, tester = mean_case
    tester.test_implementation(
        pytestconfig.rootpath,
        rtol=(
            torch.finfo(torch.float32).eps
            if isinstance(case.model, EspMeanWithFloatUsers)
            else 0
        ),
        num_inferences=2,
    )


def test_mean_uses_independent_qparams():
    case = test_cases["channels_4"]
    tester = EspTester(case.model, case.example_inputs).export_fused_quant().to_edge()
    program = tester.get_artifact().exported_program()
    (mean,) = program.graph.find_nodes(op="call_function", target=_FQ_MEAN)
    output_scale = get_scale(program, get_qparams_from_node(mean, "out"))
    assert get_scale(program, get_qparams_from_node(mean, "inp")) != output_scale
    expected = program.module()(*case.example_inputs)
    tester.run_passes().check_node_count({_ESP_MEAN: 1})
    actual = tester.get_artifact().exported_program().module()(*case.example_inputs)
    torch.testing.assert_close(
        actual, expected, atol=output_scale, rtol=torch.finfo(torch.float32).eps
    )


@pytest.mark.parametrize(
    "case,input_scale,output_scale,count,supported",
    [
        ("supported", 0.125, 0.03125, 35, True),
        (
            "accumulator_count",
            1,
            1,
            torch.iinfo(torch.int32).max // 255 + 1,
            False,
        ),
        ("multiplier_too_large", 1, 1e-12, 35, False),
        ("multiplier_too_small", 1e-12, 1, 35, False),
        ("right_shift_limit", 1e-8, 1, 35, False),
        ("left_shift_overflow", 1, 1e-8, 35, False),
        ("float_rounding_limit", 1, 1, 2**23 + 1, False),
        ("reciprocal_overflow", 1e-38, 1e-39, 35, False),
        ("float_accumulation_overflow", 1e38, 1e38, 2, False),
    ],
)
def test_mean_arithmetic_bounds(case, input_scale, output_scale, count, supported):
    assert (
        _mean_multiplier(
            _QParams(input_scale, -128), _QParams(output_scale, 127), count
        )
        is not None
    ) == supported, case


def test_mean_strict_recipe():
    case = test_cases["keepdim"]
    session = export(
        case.model,
        example_inputs=[case.example_inputs],
        export_recipe=get_esp_fused_quant_recipe(allow_cpu_fallback=False),
    )
    assert session.get_executorch_program_manager().buffer
    case = test_cases["channel_fallback"]
    with pytest.raises(RuntimeError, match="aten::mean"):
        export(
            case.model,
            example_inputs=[case.example_inputs],
            export_recipe=get_esp_fused_quant_recipe(allow_cpu_fallback=False),
        )
