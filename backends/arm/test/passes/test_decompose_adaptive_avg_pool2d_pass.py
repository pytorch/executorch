# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm._passes.decompose_adaptive_avg_pool2d_pass import (
    DecomposeAdaptiveAvgPool2dPass,
)
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir import to_edge
from torch.export import export

input_t = Tuple[torch.Tensor]


class AdaptiveAvgPoolUniform(torch.nn.Module):
    def get_inputs(self) -> input_t:
        return (torch.rand(1, 3, 8, 8),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))


class AdaptiveAvgPoolIrregular(torch.nn.Module):
    def get_inputs(self) -> input_t:
        return (torch.rand(1, 3, 7, 7),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))


class AdaptiveAvgPoolLargeStride(torch.nn.Module):
    def get_inputs(self) -> input_t:
        return (torch.rand(1, 3, 32, 32),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))


class AdaptiveAvgPoolAsymmetric(torch.nn.Module):
    def get_inputs(self) -> input_t:
        return (torch.rand(1, 3, 9, 13),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.adaptive_avg_pool2d(x, (2, 3))


class AdaptiveAvgPoolKeepWidth(torch.nn.Module):
    def get_inputs(self) -> input_t:
        return (torch.rand(1, 3, 10, 16),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.adaptive_avg_pool2d(x, (2, None))


def _run_static_decomposition(module: torch.nn.Module, inputs: input_t):
    ep = export(module, inputs)
    edge_model = to_edge(ep)
    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP")):
        edge_model = edge_model.transform([DecomposeAdaptiveAvgPool2dPass()])
    return edge_model.exported_program().graph_module


def test_decompose_adaptive_avg_pool2d_uniform_regions_rewrite_to_avg_pool2d():
    module = AdaptiveAvgPoolUniform()
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten__adaptive_avg_pool2d_default": 1,
        },
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_avg_pool2d_default": 1,
        },
        ops_not_after_pass=[
            "executorch_exir_dialects_edge__ops_aten__adaptive_avg_pool2d_default",
            "executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor",
            "executorch_exir_dialects_edge__ops_aten_cat_default",
            "executorch_exir_dialects_backend__ops_tosa_AVG_POOL2D_ADAPTIVE_default",
        ],
        pass_list=[DecomposeAdaptiveAvgPool2dPass],
    )
    pipeline.run()


def test_decompose_adaptive_avg_pool2d_no_target_irregular_regions():
    module = AdaptiveAvgPoolIrregular()
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten__adaptive_avg_pool2d_default": 1,
        },
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_avg_pool2d_default": 16,
            "executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor": 32,
            "executorch_exir_dialects_edge__ops_aten_cat_default": 5,
        },
        ops_not_after_pass=[
            "executorch_exir_dialects_edge__ops_aten__adaptive_avg_pool2d_default",
        ],
        pass_list=[DecomposeAdaptiveAvgPool2dPass],
    )
    pipeline.run()


def test_decompose_adaptive_avg_pool2d_no_target_large_stride_still_decomposes():
    module = AdaptiveAvgPoolLargeStride()
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten__adaptive_avg_pool2d_default": 1,
        },
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_avg_pool2d_default": 16,
            "executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor": 32,
            "executorch_exir_dialects_edge__ops_aten_cat_default": 5,
        },
        ops_not_after_pass=[
            "executorch_exir_dialects_edge__ops_aten__adaptive_avg_pool2d_default",
        ],
        pass_list=[DecomposeAdaptiveAvgPool2dPass],
    )
    pipeline.run()


def test_decompose_adaptive_avg_pool2d_asymmetric_regions_compare_numerically():
    module = AdaptiveAvgPoolAsymmetric()
    inputs = (
        torch.arange(1, 1 + 1 * 3 * 9 * 13, dtype=torch.float32).reshape(1, 3, 9, 13),
    )
    transformed = _run_static_decomposition(module, inputs)

    reference = module(*inputs)
    result = transformed(*inputs)
    if isinstance(result, tuple):
        result = result[0]

    assert torch.allclose(result, reference)


def test_decompose_adaptive_avg_pool2d_asymmetric_regions_decompose():
    module = AdaptiveAvgPoolAsymmetric()
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten__adaptive_avg_pool2d_default": 1,
        },
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_avg_pool2d_default": 6,
            "executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor": 12,
            "executorch_exir_dialects_edge__ops_aten_cat_default": 3,
        },
        ops_not_after_pass=[
            "executorch_exir_dialects_edge__ops_aten__adaptive_avg_pool2d_default",
            "executorch_exir_dialects_backend__ops_tosa_AVG_POOL2D_ADAPTIVE_default",
        ],
        pass_list=[DecomposeAdaptiveAvgPool2dPass],
    )
    pipeline.run()


def test_decompose_adaptive_avg_pool2d_keep_width_decompose():
    module = AdaptiveAvgPoolKeepWidth()
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten__adaptive_avg_pool2d_default": 1,
        },
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_avg_pool2d_default": 32,
            "executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor": 64,
            "executorch_exir_dialects_edge__ops_aten_cat_default": 3,
        },
        ops_not_after_pass=[
            "executorch_exir_dialects_edge__ops_aten__adaptive_avg_pool2d_default",
            "executorch_exir_dialects_backend__ops_tosa_AVG_POOL2D_ADAPTIVE_default",
        ],
        pass_list=[DecomposeAdaptiveAvgPool2dPass],
    )
    pipeline.run()
