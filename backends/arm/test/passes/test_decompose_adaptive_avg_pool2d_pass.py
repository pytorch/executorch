# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest
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


def _run_static_decomposition(
    module: torch.nn.Module,
    inputs: input_t,
    specification="TOSA-1.0+FP",
):
    ep = export(module, inputs)
    edge_model = to_edge(ep)
    with TosaLoweringContext(TosaSpecification.create_from_string(specification)):
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


def test_decompose_adaptive_avg_pool2d_padded_regions():
    module = AdaptiveAvgPoolIrregular()
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
        ],
        pass_list=[DecomposeAdaptiveAvgPool2dPass],
    )
    pipeline.run()


def test_decompose_adaptive_avg_pool2d_large_stride_rewrites():
    module = AdaptiveAvgPoolLargeStride()
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
            "executorch_exir_dialects_edge__ops_aten_avg_pool2d_default": 1,
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
            "executorch_exir_dialects_edge__ops_aten_avg_pool2d_default": 1,
        },
        ops_not_after_pass=[
            "executorch_exir_dialects_edge__ops_aten__adaptive_avg_pool2d_default",
            "executorch_exir_dialects_backend__ops_tosa_AVG_POOL2D_ADAPTIVE_default",
        ],
        pass_list=[DecomposeAdaptiveAvgPool2dPass],
    )
    pipeline.run()


class AdaptiveAvgPoolShape(torch.nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, tensor):
        return torch.nn.functional.adaptive_avg_pool2d(tensor, self.output_size)


@pytest.mark.parametrize(
    "input_size,output_size,specification,pool_count",
    [
        ((17, 17), (5, 5), "TOSA-1.0+FP", 9),
        ((8, 8), (3, 3), "TOSA-1.0+FP", 1),
        ((7, 7), (10, 10), "TOSA-1.0+FP", 49),
        ((17, 23), (5, 7), "TOSA-1.0+FP", 9),
        ((23, 17), (7, 5), "TOSA-1.0+FP", 9),
        ((1, 1), (3, 3), "TOSA-1.0+FP", 9),
        ((7, 7), (4, 4), "TOSA-1.0+INT+u55", 1),
        ((32, 32), (4, 4), "TOSA-1.0+INT+u55", 16),
        ((9, 13), (2, 3), "TOSA-1.0+INT+u55", 6),
        ((10, 16), (2, 16), "TOSA-1.0+INT+u55", 2),
        ((7, 32), (4, 2), "TOSA-1.0+INT+u55", 6),
    ],
)
def test_decompose_adaptive_avg_pool2d_regions(
    input_size, output_size, specification, pool_count
):
    module = AdaptiveAvgPoolShape(output_size)
    inputs = (torch.randn(1, 3, *input_size),)
    transformed = _run_static_decomposition(module, inputs, specification)
    pools = [
        node
        for node in transformed.graph.nodes
        if "aten.avg_pool2d.default" in str(node.target)
    ]
    assert len(pools) == pool_count
    if "+u55" in specification:
        for node in pools:
            assert all(1 <= stride <= 3 for stride in node.args[2])
            if any(node.args[3]):
                assert all(kernel <= 8 for kernel in node.args[1])
    torch.testing.assert_close(transformed(*inputs)[0], module(*inputs))


@pytest.mark.parametrize("max_stride", [None, 3])
def test_adaptive_avg_pool2d_axis_regions_cover_exact_windows(max_stride):
    for input_size in range(1, 65):
        for output_size in range(1, 65):
            regions = DecomposeAdaptiveAvgPool2dPass._pool_regions(
                input_size, output_size, max_stride
            )
            windows = []
            for region in regions:
                assert region.stride > 0
                if max_stride is not None:
                    assert region.stride <= max_stride
                count = (
                    region.end - region.start + 2 * region.padding - region.kernel
                ) // region.stride + 1
                for index in range(count):
                    start = region.start + index * region.stride - region.padding
                    windows.append(
                        (
                            max(region.start, start),
                            min(region.end, start + region.kernel),
                        )
                    )
            assert windows == [
                DecomposeAdaptiveAvgPool2dPass._static_bin_bounds(
                    input_size, output_size, index
                )
                for index in range(output_size)
            ]


def test_adaptive_avg_pool2d_bin_bounds_use_integer_arithmetic():
    input_size = 2**60 + 3
    assert DecomposeAdaptiveAvgPool2dPass._static_bin_bounds(input_size, 3, 2) == (
        2 * input_size // 3,
        input_size,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_adaptive_avg_pool2d_grouped_without_separable(dtype):
    module = AdaptiveAvgPoolShape((5, 5))
    inputs = (torch.randn(1, 3, 17, 17).to(dtype),)
    transformed = _run_static_decomposition(module, inputs)
    assert (
        sum(
            "aten.avg_pool2d.default" in str(node.target)
            for node in transformed.graph.nodes
        )
        == 9
    )
    torch.testing.assert_close(transformed(*inputs)[0], module(*inputs))
