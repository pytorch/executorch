# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn.functional as F

from executorch.backends.arm._passes import DecomposeUnsupportedBilinearResizePass
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
from executorch.backends.arm.tosa.specification import TosaSpecification
from executorch.backends.arm.vgf import VgfCompileSpec, VgfPartitioner
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export


class ExactBoundaryBilinearDownscale(torch.nn.Module):
    def forward(self, x):
        return F.interpolate(
            x,
            scale_factor=1.0 / 16.0,
            mode="bilinear",
            align_corners=False,
        )


class LegalBilinearDownscale(torch.nn.Module):
    def forward(self, x):
        return F.interpolate(
            x,
            scale_factor=1.0 / 8.0,
            mode="bilinear",
            align_corners=False,
        )


class ExactBoundaryBilinearDownscaleByOutputSize(torch.nn.Module):
    def forward(self, x):
        return F.interpolate(
            x,
            size=(16, 16),
            mode="bilinear",
            align_corners=False,
        )


def _export_exact_boundary_bilinear_downscale():
    model = ExactBoundaryBilinearDownscale()
    example_inputs = (torch.randn(1, 15, 256, 256),)
    return to_edge(export(model, example_inputs))


def _get_upsample_nodes(edge_model):
    return [
        node
        for node in edge_model.exported_program().graph.nodes
        if node.target == exir_ops.edge.aten.upsample_bilinear2d.vec
    ]


def _get_avg_pool_nodes(edge_model):
    return [
        node
        for node in edge_model.exported_program().graph.nodes
        if node.target == exir_ops.edge.aten.avg_pool2d.default
    ]


def _get_slice_nodes(edge_model):
    return [
        node
        for node in edge_model.exported_program().graph.nodes
        if node.target == exir_ops.edge.aten.slice_copy.Tensor
    ]


def _transform_with_resize_decomposition(edge_model):
    return edge_model.transform(
        [
            DecomposeUnsupportedBilinearResizePass(
                TosaSpecification.create_from_string("TOSA-1.0+FP")
            )
        ]
    )


def test_tosa_partitioner_accepts_exact_boundary_bilinear_downscale():
    edge_model = _export_exact_boundary_bilinear_downscale()

    partition_result = TOSAPartitioner(TosaCompileSpec("TOSA-1.0+FP")).partition(
        edge_model.exported_program()
    )
    upsample_node = next(
        node
        for node in partition_result.tagged_exported_program.graph_module.graph.nodes
        if node.target == exir_ops.edge.aten.upsample_bilinear2d.vec
    )

    assert upsample_node.meta["delegation_tag"].startswith("tag")


def test_vgf_partitioner_accepts_exact_boundary_bilinear_downscale():
    edge_model = _export_exact_boundary_bilinear_downscale()

    partition_result = VgfPartitioner(VgfCompileSpec("TOSA-1.0+FP")).partition(
        edge_model.exported_program()
    )
    upsample_node = next(
        node
        for node in partition_result.tagged_exported_program.graph_module.graph.nodes
        if node.target == exir_ops.edge.aten.upsample_bilinear2d.vec
    )

    assert upsample_node.meta["delegation_tag"].startswith("tag")


def test_decompose_exact_boundary_bilinear_downscale_into_exact_avg_pool():
    edge_model = _export_exact_boundary_bilinear_downscale()

    transformed = _transform_with_resize_decomposition(edge_model)
    avg_pool_nodes = _get_avg_pool_nodes(transformed)
    slice_nodes = _get_slice_nodes(transformed)

    assert len(_get_upsample_nodes(transformed)) == 0
    assert len(avg_pool_nodes) == 1
    assert len(slice_nodes) == 2
    assert slice_nodes[0].args[1:] == (2, 7, 249, 1)
    assert slice_nodes[1].args[1:] == (3, 7, 249, 1)
    assert avg_pool_nodes[0].args[1] == [2, 2]
    assert avg_pool_nodes[0].args[2] == [16, 16]
    assert list(get_first_fake_tensor(avg_pool_nodes[0]).shape[2:]) == [16, 16]


def test_decompose_exact_boundary_bilinear_downscale_by_output_size():
    model = ExactBoundaryBilinearDownscaleByOutputSize()
    edge_model = to_edge(export(model, (torch.randn(1, 15, 256, 256),)))

    transformed = _transform_with_resize_decomposition(edge_model)
    avg_pool_nodes = _get_avg_pool_nodes(transformed)
    slice_nodes = _get_slice_nodes(transformed)

    assert len(_get_upsample_nodes(transformed)) == 0
    assert len(avg_pool_nodes) == 1
    assert len(slice_nodes) == 2
    assert avg_pool_nodes[0].args[1] == [2, 2]
    assert avg_pool_nodes[0].args[2] == [16, 16]


def test_decompose_resize_pass_is_noop_for_legal_downscale():
    model = LegalBilinearDownscale()
    edge_model = to_edge(export(model, (torch.randn(1, 15, 256, 256),)))

    transformed = _transform_with_resize_decomposition(edge_model)
    upsample_nodes = _get_upsample_nodes(transformed)

    assert len(upsample_nodes) == 1
    assert upsample_nodes[0].args[1] is None


def _decomposed_boundary_bilinear(x: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(
        x[:, :, 7:-7, 7:-7],
        kernel_size=2,
        stride=16,
    )


def _make_numeric_case(kind: str) -> torch.Tensor:
    shape = (1, 3, 256, 448)
    match kind:
        case "zeros":
            return torch.zeros(shape)
        case "ones":
            return torch.ones(shape)
        case "ramp_x":
            return (
                torch.linspace(0, 1, shape[-1]).view(1, 1, 1, shape[-1]).expand(shape)
            )
        case "ramp_y":
            return (
                torch.linspace(0, 1, shape[-2]).view(1, 1, shape[-2], 1).expand(shape)
            )
        case "checkerboard":
            x = torch.arange(shape[-1]).view(1, 1, 1, shape[-1])
            y = torch.arange(shape[-2]).view(1, 1, shape[-2], 1)
            return ((x + y) % 2).float().expand(shape)
        case "horizontal_edge":
            x = torch.zeros(shape)
            x[:, :, shape[-2] // 2 :, :] = 1.0
            return x
        case "vertical_edge":
            x = torch.zeros(shape)
            x[:, :, :, shape[-1] // 2 :] = 1.0
            return x
        case "corner_impulse_tl":
            x = torch.zeros(shape)
            x[:, :, 0, 0] = 1.0
            return x
        case "corner_impulse_tr":
            x = torch.zeros(shape)
            x[:, :, 0, shape[-1] - 1] = 1.0
            return x
        case "corner_impulse_bl":
            x = torch.zeros(shape)
            x[:, :, shape[-2] - 1, 0] = 1.0
            return x
        case "corner_impulse_br":
            x = torch.zeros(shape)
            x[:, :, shape[-2] - 1, shape[-1] - 1] = 1.0
            return x
        case "center_impulse":
            x = torch.zeros(shape)
            x[:, :, shape[-2] // 2, shape[-1] // 2] = 1.0
            return x
        case _:
            raise AssertionError(f"Unhandled numeric case: {kind}")


@pytest.mark.parametrize(
    "case_name",
    [
        "zeros",
        "ones",
        "ramp_x",
        "ramp_y",
        "checkerboard",
        "horizontal_edge",
        "vertical_edge",
        "corner_impulse_tl",
        "corner_impulse_tr",
        "corner_impulse_bl",
        "corner_impulse_br",
        "center_impulse",
    ],
)
def test_exact_boundary_bilinear_downscale_numeric_edge_cases(case_name: str):
    x = _make_numeric_case(case_name)
    direct = F.interpolate(
        x,
        scale_factor=1.0 / 16.0,
        mode="bilinear",
        align_corners=False,
    )
    decomposed = _decomposed_boundary_bilinear(x)
    torch.testing.assert_close(decomposed, direct, atol=1e-6, rtol=1e-6)
