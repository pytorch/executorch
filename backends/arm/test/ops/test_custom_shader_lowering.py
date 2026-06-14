# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import shutil
import sys
from pathlib import Path

import executorch.backends.arm.tosa.dialect  # noqa: F401
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from backends.arm.test._custom_vgf_test_utils import (
    EncodeSamplerGridSampleToTosaCustomPass,
    EncodeTestAddToTosaCustomPass,
    EncodeThreesToTosaCustomPass,
    register_test_shader_library_ops,
    register_test_threes_library_ops,
    rewrite_aten_add_to_test_shader,
    rewrite_aten_grid_sample_to_test_shader,
    TEST_ADD_OPERATOR,
    TEST_GRID_READ_TENSOR_OPERATOR,
    TEST_SHADER_DOMAIN,
    THREES_DOMAIN,
    THREES_OPERATOR,
)
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.backends.arm.vgf._passes.rewrite_grid_sampler_to_tosa_custom import (
    RewriteGridSamplerToTosaCustomPass,
)
from executorch.backends.arm.vgf.shaders.grid_sampler import (
    decode_payload,
    grid_sampler_2d_operator_name,
)
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export


class _AddModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


class _GridSampleModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )


class _ThreesModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.arm_test_shader_ops.threes.default(x)


class _GridReadTensorModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return torch.ops.arm_test_vulkan_custom_shader.grid_read_tensor_debug.default(
            x,
            grid,
            "bilinear",
            "zeros",
            False,
        )


# Covers lowering of a standalone custom op to a buffer-backed tosa.CUSTOM.
# Checks the emitted custom node carries the expected operator, domain, and buffer descriptors.
def test_new_custom_op_vgf_lowers_to_tosa_custom_buffer_shader():
    if shutil.which("glslc") is None:
        pytest.skip("glslc not found")
    register_test_threes_library_ops()
    exported = export(_ThreesModule(), (torch.randn(16),))

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP")):
        EncodeThreesToTosaCustomPass().call(exported.graph_module)

    custom_node = next(
        node
        for node in exported.graph_module.graph.nodes
        if node.target == exir_ops.backend.tosa.CUSTOM.default
    )
    payload = json.loads(
        bytes(custom_node.kwargs["implementation_attrs"]).decode("utf-8")
    )

    assert custom_node.kwargs["operator_name"] == THREES_OPERATOR
    assert custom_node.kwargs["domain_name"] == THREES_DOMAIN
    assert payload["input_0_vkdescriptortype"] == "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER"
    assert payload["output_0_vkdescriptortype"] == "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER"


# Covers replacing aten.add with a shader-backed custom op.
# Checks the rewritten node lowers to tosa.CUSTOM with storage-buffer descriptors.
def test_replacement_op_vgf_lowers_to_tosa_custom_shader():
    if shutil.which("glslc") is None:
        pytest.skip("glslc not found")
    register_test_shader_library_ops()
    exported = export(_AddModule(), (torch.randn(16), torch.randn(16)))
    rewrite_aten_add_to_test_shader(exported.graph_module)

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP")):
        EncodeTestAddToTosaCustomPass().call(exported.graph_module)

    custom_node = next(
        node
        for node in exported.graph_module.graph.nodes
        if node.target == exir_ops.backend.tosa.CUSTOM.default
    )
    payload = json.loads(
        bytes(custom_node.kwargs["implementation_attrs"]).decode("utf-8")
    )

    assert custom_node.kwargs["operator_name"] == TEST_ADD_OPERATOR
    assert custom_node.kwargs["domain_name"] == TEST_SHADER_DOMAIN
    assert payload["input_0_vkdescriptortype"] == "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER"
    assert payload["output_0_vkdescriptortype"] == "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER"


# Covers the in-tree grid-sampler rewrite path.
# Checks grid_sampler_2d.default lowers to tosa.CUSTOM with the Vulkan shader domain.
def test_in_tree_grid_sampler_vgf_lowers_to_tosa_custom():
    edge_model = to_edge(
        export(_GridSampleModule(), (torch.randn(1, 3, 8, 8), torch.randn(1, 4, 4, 2)))
    )

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP")):
        transformed = edge_model.transform([RewriteGridSamplerToTosaCustomPass()])

    nodes = list(transformed.exported_program().graph.nodes)
    custom_node = next(
        node for node in nodes if node.target == exir_ops.backend.tosa.CUSTOM.default
    )

    assert custom_node.kwargs["operator_name"] == grid_sampler_2d_operator_name(
        interpolation_mode=0,
        padding_mode=0,
        align_corners=False,
    )
    assert custom_node.kwargs["domain_name"] == "com.arm.VulkanCustomShader"


# Covers sampler/image descriptor selection during lowering.
# Checks the lowered payload uses combined-image-sampler input, tensor grid input, and storage-image output.
def test_sampler_shader_vgf_lowering_emits_expected_descriptor_types():
    if shutil.which("glslc") is None:
        pytest.skip("glslc not found")
    register_test_shader_library_ops()
    exported = export(
        _GridSampleModule(),
        (
            torch.randn(1, 4, 8, 8).contiguous(memory_format=torch.channels_last),
            torch.randn(1, 4, 4, 2),
        ),
    )
    rewrite_aten_grid_sample_to_test_shader(exported.graph_module)

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP")):
        EncodeSamplerGridSampleToTosaCustomPass().call(exported.graph_module)

    custom_node = next(
        node
        for node in exported.graph_module.graph.nodes
        if node.target == exir_ops.backend.tosa.CUSTOM.default
    )
    payload = json.loads(
        bytes(custom_node.kwargs["implementation_attrs"]).decode("utf-8")
    )

    assert (
        payload["input_0_vkdescriptortype"]
        == "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER"
    )
    assert payload["input_1_vkdescriptortype"] == "VK_DESCRIPTOR_TYPE_TENSOR_ARM"
    assert payload["output_0_vkdescriptortype"] == "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE"


def test_grid_read_shader_vgf_lowering_uses_distinct_custom_operator():
    if shutil.which("glslc") is None:
        pytest.skip("glslc not found")
    register_test_shader_library_ops()
    exported = export(
        _GridReadTensorModule(),
        (
            torch.randn(1, 4, 8, 8).contiguous(memory_format=torch.channels_last),
            torch.randn(1, 4, 9, 2),
        ),
    )

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP")):
        EncodeSamplerGridSampleToTosaCustomPass().call(exported.graph_module)

    custom_node = next(
        node
        for node in exported.graph_module.graph.nodes
        if node.target == exir_ops.backend.tosa.CUSTOM.default
    )

    assert custom_node.kwargs["operator_name"] == TEST_GRID_READ_TENSOR_OPERATOR


def test_sampler_shader_vgf_lowering_rejects_three_channel_image_payload():
    if shutil.which("glslc") is None:
        pytest.skip("glslc not found")
    register_test_shader_library_ops()
    exported = export(
        _GridSampleModule(),
        (
            torch.randn(1, 3, 8, 8).contiguous(memory_format=torch.channels_last),
            torch.randn(1, 4, 4, 2),
        ),
    )
    rewrite_aten_grid_sample_to_test_shader(exported.graph_module)

    with (
        TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP")),
        pytest.raises(
            ValueError,
            match="Image-backed grid_sample requires 1, 2, or 4 channels; got 3",
        ),
    ):
        EncodeSamplerGridSampleToTosaCustomPass().call(exported.graph_module)


# Covers decoding of implementation_attrs after lowering.
# Checks the payload exposes the expected entry point and binding numbering.
def test_shader_lowering_vgf_decodes_expected_implementation_attrs():
    edge_model = to_edge(
        export(_GridSampleModule(), (torch.randn(1, 3, 8, 8), torch.randn(1, 4, 4, 2)))
    )

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP")):
        transformed = edge_model.transform([RewriteGridSamplerToTosaCustomPass()])

    custom_node = next(
        node
        for node in transformed.exported_program().graph.nodes
        if node.target == exir_ops.backend.tosa.CUSTOM.default
    )
    payload = decode_payload(custom_node.kwargs["implementation_attrs"])

    assert payload["entry_point"] == "main"
    assert payload["input_0_binding"] == 0
    assert payload["input_1_binding"] == 1
    assert payload["output_0_binding"] == 2
