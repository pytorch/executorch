# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.arm.tosa.dialect  # noqa: F401
import pytest
import torch
import torch.nn.functional as F
from executorch.backends.arm._passes import FoldAndAnnotateQParamsPass
from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.backends.arm.vgf._passes.rewrite_grid_sampler_to_tosa_custom import (
    RewriteGridSamplerToTosaCustomPass,
)
from executorch.backends.arm.vgf.shaders.grid_sampler import (
    CUSTOM_SHADER_DOMAIN_NAME,
    decode_payload,
    grid_sampler_2d_operator_name,
    GRID_SAMPLER_2D_SAMPLER_INT8_VK_FORMAT,
    GRID_SAMPLER_2D_SAMPLER_VK_FORMAT,
    GRID_SAMPLER_2D_SHADER_ENTRY_POINT,
    GRID_SAMPLER_2D_SHADER_LANGUAGE,
    GRID_SAMPLER_2D_VK_FORMAT,
    GRID_SAMPLER_2D_WORKGROUP_SIZES,
)
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


class GridSampler2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.interpolation_mode_ = 0
        self.padding_mode_ = 0
        self.align_corners_ = False

    def forward(self, x, grid):
        mode = ("bilinear", "nearest", "bicubic")[self.interpolation_mode_]
        return F.grid_sample(
            x,
            grid,
            mode=mode,
            padding_mode="zeros" if self.padding_mode_ == 0 else "border",
            align_corners=self.align_corners_,
        )


def test_rewrite_grid_sampler_to_tosa_custom_vgf_no_target():
    model = GridSampler2d()
    example_inputs = (
        torch.randn(1, 3, 8, 8),
        torch.randn(1, 4, 4, 2),
    )

    edge_model = to_edge(export(model, example_inputs))
    nodes = list(edge_model.exported_program().graph.nodes)

    assert any(
        node.target == exir_ops.edge.aten.grid_sampler_2d.default for node in nodes
    )

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP")):
        edge_model = edge_model.transform([RewriteGridSamplerToTosaCustomPass()])
    nodes = list(edge_model.exported_program().graph.nodes)

    assert not any(
        node.target == exir_ops.edge.aten.grid_sampler_2d.default for node in nodes
    )

    custom_node = next(
        node for node in nodes if node.target == exir_ops.backend.tosa.CUSTOM.default
    )
    assert custom_node.kwargs["operator_name"] == grid_sampler_2d_operator_name(
        interpolation_mode=0,
        padding_mode=0,
        align_corners=False,
    )
    assert custom_node.kwargs["domain_name"] == CUSTOM_SHADER_DOMAIN_NAME

    payload = decode_payload(custom_node.kwargs["implementation_attrs"])
    assert payload["entry_point"] == GRID_SAMPLER_2D_SHADER_ENTRY_POINT
    assert payload["workgroup_sizes"] == GRID_SAMPLER_2D_WORKGROUP_SIZES
    assert payload["shader_language"] == GRID_SAMPLER_2D_SHADER_LANGUAGE
    assert payload["input_0_type"] == "Image"
    assert payload["input_0_vkformat"] == GRID_SAMPLER_2D_SAMPLER_VK_FORMAT
    assert (
        payload["input_0_vkdescriptortype"]
        == "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER"
    )
    assert payload["input_0_binding"] == 0
    assert payload["input_0_descriptorset"] == 0
    assert payload["input_1_type"] == "Tensor"
    assert payload["input_1_vkformat"] == GRID_SAMPLER_2D_VK_FORMAT
    assert payload["input_1_vkdescriptortype"] == "VK_DESCRIPTOR_TYPE_TENSOR_ARM"
    assert payload["input_1_binding"] == 1
    assert payload["input_1_descriptorset"] == 0
    assert payload["output_0_type"] == "Image"
    assert payload["output_0_vkformat"] == GRID_SAMPLER_2D_SAMPLER_VK_FORMAT
    assert payload["output_0_vkdescriptortype"] == "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE"
    assert payload["output_0_binding"] == 2
    assert payload["output_0_descriptorset"] == 0
    assert any(node.target == exir_ops.edge.aten.slice_copy.Tensor for node in nodes)


def test_rewrite_grid_sampler_to_tosa_custom_no_target_uses_sampler_for_c4():
    model = GridSampler2d()
    example_inputs = (
        torch.randn(1, 4, 8, 8),
        torch.randn(1, 4, 4, 2),
    )

    edge_model = to_edge(export(model, example_inputs))
    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP")):
        edge_model = edge_model.transform([RewriteGridSamplerToTosaCustomPass()])
    nodes = list(edge_model.exported_program().graph.nodes)

    custom_node = next(
        node for node in nodes if node.target == exir_ops.backend.tosa.CUSTOM.default
    )
    payload = decode_payload(custom_node.kwargs["implementation_attrs"])

    assert payload["shader_language"] == GRID_SAMPLER_2D_SHADER_LANGUAGE
    assert (
        payload["input_0_vkdescriptortype"]
        == "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER"
    )
    assert payload["input_0_vkformat"] == GRID_SAMPLER_2D_SAMPLER_VK_FORMAT
    assert payload["input_1_vkdescriptortype"] == "VK_DESCRIPTOR_TYPE_TENSOR_ARM"
    assert payload["output_0_vkdescriptortype"] == "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE"
    assert payload["output_0_vkformat"] == GRID_SAMPLER_2D_SAMPLER_VK_FORMAT


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("use_composable_quantizer", [False, True])
def test_quantized_grid_sampler_uses_int8_sampler_payload(
    channels, use_composable_quantizer
):
    model = GridSampler2d().eval()
    example_inputs = (
        torch.randn(1, channels, 8, 8),
        torch.rand(1, 4, 4, 2),
    )
    quantizer = TOSAQuantizer(
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
        use_composable_quantizer=use_composable_quantizer,
    )
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=False))

    exported = export(model, example_inputs, strict=True)
    prepared = prepare_pt2e(exported.module(), quantizer)
    prepared(*example_inputs)
    converted = convert_pt2e(prepared)

    edge_model = to_edge(export(converted, example_inputs, strict=True))
    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP+INT")):
        edge_model = edge_model.transform(
            [FoldAndAnnotateQParamsPass(), RewriteGridSamplerToTosaCustomPass()]
        )
    nodes = list(edge_model.exported_program().graph.nodes)

    custom_node = next(
        node for node in nodes if node.target == exir_ops.backend.tosa.CUSTOM.default
    )
    payload = decode_payload(custom_node.kwargs["implementation_attrs"])

    assert payload["input_0_type"] == "Image"
    assert payload["input_0_vkformat"] == GRID_SAMPLER_2D_SAMPLER_INT8_VK_FORMAT
    assert payload["input_1_type"] == "Tensor"
    assert payload["input_1_vkformat"] == GRID_SAMPLER_2D_VK_FORMAT
    assert payload["output_0_type"] == "Image"
    assert payload["output_0_vkformat"] == GRID_SAMPLER_2D_SAMPLER_INT8_VK_FORMAT
    assert custom_node.meta["input_qparams"][0].qmin == -127
    assert custom_node.meta["input_qparams"][0].qmax == 127
    assert next(iter(custom_node.meta["output_qparams"].values())).qmin == -127
    assert next(iter(custom_node.meta["output_qparams"].values())).qmax == 127


def test_rewrite_grid_sampler_to_tosa_custom_c3_pad_for_align_corners():
    model = GridSampler2d()
    model.align_corners_ = True
    example_inputs = (
        torch.randn(1, 3, 8, 8),
        torch.randn(1, 4, 4, 2),
    )

    edge_model = to_edge(export(model, example_inputs))
    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP")):
        edge_model = edge_model.transform([RewriteGridSamplerToTosaCustomPass()])
    nodes = list(edge_model.exported_program().graph.nodes)

    custom_node = next(
        node for node in nodes if node.target == exir_ops.backend.tosa.CUSTOM.default
    )
    payload = decode_payload(custom_node.kwargs["implementation_attrs"])

    assert payload["input_0_type"] == "Image"
    assert payload["input_0_vkformat"] == GRID_SAMPLER_2D_SAMPLER_VK_FORMAT
    assert (
        payload["input_0_vkdescriptortype"]
        == "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER"
    )
    assert payload["output_0_type"] == "Image"
    assert payload["output_0_vkformat"] == GRID_SAMPLER_2D_SAMPLER_VK_FORMAT
    assert any(node.target == exir_ops.edge.aten.cat.default for node in nodes)
    assert any(node.target == exir_ops.edge.aten.slice_copy.Tensor for node in nodes)


def test_rewrite_grid_sampler_to_tosa_custom_no_c3_pad_for_bicubic():
    model = GridSampler2d()
    model.interpolation_mode_ = 2
    example_inputs = (
        torch.randn(1, 3, 8, 8),
        torch.randn(1, 4, 4, 2),
    )

    edge_model = to_edge(export(model, example_inputs))
    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP")):
        edge_model = edge_model.transform([RewriteGridSamplerToTosaCustomPass()])
    nodes = list(edge_model.exported_program().graph.nodes)

    custom_node = next(
        node for node in nodes if node.target == exir_ops.backend.tosa.CUSTOM.default
    )
    payload = decode_payload(custom_node.kwargs["implementation_attrs"])

    assert payload["input_0_type"] == "Tensor"
    assert not any(node.target == exir_ops.edge.aten.cat.default for node in nodes)
    assert not any(
        node.target == exir_ops.edge.aten.slice_copy.Tensor for node in nodes
    )
