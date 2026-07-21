# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.arm.tosa.dialect  # noqa: F401
import pytest
import torch
import torch.nn.functional as F
from executorch.backends.arm._passes import FoldAndAnnotateQParamsPass
from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_quantization_config,
    VgfQuantizer,
)
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.backends.arm.vgf import VgfCompileSpec
from executorch.backends.arm.vgf._passes import (
    InsertGridSamplerGridDequantPass,
    RewriteGridSamplerToTosaCustomPass,
)
from executorch.backends.arm.vgf.shaders.grid_sampler import (
    CUSTOM_SHADER_DOMAIN_NAME,
    decode_payload,
    grid_sampler_2d_operator_name,
    GRID_SAMPLER_2D_QUANTIZED_GRID_VK_FORMAT,
    GRID_SAMPLER_2D_SAMPLER_INT8_VK_FORMAT,
    GRID_SAMPLER_2D_SAMPLER_VK_FORMAT,
    GRID_SAMPLER_2D_SHADER_ENTRY_POINT,
    GRID_SAMPLER_2D_SHADER_LANGUAGE,
    GRID_SAMPLER_2D_VK_FORMAT,
)
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
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


class _RewriteGridSamplerToTosaCustomExportPass(ExportedProgramPassBase):
    # The quantized grid-sampler rewrite materializes exported constant
    # placeholders for grid scale/zero-point, so it needs ExportedProgram
    # context. Production VGF lowering injects that context via the Arm pass
    # manager adapter, but this unit test drives the pass through the generic
    # EXIR transform path instead. Wrap the graph pass so the test exercises
    # the same rewrite logic without depending on the Arm-specific adapter.
    def call(self, exported_program):
        rewrite_pass = RewriteGridSamplerToTosaCustomPass(exported_program)
        result = rewrite_pass(exported_program.graph_module)
        exported_program._graph_module = result.graph_module
        return ExportedProgramPassResult(exported_program, result.modified)


def test_get_first_user_input_placeholder_accepts_renamed_placeholder_node():
    model = GridSampler2d()
    example_inputs = (
        torch.randn(1, 3, 8, 8),
        torch.randn(1, 4, 4, 2),
    )

    exported_program = to_edge(export(model, example_inputs)).exported_program()
    first_placeholder = next(
        node for node in exported_program.graph.nodes if node.op == "placeholder"
    )
    original_target = first_placeholder.target
    first_placeholder.name = f"{original_target}_renamed"

    rewrite_pass = RewriteGridSamplerToTosaCustomPass(exported_program)

    assert (
        rewrite_pass._get_first_user_input_placeholder(exported_program.graph)
        is first_placeholder
    )
    assert first_placeholder.target == original_target


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
    assert payload["workgroup_sizes"] == [1, 1, 1]
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


def test_rewrite_grid_sampler_to_tosa_custom_sampler_dispatch_rounds_up_output():
    model = GridSampler2d()
    example_inputs = (
        torch.randn(1, 4, 32, 32),
        torch.randn(1, 17, 9, 2),
    )

    edge_model = to_edge(export(model, example_inputs))
    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP")):
        edge_model = edge_model.transform([RewriteGridSamplerToTosaCustomPass()])
    exported_program = edge_model.exported_program()
    nodes = list(exported_program.graph.nodes)

    custom_node = next(
        node for node in nodes if node.target == exir_ops.backend.tosa.CUSTOM.default
    )
    payload = decode_payload(custom_node.kwargs["implementation_attrs"])

    assert payload["input_0_type"] == "Image"
    assert payload["output_0_type"] == "Image"
    assert payload["workgroup_sizes"] == [2, 3, 1]


def test_rewrite_grid_sampler_to_tosa_custom_no_target_uses_sampler_for_c4():
    model = GridSampler2d()
    example_inputs = (
        torch.randn(1, 4, 8, 8),
        torch.randn(1, 4, 4, 2),
    )

    edge_model = to_edge(export(model, example_inputs))
    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP")):
        edge_model = edge_model.transform([RewriteGridSamplerToTosaCustomPass()])
    exported_program = edge_model.exported_program()
    nodes = list(exported_program.graph.nodes)

    custom_node = next(
        node for node in nodes if node.target == exir_ops.backend.tosa.CUSTOM.default
    )
    payload = decode_payload(custom_node.kwargs["implementation_attrs"])

    assert payload["shader_language"] == GRID_SAMPLER_2D_SHADER_LANGUAGE
    assert payload["workgroup_sizes"] == [1, 1, 1]
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
    quantizer = VgfQuantizer(
        VgfCompileSpec("TOSA-1.0+INT"),
        use_composable_quantizer=use_composable_quantizer,
    )
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=False))

    exported = export(model, example_inputs, strict=True)
    prepared = prepare_pt2e(exported.module(), quantizer)
    prepared(*example_inputs)
    converted = convert_pt2e(prepared)

    edge_model = to_edge(export(converted, example_inputs, strict=True))
    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP+INT")):
        edge_model = edge_model.transform([FoldAndAnnotateQParamsPass()])
        grid_sampler_node = next(
            node
            for node in edge_model.exported_program().graph.nodes
            if node.target == exir_ops.edge.aten.grid_sampler_2d.default
        )
        expected_grid_qparams = grid_sampler_node.meta["input_qparams"][1]
        expected_grid_scale = torch.tensor(
            [expected_grid_qparams.get_scale_per_tensor()], dtype=torch.float32
        )
        expected_grid_zero_point = torch.tensor(
            [expected_grid_qparams.get_zp_per_tensor()], dtype=torch.int32
        )
        edge_model = edge_model.transform(
            [
                InsertGridSamplerGridDequantPass(),
                _RewriteGridSamplerToTosaCustomExportPass(),
            ]
        )
    exported_program = edge_model.exported_program()
    nodes = list(exported_program.graph.nodes)

    custom_node = next(
        node for node in nodes if node.target == exir_ops.backend.tosa.CUSTOM.default
    )
    payload = decode_payload(custom_node.kwargs["implementation_attrs"])
    grid_input = custom_node.args[0][1]
    grid_scale_input = custom_node.args[0][2]
    grid_zero_point_input = custom_node.args[0][3]
    assert payload["input_0_type"] == "Image"
    assert payload["input_0_vkformat"] == GRID_SAMPLER_2D_SAMPLER_INT8_VK_FORMAT
    assert payload["input_1_type"] == "Tensor"
    assert payload["input_1_vkformat"] == GRID_SAMPLER_2D_QUANTIZED_GRID_VK_FORMAT
    assert payload["input_2_type"] == "Tensor"
    assert payload["input_2_vkformat"] == "VK_FORMAT_R32_SFLOAT"
    assert payload["input_2_binding"] == 3
    assert payload["input_3_type"] == "Tensor"
    assert payload["input_3_vkformat"] == "VK_FORMAT_R32_SINT"
    assert payload["input_3_binding"] == 4
    assert payload["output_0_type"] == "Image"
    assert payload["output_0_vkformat"] == GRID_SAMPLER_2D_SAMPLER_INT8_VK_FORMAT
    assert payload["output_0_binding"] == 2
    assert grid_input.meta["val"].dtype == torch.int8
    assert grid_scale_input.op == "placeholder"
    assert grid_scale_input.meta["val"].dtype == torch.float32
    assert grid_scale_input.meta["val"].shape == expected_grid_scale.shape
    assert torch.equal(
        exported_program.constants[grid_scale_input.name], expected_grid_scale
    )
    assert grid_zero_point_input.op == "placeholder"
    assert grid_zero_point_input.meta["val"].dtype == torch.int32
    assert grid_zero_point_input.meta["val"].shape == expected_grid_zero_point.shape
    assert torch.equal(
        exported_program.constants[grid_zero_point_input.name], expected_grid_zero_point
    )
    assert grid_input.target not in (
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
    )
    assert custom_node.meta["input_qparams"][0].qmin == -127
    assert custom_node.meta["input_qparams"][0].qmax == 127
    assert 1 not in custom_node.meta["input_qparams"]
    assert next(iter(custom_node.meta["output_qparams"].values())).qmin == -127
    assert next(iter(custom_node.meta["output_qparams"].values())).qmax == 127


def test_quantized_grid_sampler_dequantizes_grid_for_non_sampler_path():
    model = GridSampler2d().eval()
    model.interpolation_mode_ = 2
    example_inputs = (
        torch.randn(1, 4, 8, 8),
        torch.rand(1, 4, 4, 2),
    )
    quantizer = VgfQuantizer(VgfCompileSpec("TOSA-1.0+INT"))
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=False))

    exported = export(model, example_inputs, strict=True)
    prepared = prepare_pt2e(exported.module(), quantizer)
    prepared(*example_inputs)
    converted = convert_pt2e(prepared)

    edge_model = to_edge(export(converted, example_inputs, strict=True))
    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP+INT")):
        edge_model = edge_model.transform(
            [FoldAndAnnotateQParamsPass(), InsertGridSamplerGridDequantPass()]
        )

    grid_sampler_node = next(
        node
        for node in edge_model.exported_program().graph.nodes
        if node.target == exir_ops.edge.aten.grid_sampler_2d.default
    )
    grid_input = grid_sampler_node.args[1]

    assert (
        grid_input.target
        == exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
    )
    assert grid_input.meta["val"].dtype == torch.float32
    assert 1 not in grid_sampler_node.meta["input_qparams"]


def test_quantized_grid_sampler_rejects_dequantized_grid_with_int8_image_payload():
    model = GridSampler2d().eval()
    example_inputs = (
        torch.randn(1, 4, 8, 8),
        torch.rand(1, 4, 4, 2),
    )
    quantizer = VgfQuantizer(VgfCompileSpec("TOSA-1.0+INT"))
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=False))

    exported = export(model, example_inputs, strict=True)
    prepared = prepare_pt2e(exported.module(), quantizer)
    prepared(*example_inputs)
    converted = convert_pt2e(prepared)

    edge_model = to_edge(export(converted, example_inputs, strict=True))
    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP+INT")):
        edge_model = edge_model.transform([FoldAndAnnotateQParamsPass()])

    exported_program = edge_model.exported_program()
    grid_sampler_node = next(
        node
        for node in exported_program.graph.nodes
        if node.target == exir_ops.edge.aten.grid_sampler_2d.default
    )
    grid_node = grid_sampler_node.args[1]
    grid_node.meta["val"] = torch.zeros_like(grid_node.meta["val"], dtype=torch.int8)
    grid_sampler_node.meta["input_qparams"][1] = QuantArgs(
        scale=[0.01, 0.01],
        zp=[0, 0],
        qmin=-128,
        qmax=127,
        dtype=torch.int8,
        axis=3,
        per_channel=True,
    )

    with pytest.raises(
        RuntimeError,
        match="grid_sampler grid dequant only supports per-tensor qparams",
    ):
        InsertGridSamplerGridDequantPass()(exported_program.graph_module)


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

    assert payload["workgroup_sizes"] == [1, 1, 1]
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
    assert payload["workgroup_sizes"] == [1, 1, 1]
    assert payload["input_1_vkdescriptortype"] == "VK_DESCRIPTOR_TYPE_TENSOR_ARM"
    assert not any(node.target == exir_ops.edge.aten.cat.default for node in nodes)
    assert not any(
        node.target == exir_ops.edge.aten.slice_copy.Tensor for node in nodes
    )


def test_rewrite_grid_sampler_to_tosa_custom_buffer_dispatch_rounds_up_output():
    model = GridSampler2d()
    model.interpolation_mode_ = 2
    example_inputs = (
        torch.randn(1, 4, 32, 32),
        torch.randn(1, 17, 9, 2),
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
    assert payload["output_0_type"] == "Tensor"
    assert payload["input_1_vkdescriptortype"] == "VK_DESCRIPTOR_TYPE_TENSOR_ARM"
    assert payload["workgroup_sizes"] == [2, 3, 1]
