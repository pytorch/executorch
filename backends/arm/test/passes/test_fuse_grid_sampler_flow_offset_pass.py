# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import base64
import shutil
import subprocess  # nosec B404 - fixed shader compiler invocation
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_a16w8_quantization_config,
    get_symmetric_quantization_config,
)
from executorch.backends.arm.quantizer.quantization_config import (
    _is_canonical_flow_offset_grid_sampler,
    VGFQuantizationConfig,
)
from executorch.backends.arm.vgf._passes import fuse_grid_sampler_flow_offset
from executorch.backends.arm.vgf._passes.fuse_grid_sampler_flow_offset import (
    _match_flow_offset_grid,
    FuseGridSamplerFlowOffsetPass,
)
from executorch.backends.arm.vgf.shaders import grid_sampler as grid_sampler_shaders
from executorch.backends.arm.vgf.shaders.grid_sampler import (
    build_flow_offset_grid_sampler_payload,
    decode_payload,
    flow_offset_grid_sampler_operator_name,
    GRID_SAMPLER_2D_QUANTIZED_GRID_VK_FORMAT,
    GRID_SAMPLER_2D_SAMPLER_INT8_VK_FORMAT,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import Graph
from torchao.quantization.pt2e.quantizer import (
    FixedQParamsQuantizationSpec,
    QuantizationSpec,
)


def _build_grid_axis(
    graph: Graph,
    *,
    steps: int,
    view_shape: list[int],
    expand_shape: list[int],
) -> torch.fx.Node:
    axis = graph.call_function(
        exir_ops.edge.aten.linspace.default,
        args=(-1.0, 1.0, steps),
    )
    axis = graph.call_function(
        exir_ops.edge.aten.view_copy.default,
        args=(axis, view_shape),
    )
    return graph.call_function(
        exir_ops.edge.aten.expand_copy.default,
        args=(axis, expand_shape),
    )


def _find_grid_sampler(graph: Graph) -> torch.fx.Node:
    return next(
        node
        for node in graph.nodes
        if node.target == exir_ops.edge.aten.grid_sampler_2d.default
    )


def _build_flow_offset_grid_sampler(
    *,
    flow_channel_offset=0,
    interpolation=0,
    padding=1,
    align_corners=True,
    height=8,
    width=12,
    reverse_add_operands=False,
    add_alpha=None,
    slice_step=None,
    permutation=(0, 2, 3, 1),
):
    graph = Graph()
    image = graph.placeholder("image")
    image.meta["val"] = torch.empty(1, 4, height, width)
    flow = graph.placeholder("flow")
    flow.meta["val"] = torch.empty(1, 4, height, width)

    horizontal = _build_grid_axis(
        graph,
        steps=width,
        view_shape=[1, 1, 1, width],
        expand_shape=[1, -1, height, -1],
    )
    vertical = _build_grid_axis(
        graph,
        steps=height,
        view_shape=[1, 1, height, 1],
        expand_shape=[1, -1, -1, width],
    )
    base_grid = graph.call_function(
        exir_ops.edge.aten.cat.default,
        args=([horizontal, vertical], 1),
    )
    base_grid = graph.call_function(
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        args=(base_grid, 0.01, 0, -128, 127, torch.int8),
    )
    base_grid = graph.call_function(
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        args=(base_grid, 0.01, 0, -128, 127, torch.int8),
    )
    flow_slice = graph.call_function(
        exir_ops.edge.aten.slice_copy.Tensor,
        args=(flow, 1, flow_channel_offset, flow_channel_offset + 2),
        kwargs={} if slice_step is None else {"step": slice_step},
    )
    add_args = (
        (flow_slice, base_grid) if reverse_add_operands else (base_grid, flow_slice)
    )
    add_kwargs = {} if add_alpha is None else {"alpha": add_alpha}
    grid = graph.call_function(
        exir_ops.edge.aten.add.Tensor, args=add_args, kwargs=add_kwargs
    )
    grid = graph.call_function(
        exir_ops.edge.aten.permute_copy.default,
        args=(grid, list(permutation)),
    )
    grid_sampler = graph.call_function(
        exir_ops.edge.aten.grid_sampler_2d.default,
        args=(image, grid, interpolation, padding, align_corners),
    )
    grid_sampler.meta["val"] = torch.empty(1, 4, height, width)
    return grid_sampler, flow


def _build_quantized_flow_offset_graph_module(
    *, image_channels=4, flow_channel_offset=0
):
    grid_sampler, flow = _build_flow_offset_grid_sampler(
        flow_channel_offset=flow_channel_offset
    )
    image = grid_sampler.args[0]
    assert isinstance(image, torch.fx.Node)

    image_qparams = QuantArgs(0.02, -3, -127, 127, torch.int8)
    flow_qparams = QuantArgs(0.04, -5, -128, 127, torch.int8)
    output_qparams = QuantArgs(0.03, 4, -127, 127, torch.int8)

    image.meta["val"] = torch.empty(1, image_channels, 8, 12, dtype=torch.int8)
    flow.meta["val"] = torch.empty(1, 4, 8, 12, dtype=torch.int8)
    flow.meta["output_qparams"] = {0: flow_qparams}
    grid_sampler.meta["val"] = torch.empty(1, image_channels, 8, 12, dtype=torch.int8)
    grid_sampler.meta["input_qparams"] = {0: image_qparams}
    grid_sampler.meta["output_qparams"] = {0: output_qparams}
    grid_sampler.graph.output(grid_sampler)

    graph_module = torch.fx.GraphModule(torch.nn.Module(), grid_sampler.graph)
    return (
        graph_module,
        flow,
        image_qparams,
        flow_qparams,
        output_qparams,
    )


def _build_payload(**overrides):
    kwargs = {
        "input_shape": (1, 4, 8, 12),
        "output_shape": (1, 4, 8, 12),
        "flow_shape": (1, 4, 8, 12),
        "input_scale": 0.02,
        "input_zero_point": -3,
        "output_scale": 0.03,
        "output_zero_point": 4,
        "flow_scale": 0.04,
        "flow_zero_point": -5,
        "flow_channel_offset": 0,
    }
    kwargs.update(overrides)
    return build_flow_offset_grid_sampler_payload(**kwargs)


@pytest.fixture(scope="module")
def _arm_tensor_glslc():
    glslc = shutil.which("glslc")
    if glslc is None:
        pytest.skip("glslc not found")
    source = (
        "#version 450\n"
        "#extension GL_ARM_tensors : require\n"
        "#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require\n"
        "layout(set = 0, binding = 0) uniform tensorARM<int8_t, 4> value;\n"
        "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\n"
        "void main() {}\n"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "arm_tensor_probe.glsl"
        spirv_path = Path(tmpdir) / "arm_tensor_probe.spv"
        source_path.write_text(source, encoding="utf-8")
        result = subprocess.run(  # nosec B603 - glslc path is resolved from PATH.
            [glslc, "-fshader-stage=compute", str(source_path), "-o", str(spirv_path)],
            check=False,
            capture_output=True,
        )
    if result.returncode != 0:
        pytest.skip("glslc does not support GL_ARM_tensors")


def _build_aten_flow_offset_grid_sampler(
    *,
    padding=1,
    height=8,
    width=12,
    image_shape=None,
    flow_shape=None,
    output_shape=None,
    reverse_add_operands=False,
    add_alpha=None,
    slice_step=None,
    linspace_dtype=None,
):
    graph = Graph()
    image = graph.placeholder("image")
    image.meta["val"] = torch.empty(image_shape or (1, 4, height, width))
    flow = graph.placeholder("flow")
    flow.meta["val"] = torch.empty(flow_shape or (1, 4, height, width))

    linspace_kwargs = {} if linspace_dtype is None else {"dtype": linspace_dtype}

    horizontal = graph.call_function(
        torch.ops.aten.linspace.default,
        args=(-1.0, 1.0, width),
        kwargs=linspace_kwargs,
    )
    horizontal = graph.call_function(
        torch.ops.aten.view.default,
        args=(horizontal, [1, 1, 1, width]),
    )
    horizontal = graph.call_function(
        torch.ops.aten.expand.default,
        args=(horizontal, [1, -1, height, -1]),
    )
    vertical = graph.call_function(
        torch.ops.aten.linspace.default,
        args=(-1.0, 1.0, height),
        kwargs=linspace_kwargs,
    )
    vertical = graph.call_function(
        torch.ops.aten.view.default,
        args=(vertical, [1, 1, height, 1]),
    )
    vertical = graph.call_function(
        torch.ops.aten.expand.default,
        args=(vertical, [1, -1, -1, width]),
    )
    base_grid = graph.call_function(
        torch.ops.aten.cat.default,
        args=([horizontal, vertical], 1),
    )
    base_grid.meta["val"] = torch.empty(1, 2, height, width)
    flow_slice = graph.call_function(
        torch.ops.aten.slice.Tensor,
        args=(flow, 1, 0, 2),
        kwargs={} if slice_step is None else {"step": slice_step},
    )
    add_args = (
        (flow_slice, base_grid) if reverse_add_operands else (base_grid, flow_slice)
    )
    grid = graph.call_function(
        torch.ops.aten.add.Tensor,
        args=add_args,
        kwargs={} if add_alpha is None else {"alpha": add_alpha},
    )
    grid = graph.call_function(
        torch.ops.aten.permute.default,
        args=(grid, [0, 2, 3, 1]),
    )
    grid_sampler = graph.call_function(
        torch.ops.aten.grid_sampler.default,
        args=(image, grid, 0, padding, True),
    )
    grid_sampler.meta["val"] = torch.empty(
        output_shape or tuple(image.meta["val"].shape)
    )
    return grid_sampler, image


def _vgf_quantization_config(config=None):
    config = config or get_symmetric_quantization_config()
    return VGFQuantizationConfig(
        config.input_activation,
        config.output_activation,
        config.weight,
        config.bias,
        config.label,
    )


@pytest.mark.parametrize(
    ("kwargs", "expected_offset"),
    [
        pytest.param({}, 0, id="first-flow-pair"),
        pytest.param({"flow_channel_offset": 2}, 2, id="second-flow-pair"),
        pytest.param({"reverse_add_operands": True}, 0, id="reversed-add"),
        pytest.param({"add_alpha": 1}, 0, id="explicit-unit-alpha"),
        pytest.param({"slice_step": 1}, 0, id="explicit-unit-slice-step"),
    ],
)
def test_match_flow_offset_grid_accepts_supported_variants(kwargs, expected_offset):
    grid_sampler, flow = _build_flow_offset_grid_sampler(**kwargs)

    assert _match_flow_offset_grid(grid_sampler) == (flow, expected_offset)


def test_match_flow_offset_grid_accepts_identity_base_grid_requantization():
    grid_sampler, flow = _build_flow_offset_grid_sampler()
    quantize = next(
        node
        for node in grid_sampler.graph.nodes
        if node.target == exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
    )
    dequantize = quantize.args[0]
    quantize.update_arg(1, 0.007842528633773327)
    quantize.update_arg(2, -1)
    dequantize.update_arg(1, 0.007842586375772953)
    dequantize.update_arg(2, -1)

    assert _match_flow_offset_grid(grid_sampler) == (flow, 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"interpolation": 1}, id="nearest-interpolation"),
        pytest.param({"padding": 0}, id="zero-padding"),
        pytest.param({"align_corners": False}, id="unaligned-corners"),
        pytest.param({"flow_channel_offset": 1}, id="non-pair-flow-slice"),
        pytest.param({"slice_step": 2}, id="non-unit-slice-step"),
        pytest.param({"add_alpha": 2}, id="non-unit-add-alpha"),
        pytest.param({"height": 1}, id="singleton-height"),
        pytest.param({"width": 1}, id="singleton-width"),
        pytest.param({"permutation": (0, 3, 2, 1)}, id="wrong-permutation"),
    ],
)
def test_match_flow_offset_grid_rejects_unsupported_variants(kwargs):
    grid_sampler, _ = _build_flow_offset_grid_sampler(**kwargs)

    assert _match_flow_offset_grid(grid_sampler) is None


@pytest.mark.parametrize(
    ("start", "kwargs"),
    [
        pytest.param(-0.5, {}, id="wrong-start"),
        pytest.param(-1.0, {"dtype": torch.int8}, id="integer-dtype"),
    ],
)
def test_match_flow_offset_grid_rejects_noncanonical_linspace(start, kwargs):
    grid_sampler, _ = _build_flow_offset_grid_sampler()
    horizontal = next(
        node
        for node in grid_sampler.graph.nodes
        if node.target == exir_ops.edge.aten.linspace.default
    )
    horizontal.update_arg(0, start)
    horizontal.kwargs = kwargs

    assert _match_flow_offset_grid(grid_sampler) is None


@pytest.mark.parametrize(
    ("arg_index", "value"),
    [
        (1, 0.02),
        (2, 1),
        (3, -127),
        (4, 126),
        (5, torch.int16),
    ],
)
def test_match_flow_offset_grid_rejects_requantized_base_grid(arg_index, value):
    grid_sampler, _ = _build_flow_offset_grid_sampler()
    quantize = next(
        node
        for node in grid_sampler.graph.nodes
        if node.target == exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
    )
    quantize.update_arg(arg_index, value)

    assert _match_flow_offset_grid(grid_sampler) is None


@pytest.mark.parametrize("image_shape", [(2, 4, 8, 12), (1, 2, 8, 12)])
def test_match_flow_offset_grid_rejects_unsupported_image_shape(image_shape):
    grid_sampler, _ = _build_flow_offset_grid_sampler()
    image = grid_sampler.args[0]
    assert isinstance(image, torch.fx.Node)
    image.meta["val"] = torch.empty(image_shape)

    assert _match_flow_offset_grid(grid_sampler) is None


@pytest.mark.parametrize("value_name", ["input", "output"])
def test_match_flow_offset_grid_rejects_different_spatial_shape(value_name):
    grid_sampler, _ = _build_flow_offset_grid_sampler()
    value_node = grid_sampler.args[0] if value_name == "input" else grid_sampler
    assert isinstance(value_node, torch.fx.Node)
    value_node.meta["val"] = torch.empty(1, 4, 7, 12)

    assert _match_flow_offset_grid(grid_sampler) is None


def test_match_flow_offset_grid_rejects_non_4d_flow():
    grid_sampler, flow = _build_flow_offset_grid_sampler()
    flow.meta["val"] = torch.empty(1, 4, 8)

    assert _match_flow_offset_grid(grid_sampler) is None


@pytest.mark.parametrize(("image_channels", "flow_channel_offset"), [(3, 0), (4, 2)])
def test_fuse_flow_offset_grid_sampler_rewrites_image_paths(
    monkeypatch, image_channels, flow_channel_offset
):
    (
        graph_module,
        flow,
        image_qparams,
        flow_qparams,
        output_qparams,
    ) = _build_quantized_flow_offset_graph_module(
        image_channels=image_channels,
        flow_channel_offset=flow_channel_offset,
    )
    monkeypatch.setattr(
        fuse_grid_sampler_flow_offset,
        "build_flow_offset_grid_sampler_payload",
        lambda **kwargs: kwargs,
    )

    result = FuseGridSamplerFlowOffsetPass()(graph_module)
    nodes = list(result.graph_module.graph.nodes)

    assert result.modified
    assert not any(
        node.target == exir_ops.edge.aten.grid_sampler_2d.default for node in nodes
    )
    custom_node = next(
        node for node in nodes if node.target == exir_ops.backend.tosa.CUSTOM.default
    )
    assert custom_node.args[0][1] is flow
    payload = decode_payload(custom_node.kwargs["implementation_attrs"])
    assert payload["input_scale"] == image_qparams.get_scale_per_tensor()
    assert payload["input_zero_point"] == image_qparams.get_zp_per_tensor()
    assert payload["flow_scale"] == flow_qparams.get_scale_per_tensor()
    assert payload["flow_zero_point"] == flow_qparams.get_zp_per_tensor()
    assert payload["output_scale"] == output_qparams.get_scale_per_tensor()
    assert payload["output_zero_point"] == output_qparams.get_zp_per_tensor()
    assert payload["flow_channel_offset"] == flow_channel_offset

    pad_nodes = [
        node
        for node in nodes
        if node.target == exir_ops.edge.aten.constant_pad_nd.default
    ]
    slice_count = sum(
        node.target == exir_ops.edge.aten.slice_copy.Tensor for node in nodes
    )
    if image_channels == 3:
        assert len(pad_nodes) == 1
        assert pad_nodes[0].meta["input_qparams"] == {0: image_qparams}
        assert pad_nodes[0].meta["output_qparams"] == {0: image_qparams}
        assert slice_count == 1
    else:
        assert not pad_nodes
        assert slice_count == 0
    assert tuple(result.graph_module.graph.output_node().args[0].meta["val"].shape) == (
        1,
        image_channels,
        8,
        12,
    )


@pytest.mark.parametrize(
    "unsupported_qparams",
    [
        pytest.param(None, id="missing"),
        pytest.param(
            QuantArgs(0.02, 0, -32768, 32767, torch.int16),
            id="non-int8",
        ),
    ],
)
@pytest.mark.parametrize("qparams_name", ["image", "flow", "output"])
def test_fuse_flow_offset_grid_sampler_leaves_unsupported_qparams_unfused(
    qparams_name, unsupported_qparams
):
    graph_module, flow, *_ = _build_quantized_flow_offset_graph_module()
    grid_sampler = _find_grid_sampler(graph_module.graph)
    if qparams_name == "image":
        qparams_node, qparams_key = grid_sampler, "input_qparams"
    elif qparams_name == "flow":
        qparams_node, qparams_key = flow, "output_qparams"
    else:
        qparams_node, qparams_key = grid_sampler, "output_qparams"
    if unsupported_qparams is None:
        del qparams_node.meta[qparams_key]
    else:
        qparams_node.meta[qparams_key] = {0: unsupported_qparams}

    result = FuseGridSamplerFlowOffsetPass()(graph_module)

    assert not result.modified
    assert any(
        node.target == exir_ops.edge.aten.grid_sampler_2d.default
        for node in result.graph_module.graph.nodes
    )


@pytest.mark.parametrize(
    ("qparams_name", "qmin", "qmax"),
    [
        ("image", -128, 127),
        ("image", -127, 126),
        ("output", -128, 127),
        ("output", -127, 126),
    ],
)
def test_fuse_flow_offset_grid_sampler_leaves_non_snorm_qparams_unfused(
    qparams_name, qmin, qmax
):
    graph_module, *_ = _build_quantized_flow_offset_graph_module()
    grid_sampler = _find_grid_sampler(graph_module.graph)
    unsupported_qparams = QuantArgs(0.02, -3, qmin, qmax, torch.int8)
    qparams_key = "input_qparams" if qparams_name == "image" else "output_qparams"
    grid_sampler.meta[qparams_key] = {0: unsupported_qparams}

    result = FuseGridSamplerFlowOffsetPass()(graph_module)

    assert not result.modified
    assert any(
        node.target == exir_ops.edge.aten.grid_sampler_2d.default
        for node in result.graph_module.graph.nodes
    )


@pytest.mark.parametrize("compiler_failure", ["missing", "compile"])
def test_fuse_flow_offset_grid_sampler_leaves_compiler_failure_unfused(
    monkeypatch, compiler_failure
):
    graph_module, *_ = _build_quantized_flow_offset_graph_module(image_channels=3)
    if compiler_failure == "missing":
        monkeypatch.setattr(grid_sampler_shaders.shutil, "which", lambda _: None)
    else:
        monkeypatch.setattr(
            grid_sampler_shaders.shutil, "which", lambda _: "/usr/bin/glslc"
        )
        monkeypatch.setattr(
            grid_sampler_shaders.subprocess,
            "run",
            Mock(side_effect=subprocess.CalledProcessError(1, ["glslc"])),
        )

    result = FuseGridSamplerFlowOffsetPass()(graph_module)
    nodes = list(result.graph_module.graph.nodes)

    assert not result.modified
    assert any(
        node.target == exir_ops.edge.aten.grid_sampler_2d.default for node in nodes
    )
    assert not any(
        node.target == exir_ops.edge.aten.constant_pad_nd.default for node in nodes
    )


@pytest.mark.parametrize("flow_channel_offset", [0, 2])
def test_build_flow_offset_grid_sampler_payload(monkeypatch, flow_channel_offset):
    compiled_shader = base64.b64encode(b"\x03\x02\x23\x07").decode("ascii")
    monkeypatch.setattr(
        grid_sampler_shaders,
        "_compile_flow_offset_grid_sampler_shader",
        lambda **_: compiled_shader,
    )
    payload = _build_payload(flow_channel_offset=flow_channel_offset)

    assert payload["operator_name"] == flow_offset_grid_sampler_operator_name()
    assert payload["flow_channel_offset"] == flow_channel_offset
    assert payload["input_0_vkformat"] == GRID_SAMPLER_2D_SAMPLER_INT8_VK_FORMAT
    assert payload["input_1_vkformat"] == GRID_SAMPLER_2D_QUANTIZED_GRID_VK_FORMAT
    assert payload["output_0_vkformat"] == GRID_SAMPLER_2D_SAMPLER_INT8_VK_FORMAT
    assert payload["shader_code"] == compiled_shader


@pytest.mark.usefixtures("_arm_tensor_glslc")
@pytest.mark.parametrize("flow_channel_offset", [0, 2])
def test_compile_flow_offset_grid_sampler_shader(flow_channel_offset):
    payload = _build_payload(flow_channel_offset=flow_channel_offset)

    assert base64.b64decode(payload["shader_code"])[:4] == b"\x03\x02\x23\x07"


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"input_shape": (1, 3, 8, 12)}, "expected static NCHW.*input"),
        ({"output_shape": (1, 3, 8, 12)}, "expected static NCHW.*output"),
        ({"flow_shape": (1, 2, 8, 12)}, "expected static NCHW.*flow"),
        (
            {"output_shape": (1, 4, 7, 12)},
            "input and output spatial shapes must match",
        ),
        (
            {"flow_shape": (1, 4, 7, 12)},
            "flow and output spatial shapes must match",
        ),
        (
            {
                "input_shape": (1, 4, 1, 12),
                "output_shape": (1, 4, 1, 12),
                "flow_shape": (1, 4, 1, 12),
            },
            "requires H and W greater than 1",
        ),
        ({"flow_channel_offset": 1}, "flow_channel_offset must be 0 or 2"),
    ],
)
def test_build_flow_offset_grid_sampler_payload_rejects_invalid_contract(
    overrides, error
):
    with pytest.raises(ValueError, match=error):
        _build_payload(**overrides)


def test_vgf_quantization_uses_snorm_safe_observed_qparams_for_flow_offset_sampler():
    grid_sampler, image = _build_aten_flow_offset_grid_sampler()
    config = _vgf_quantization_config()

    assert _is_canonical_flow_offset_grid_sampler(grid_sampler)
    input_qspec = config.get_input_act_qspec(grid_sampler, image)
    output_qspec = config.get_output_act_qspec(grid_sampler)
    for qspec, configured_qspec in (
        (input_qspec, config.input_activation),
        (output_qspec, config.output_activation),
    ):
        assert isinstance(qspec, QuantizationSpec)
        assert isinstance(configured_qspec, QuantizationSpec)
        assert qspec.observer_or_fake_quant_ctr == (
            configured_qspec.observer_or_fake_quant_ctr
        )
        assert qspec.quant_min == -127
        assert qspec.quant_max == 127


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="canonical"),
        pytest.param({"image_shape": (1, 3, 8, 12)}, id="three-channel-image"),
        pytest.param({"reverse_add_operands": True}, id="reversed-add"),
        pytest.param({"add_alpha": 1}, id="explicit-unit-alpha"),
        pytest.param({"slice_step": 1}, id="explicit-unit-slice-step"),
    ],
)
def test_vgf_quantization_matches_supported_flow_offset_variants(kwargs):
    grid_sampler, _ = _build_aten_flow_offset_grid_sampler(**kwargs)

    assert _is_canonical_flow_offset_grid_sampler(grid_sampler)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"padding": 0}, id="zero-padding"),
        pytest.param({"add_alpha": 2}, id="non-unit-add-alpha"),
        pytest.param({"slice_step": 2}, id="non-unit-slice-step"),
        pytest.param({"linspace_dtype": torch.int8}, id="integer-linspace"),
        pytest.param({"height": 1}, id="singleton-height"),
        pytest.param({"width": 1}, id="singleton-width"),
        pytest.param({"flow_shape": (1, 4)}, id="rank-two-flow"),
        pytest.param({"flow_shape": (1, 4, 8)}, id="rank-three-flow"),
        pytest.param({"flow_shape": (1, 4, 8, 12, 1)}, id="rank-five-flow"),
        pytest.param({"image_shape": (2, 4, 8, 12)}, id="batched-image"),
        pytest.param({"image_shape": (1, 2, 8, 12)}, id="two-channel-image"),
        pytest.param({"image_shape": (1, 4, 7, 12)}, id="different-image-shape"),
        pytest.param({"output_shape": (1, 4, 7, 12)}, id="different-output-shape"),
    ],
)
def test_vgf_quantization_keeps_fixed_qparams_for_unsupported_variants(kwargs):
    grid_sampler, image = _build_aten_flow_offset_grid_sampler(**kwargs)
    config = _vgf_quantization_config()

    assert not _is_canonical_flow_offset_grid_sampler(grid_sampler)
    assert isinstance(
        config.get_input_act_qspec(grid_sampler, image),
        FixedQParamsQuantizationSpec,
    )
    assert isinstance(
        config.get_output_act_qspec(grid_sampler),
        FixedQParamsQuantizationSpec,
    )


def test_vgf_quantization_leaves_a16w8_flow_offset_sampler_unquantized():
    grid_sampler, image = _build_aten_flow_offset_grid_sampler()
    config = _vgf_quantization_config(get_symmetric_a16w8_quantization_config())

    assert _is_canonical_flow_offset_grid_sampler(grid_sampler)
    assert config.get_input_act_qspec(grid_sampler, image) is None
    assert config.get_output_act_qspec(grid_sampler) is None
