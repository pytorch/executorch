# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json

import pytest
import torch
from examples.arm.QAT_example.rife_vgf import shaders as warp_downsample_shaders
from examples.arm.QAT_example.rife_vgf.extension import (
    _ensure_warp_downsample_ops_registered,
)
from examples.arm.QAT_example.rife_vgf.passes.rewrite_warp_downsample_to_tosa_custom import (
    _restore_normalized_boundary_input,
    _target_scale,
    RewriteWarpDownsampleToTosaCustomPass,
)
from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.backends.arm.constants import NHWC_INVERSE_ORDER
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult


class _Schema:
    name = "rife::warp_downsample8"
    overload_name = ""


class _EdgeTarget:
    __name__ = "rife.warp_downsample8.default"
    _schema = _Schema()


class _UnsupportedEdgeTarget:
    __name__ = "rife.warp_downsample20.default"


@pytest.fixture(autouse=True)
def _stub_warp_downsample_shader_compile(monkeypatch) -> None:
    monkeypatch.setattr(
        warp_downsample_shaders,
        "_compile_shader_source",
        lambda source: "compiled_shader",
    )


def test_target_scale_matches_edge_op_name_exactly() -> None:
    assert _target_scale(_EdgeTarget()) == 8
    assert _target_scale(_UnsupportedEdgeTarget()) is None


def _warp_downsample_graph(
    *,
    image_shape: tuple[int, int, int, int] = (1, 4, 16, 16),
    flow_shape: tuple[int, int, int, int] = (1, 2, 16, 16),
    output_shape: tuple[int, int, int, int] = (1, 4, 8, 8),
    dtype: torch.dtype = torch.int8,
    with_qparams: bool = True,
) -> torch.fx.GraphModule:
    _ensure_warp_downsample_ops_registered()
    graph = torch.fx.Graph()
    image = graph.placeholder("image")
    flow = graph.placeholder("flow")
    image.meta["val"] = torch.empty(*image_shape, dtype=dtype)
    flow.meta["val"] = torch.empty(*flow_shape, dtype=dtype)
    warp = graph.call_function(
        torch.ops.rife.warp_downsample2.default,
        (image, flow),
    )
    warp.meta["val"] = torch.empty(*output_shape, dtype=dtype)
    if with_qparams:
        snorm_qparams = QuantArgs(1.0 / 127.0, 0, -127, 127, torch.int8)
        flow_qparams = QuantArgs(0.25, -3, -128, 127, torch.int8)
        warp.meta["input_qparams"] = {0: snorm_qparams, 1: flow_qparams}
        warp.meta["output_qparams"] = {0: snorm_qparams}
    graph.output(warp)
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def _rewrite_warp_downsample(graph_module: torch.fx.GraphModule) -> PassResult:
    result = RewriteWarpDownsampleToTosaCustomPass()(graph_module)
    assert result is not None
    return result


def test_rewrite_warp_downsample_rejects_flow_spatial_mismatch() -> None:
    graph_module = _warp_downsample_graph(flow_shape=(1, 2, 8, 16))

    with pytest.raises(RuntimeError, match=r"flow \[1, 2 or 4, H, W\]"):
        with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT")):
            RewriteWarpDownsampleToTosaCustomPass()(graph_module)


def test_rewrite_warp_downsample_rejects_output_shape_mismatch() -> None:
    graph_module = _warp_downsample_graph(output_shape=(1, 4, 7, 8))

    with pytest.raises(RuntimeError, match="H / scale"):
        with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT")):
            RewriteWarpDownsampleToTosaCustomPass()(graph_module)


def test_rewrite_warp_downsample_rejects_fp32() -> None:
    graph_module = _warp_downsample_graph(dtype=torch.float32)

    with pytest.raises(RuntimeError, match="requires int8 NCHW input"):
        with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT")):
            RewriteWarpDownsampleToTosaCustomPass()(graph_module)


def test_rewrite_warp_downsample_pads_c3_input_and_slices_c3_output() -> None:
    graph_module = _warp_downsample_graph(
        image_shape=(1, 3, 16, 16),
        output_shape=(1, 3, 8, 8),
    )

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT")):
        result = _rewrite_warp_downsample(graph_module)

    assert result.modified
    custom_nodes = [
        node
        for node in result.graph_module.graph.nodes
        if node.target == exir_ops.backend.tosa.CUSTOM.default
    ]
    pad_nodes = [
        node
        for node in result.graph_module.graph.nodes
        if node.target == exir_ops.edge.aten.constant_pad_nd.default
    ]
    slice_nodes = [
        node
        for node in result.graph_module.graph.nodes
        if node.target == exir_ops.edge.aten.slice_copy.Tensor
    ]
    assert len(custom_nodes) == 1
    assert len(pad_nodes) == 1
    assert len(slice_nodes) == 1
    assert custom_nodes[0].meta["val"][0].shape == torch.Size((1, 8, 8, 4))
    assert slice_nodes[0].meta["val"].shape == torch.Size((1, 3, 8, 8))


def test_rewrite_warp_downsample_allows_shared_flow() -> None:
    _ensure_warp_downsample_ops_registered()
    graph = torch.fx.Graph()
    image0 = graph.placeholder("image0")
    image1 = graph.placeholder("image1")
    flow = graph.placeholder("flow")
    image0.meta["val"] = torch.empty(1, 4, 16, 16, dtype=torch.int8)
    image1.meta["val"] = torch.empty(1, 4, 16, 16, dtype=torch.int8)
    flow.meta["val"] = torch.empty(1, 2, 16, 16, dtype=torch.int8)
    warp0 = graph.call_function(
        torch.ops.rife.warp_downsample2.default,
        (image0, flow),
    )
    warp0.meta["val"] = torch.empty(1, 4, 8, 8, dtype=torch.int8)
    warp1 = graph.call_function(
        torch.ops.rife.warp_downsample2.default,
        (image1, flow),
    )
    warp1.meta["val"] = torch.empty(1, 4, 8, 8, dtype=torch.int8)
    snorm_qparams = QuantArgs(1.0 / 127.0, 0, -127, 127, torch.int8)
    flow_qparams = QuantArgs(0.25, -3, -128, 127, torch.int8)
    warp0.meta["input_qparams"] = {0: snorm_qparams, 1: flow_qparams}
    warp0.meta["output_qparams"] = {0: snorm_qparams}
    warp1.meta["input_qparams"] = {0: snorm_qparams, 1: flow_qparams}
    warp1.meta["output_qparams"] = {0: snorm_qparams}
    graph.output((warp0, warp1))
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT")):
        result = _rewrite_warp_downsample(graph_module)

    assert result.modified
    custom_nodes = [
        node
        for node in result.graph_module.graph.nodes
        if node.target == exir_ops.backend.tosa.CUSTOM.default
    ]
    assert len(custom_nodes) == 2


def test_rewrite_warp_downsample_reuses_shared_c3_padding() -> None:
    _ensure_warp_downsample_ops_registered()
    graph = torch.fx.Graph()
    image = graph.placeholder("image")
    flow0 = graph.placeholder("flow0")
    flow1 = graph.placeholder("flow1")
    image.meta["val"] = torch.empty(1, 3, 16, 16, dtype=torch.int8)
    flow0.meta["val"] = torch.empty(1, 2, 16, 16, dtype=torch.int8)
    flow1.meta["val"] = torch.empty(1, 2, 16, 16, dtype=torch.int8)
    warp0 = graph.call_function(
        torch.ops.rife.warp_downsample2.default,
        (image, flow0),
    )
    warp0.meta["val"] = torch.empty(1, 3, 8, 8, dtype=torch.int8)
    warp1 = graph.call_function(
        torch.ops.rife.warp_downsample2.default,
        (image, flow1),
    )
    warp1.meta["val"] = torch.empty(1, 3, 8, 8, dtype=torch.int8)
    snorm_qparams = QuantArgs(1.0 / 127.0, 0, -127, 127, torch.int8)
    flow_qparams = QuantArgs(0.25, -3, -128, 127, torch.int8)
    warp0.meta["input_qparams"] = {0: snorm_qparams, 1: flow_qparams}
    warp0.meta["output_qparams"] = {0: snorm_qparams}
    warp1.meta["input_qparams"] = {0: snorm_qparams, 1: flow_qparams}
    warp1.meta["output_qparams"] = {0: snorm_qparams}
    graph.output((warp0, warp1))
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT")):
        result = _rewrite_warp_downsample(graph_module)

    pad_nodes = [
        node
        for node in result.graph_module.graph.nodes
        if node.target == exir_ops.edge.aten.constant_pad_nd.default
    ]
    custom_nodes = [
        node
        for node in result.graph_module.graph.nodes
        if node.target == exir_ops.backend.tosa.CUSTOM.default
    ]
    assert len(pad_nodes) == 1
    assert len(custom_nodes) == 2


def test_rewrite_warp_downsample_does_not_reuse_later_padding() -> None:
    _ensure_warp_downsample_ops_registered()
    graph = torch.fx.Graph()
    image = graph.placeholder("image")
    flow = graph.placeholder("flow")
    image.meta["val"] = torch.empty(1, 3, 16, 16, dtype=torch.int8)
    flow.meta["val"] = torch.empty(1, 2, 16, 16, dtype=torch.int8)
    warp = graph.call_function(
        torch.ops.rife.warp_downsample2.default,
        (image, flow),
    )
    warp.meta["val"] = torch.empty(1, 3, 8, 8, dtype=torch.int8)
    snorm_qparams = QuantArgs(1.0 / 127.0, 0, -127, 127, torch.int8)
    flow_qparams = QuantArgs(0.25, -3, -128, 127, torch.int8)
    warp.meta["input_qparams"] = {0: snorm_qparams, 1: flow_qparams}
    warp.meta["output_qparams"] = {0: snorm_qparams}
    later_pad = graph.call_function(
        exir_ops.edge.aten.constant_pad_nd.default,
        (image, [0, 0, 0, 0, 0, 1], 0),
    )
    later_pad.meta["val"] = torch.empty(1, 4, 16, 16, dtype=torch.int8)
    graph.output((warp, later_pad))
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT")):
        result = _rewrite_warp_downsample(graph_module)

    pad_nodes = [
        node
        for node in result.graph_module.graph.nodes
        if node.target == exir_ops.edge.aten.constant_pad_nd.default
    ]
    assert len(pad_nodes) == 2
    result.graph_module.graph.lint()


def test_rewrite_warp_downsample_folds_flow_slice_into_channel_offset(
    monkeypatch,
) -> None:
    captured_source = {}

    def _capture_shader(source: str) -> str:
        captured_source["source"] = source
        return "compiled_shader"

    monkeypatch.setattr(
        warp_downsample_shaders,
        "_compile_shader_source",
        _capture_shader,
    )
    _ensure_warp_downsample_ops_registered()
    graph = torch.fx.Graph()
    image = graph.placeholder("image")
    flow = graph.placeholder("flow")
    image.meta["val"] = torch.empty(1, 4, 16, 16, dtype=torch.int8)
    flow.meta["val"] = torch.empty(1, 4, 16, 16, dtype=torch.int8)
    sliced_flow = graph.call_function(
        exir_ops.edge.aten.slice_copy.Tensor,
        (flow, 1, 2, 4),
    )
    sliced_flow.meta["val"] = torch.empty(1, 2, 16, 16, dtype=torch.int8)
    warp = graph.call_function(
        torch.ops.rife.warp_downsample2.default,
        (image, sliced_flow),
    )
    warp.meta["val"] = torch.empty(1, 4, 8, 8, dtype=torch.int8)
    snorm_qparams = QuantArgs(1.0 / 127.0, 0, -127, 127, torch.int8)
    flow_qparams = QuantArgs(0.25, -3, -128, 127, torch.int8)
    warp.meta["input_qparams"] = {0: snorm_qparams, 1: flow_qparams}
    warp.meta["output_qparams"] = {0: snorm_qparams}
    graph.output(warp)
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT")):
        result = _rewrite_warp_downsample(graph_module)

    custom_nodes = [
        node
        for node in result.graph_module.graph.nodes
        if node.target == exir_ops.backend.tosa.CUSTOM.default
    ]
    assert len(custom_nodes) == 1
    custom_node = custom_nodes[0]
    _, custom_flow = custom_node.args[0]
    assert custom_flow.name == "flow"
    assert all(
        node.target != exir_ops.edge.aten.slice_copy.Tensor
        for node in result.graph_module.graph.nodes
    )
    assert "kFlowChannelOffset = 2u" in captured_source["source"]
    assert "uint[](0u, channel, uint(p.y), uint(p.x))" in captured_source["source"]
    payload = json.loads(bytes(custom_node.kwargs["implementation_attrs"]).decode())
    assert payload["input_1_vkformat"] == "VK_FORMAT_R8_SINT"


def test_rewrite_warp_downsample_uses_int8_flow_payload(monkeypatch) -> None:
    monkeypatch.setattr(
        warp_downsample_shaders,
        "_compile_shader_source",
        lambda source: "compiled_shader",
    )
    _ensure_warp_downsample_ops_registered()
    graph = torch.fx.Graph()
    image = graph.placeholder("image")
    flow = graph.placeholder("flow")
    image.meta["val"] = torch.empty(1, 4, 16, 16, dtype=torch.int8)
    flow.meta["val"] = torch.empty(1, 2, 16, 16, dtype=torch.int8)
    warp = graph.call_function(
        torch.ops.rife.warp_downsample2.default,
        (image, flow),
    )
    warp.meta["val"] = torch.empty(1, 4, 8, 8, dtype=torch.int8)
    snorm_qparams = QuantArgs(1.0 / 127.0, 0, -127, 127, torch.int8)
    flow_qparams = QuantArgs(0.25, -3, -128, 127, torch.int8)
    warp.meta["input_qparams"] = {0: snorm_qparams, 1: flow_qparams}
    warp.meta["output_qparams"] = {0: snorm_qparams}
    graph.output(warp)
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+INT")):
        result = _rewrite_warp_downsample(graph_module)

    custom_nodes = [
        node
        for node in result.graph_module.graph.nodes
        if node.target == exir_ops.backend.tosa.CUSTOM.default
    ]
    assert len(custom_nodes) == 1
    custom_node = custom_nodes[0]
    payload = json.loads(bytes(custom_node.kwargs["implementation_attrs"]).decode())
    assert payload["input_1_vkformat"] == "VK_FORMAT_R8_SINT"
    assert payload["shader_code"] == "compiled_shader"
    assert set(custom_node.meta["input_qparams"]) == {0, 1}


def _center_2x2_reference(image: torch.Tensor, scale: int) -> torch.Tensor:
    offset0 = scale // 2 - 1
    offset1 = scale // 2
    output_h = image.shape[2] // scale
    output_w = image.shape[3] // scale
    result = torch.empty(image.shape[0], image.shape[1], output_h, output_w)
    for y in range(output_h):
        for x in range(output_w):
            base_y = y * scale
            base_x = x * scale
            result[:, :, y, x] = 0.25 * (
                image[:, :, base_y + offset0, base_x + offset0]
                + image[:, :, base_y + offset0, base_x + offset1]
                + image[:, :, base_y + offset1, base_x + offset0]
                + image[:, :, base_y + offset1, base_x + offset1]
            )
    return result


def _full_window_avg_reference(image: torch.Tensor, scale: int) -> torch.Tensor:
    return torch.nn.functional.avg_pool2d(image, kernel_size=scale, stride=scale)


@pytest.mark.parametrize("scale", (4, 8))
def test_warp_downsample_reference_uses_center_2x2_not_full_window_average(
    scale: int,
) -> None:
    side = scale * 2
    image = torch.arange(float(side * side)).square().reshape(1, 1, side, side)

    reference = _center_2x2_reference(image, scale)

    offset0 = scale // 2 - 1
    offset1 = scale // 2
    expected = torch.empty_like(reference)
    for y in range(2):
        for x in range(2):
            base_y = y * scale
            base_x = x * scale
            expected[:, :, y, x] = torch.tensor(
                [
                    [
                        0.25
                        * (
                            image[0, 0, base_y + offset0, base_x + offset0]
                            + image[0, 0, base_y + offset0, base_x + offset1]
                            + image[0, 0, base_y + offset1, base_x + offset0]
                            + image[0, 0, base_y + offset1, base_x + offset1]
                        )
                    ]
                ]
            )

    assert torch.equal(reference, expected)
    assert not torch.equal(reference, _full_window_avg_reference(image, scale))


def _normalized_boundary_graph(
    *, shared_source: bool = False, shared_permute: bool = False
) -> tuple[torch.fx.GraphModule, torch.fx.Node, torch.fx.Node]:
    graph = torch.fx.Graph()
    source = graph.placeholder("source")
    source.meta["val"] = torch.empty(1, 16, 16, 4)
    inverse = graph.call_function(
        exir_ops.edge.aten.permute_copy.default,
        (source, list(NHWC_INVERSE_ORDER)),
    )
    inverse.meta["val"] = torch.empty(1, 4, 16, 16)
    if shared_source:
        extra = graph.call_function(
            exir_ops.edge.aten.alias_copy.default,
            (source,),
        )
        extra.meta["val"] = source.meta["val"]
    if shared_permute:
        extra = graph.call_function(
            exir_ops.edge.aten.alias_copy.default,
            (inverse,),
        )
        extra.meta["val"] = inverse.meta["val"]
    graph.output(inverse)
    return torch.fx.GraphModule(torch.nn.Module(), graph), source, inverse


def test_restore_normalized_boundary_input_keeps_exclusive_boundary() -> None:
    _, source, inverse = _normalized_boundary_graph()

    restored = _restore_normalized_boundary_input(inverse)

    assert restored is source
    assert tuple(source.meta["val"].shape) == (1, 4, 16, 16)


@pytest.mark.parametrize("shared_source,shared_permute", ((True, False), (False, True)))
def test_restore_normalized_boundary_input_skips_shared_boundary(
    shared_source: bool, shared_permute: bool
) -> None:
    _, source, inverse = _normalized_boundary_graph(
        shared_source=shared_source, shared_permute=shared_permute
    )

    restored = _restore_normalized_boundary_input(inverse)

    assert restored is inverse
    assert tuple(source.meta["val"].shape) == (1, 16, 16, 4)
