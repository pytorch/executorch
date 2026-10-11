# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import base64
import shutil

import executorch.backends.arm.tosa.dialect  # noqa: F401
import executorch.extension.llm.custom_ops.op_fallback  # noqa: F401
import pytest
import torch
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.backends.arm.vgf import VgfCompileSpec
from executorch.backends.arm.vgf._passes import RewriteCustomSDPAToTosaCustomPass
from executorch.backends.arm.vgf.backend import VgfBackend
from executorch.backends.arm.vgf.partitioner import VgfPartitioner
from executorch.backends.arm.vgf.tosa_custom_sdpa import (
    CUSTOM_SDPA_DOMAIN_NAME,
    custom_sdpa_operator_name,
    decode_payload,
    ensure_custom_sdpa_registered,
)
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export


class _CustomSDPAModule(torch.nn.Module):
    def __init__(self, *, is_causal: bool = True, scale: float | None = None) -> None:
        super().__init__()
        self.is_causal = is_causal
        self.scale = scale

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        q_bshd = q.transpose(1, 2).contiguous()
        k_bshd = k.transpose(1, 2).contiguous()
        v_bshd = v.transpose(1, 2).contiguous()
        output = torch.ops.llama.custom_sdpa.default(
            q_bshd,
            k_bshd,
            v_bshd,
            0,
            None,
            0.0,
            self.is_causal,
            self.scale,
        )
        return output.transpose(1, 2).contiguous()


class _CustomSDPABshdOutputModule(torch.nn.Module):
    def __init__(self, *, is_causal: bool = True, scale: float | None = None) -> None:
        super().__init__()
        self.is_causal = is_causal
        self.scale = scale

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        q_bshd = q.transpose(1, 2).contiguous()
        k_bshd = k.transpose(1, 2).contiguous()
        v_bshd = v.transpose(1, 2).contiguous()
        return torch.ops.llama.custom_sdpa.default(
            q_bshd,
            k_bshd,
            v_bshd,
            0,
            None,
            0.0,
            self.is_causal,
            self.scale,
        )


class _CustomSDPABshdInputsBhdsOutputModule(torch.nn.Module):
    def __init__(self, *, is_causal: bool = True, scale: float | None = None) -> None:
        super().__init__()
        self.is_causal = is_causal
        self.scale = scale

    def forward(
        self, q_bshd: torch.Tensor, k_bshd: torch.Tensor, v_bshd: torch.Tensor
    ) -> torch.Tensor:
        output = torch.ops.llama.custom_sdpa.default(
            q_bshd,
            k_bshd,
            v_bshd,
            0,
            None,
            0.0,
            self.is_causal,
            self.scale,
        )
        return output.transpose(1, 2).contiguous()


class _AddModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


def _to_edge_custom_sdpa(
    query_shape: tuple[int, ...],
    key_shape: tuple[int, ...],
    value_shape: tuple[int, ...],
    *,
    is_causal: bool = True,
    scale: float | None = None,
    dtype: torch.dtype = torch.float32,
):
    ensure_custom_sdpa_registered()
    q = torch.rand(query_shape, dtype=dtype)
    k = torch.rand(key_shape, dtype=dtype)
    v = torch.rand(value_shape, dtype=dtype)
    module = _CustomSDPAModule(is_causal=is_causal, scale=scale)
    return to_edge(export(module, (q, k, v), strict=True))


def _to_edge_custom_sdpa_bshd_inputs(
    query_shape: tuple[int, ...],
    key_shape: tuple[int, ...],
    value_shape: tuple[int, ...],
    *,
    is_causal: bool = True,
    scale: float | None = None,
    dtype: torch.dtype = torch.float32,
):
    ensure_custom_sdpa_registered()
    q = torch.rand(query_shape, dtype=dtype)
    k = torch.rand(key_shape, dtype=dtype)
    v = torch.rand(value_shape, dtype=dtype)
    module = _CustomSDPABshdInputsBhdsOutputModule(is_causal=is_causal, scale=scale)
    return to_edge(export(module, (q, k, v), strict=True))


def _output_shape(edge_model) -> tuple[int, ...]:
    output_node = next(
        node
        for node in edge_model.exported_program().graph.nodes
        if node.op == "output"
    )
    result_node = output_node.args[0][0]
    assert isinstance(result_node, torch.fx.Node)
    return tuple(result_node.meta["val"].shape)


def test_rewrite_custom_sdpa_to_tosa_custom_rewrites_decode_gqa_path():
    if shutil.which("glslc") is None:
        pytest.skip("glslc not found")
    edge_model = _to_edge_custom_sdpa(
        (1, 32, 1, 128),
        (1, 8, 512, 128),
        (1, 8, 512, 128),
        scale=0.125,
    )

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP")):
        transformed = edge_model.transform([RewriteCustomSDPAToTosaCustomPass()])

    nodes = list(transformed.exported_program().graph.nodes)
    assert not any(
        node.target == exir_ops.edge.llama.custom_sdpa.default for node in nodes
    )

    custom_node = next(
        node for node in nodes if node.target == exir_ops.backend.tosa.CUSTOM.default
    )
    assert custom_node.kwargs["operator_name"] == custom_sdpa_operator_name()
    assert custom_node.kwargs["domain_name"] == CUSTOM_SDPA_DOMAIN_NAME

    payload = decode_payload(custom_node.kwargs["implementation_attrs"])
    assert payload["entry_point"] == "main"
    assert payload["shader_language"] == "SPIR-V"
    assert base64.b64decode(payload["shader_code"])[:4] == b"\x03\x02\x23\x07"
    assert payload["layout"] == "BHSD"
    assert payload["query_shape"] == [1, 32, 1, 128]
    assert payload["key_shape"] == [1, 8, 512, 128]
    assert payload["value_shape"] == [1, 8, 512, 128]
    assert payload["start_pos"] == 0
    assert payload["dropout_p"] == 0.0
    assert payload["is_causal"] is True
    assert payload["scale"] == 0.125
    assert _output_shape(transformed) == (1, 32, 1, 128)
    assert payload["input_0_vkdescriptortype"] == "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER"
    assert payload["input_1_vkdescriptortype"] == "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER"
    assert payload["input_2_vkdescriptortype"] == "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER"
    assert payload["output_0_vkdescriptortype"] == "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER"
    assert payload["input_0_binding"] == 0
    assert payload["input_1_binding"] == 1
    assert payload["input_2_binding"] == 2
    assert payload["output_0_binding"] == 3

    custom_index = nodes.index(custom_node)
    adjacent_targets = {
        str(node.target)
        for node in nodes[max(0, custom_index - 3) : custom_index + 4]
        if node.op == "call_function"
    }
    assert (
        "executorch_exir_dialects_edge__ops_aten_transpose_int" not in adjacent_targets
    )
    assert (
        "executorch_exir_dialects_edge__ops_aten_contiguous_default"
        not in adjacent_targets
    )


def test_rewrite_custom_sdpa_to_tosa_custom_keeps_bshd_output_when_no_restore():
    ensure_custom_sdpa_registered()
    q = torch.rand(1, 32, 1, 128)
    k = torch.rand(1, 8, 512, 128)
    v = torch.rand(1, 8, 512, 128)
    edge_model = to_edge(export(_CustomSDPABshdOutputModule(), (q, k, v), strict=True))

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP")):
        transformed = edge_model.transform([RewriteCustomSDPAToTosaCustomPass()])

    nodes = list(transformed.exported_program().graph.nodes)
    custom_node = next(
        node for node in nodes if node.target == exir_ops.backend.tosa.CUSTOM.default
    )
    payload = decode_payload(custom_node.kwargs["implementation_attrs"])
    assert payload["layout"] == "BSHD"
    assert _output_shape(transformed) == (1, 1, 32, 128)


def test_rewrite_custom_sdpa_to_tosa_custom_keeps_restore_when_layout_is_bshd():
    edge_model = _to_edge_custom_sdpa_bshd_inputs(
        (1, 1, 32, 128),
        (1, 512, 8, 128),
        (1, 512, 8, 128),
    )

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP")):
        transformed = edge_model.transform([RewriteCustomSDPAToTosaCustomPass()])

    nodes = list(transformed.exported_program().graph.nodes)
    custom_node = next(
        node for node in nodes if node.target == exir_ops.backend.tosa.CUSTOM.default
    )
    payload = decode_payload(custom_node.kwargs["implementation_attrs"])
    assert payload["layout"] == "BSHD"
    assert _output_shape(transformed) == (1, 32, 1, 128)


def test_rewrite_custom_sdpa_to_tosa_custom_skips_non_sdpa_graphs_without_registration(
    monkeypatch: pytest.MonkeyPatch,
):
    def _fail() -> None:
        raise AssertionError("custom SDPA registration should not be reached")

    monkeypatch.setattr(
        "executorch.backends.arm.vgf._passes.rewrite_custom_sdpa_to_tosa_custom."
        "ensure_custom_sdpa_registered",
        _fail,
    )

    x = torch.rand(1, 4)
    y = torch.rand(1, 4)
    edge_model = to_edge(export(_AddModule(), (x, y), strict=True))

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP")):
        transformed = edge_model.transform([RewriteCustomSDPAToTosaCustomPass()])

    assert transformed.exported_program().graph_module.code == (
        edge_model.exported_program().graph_module.code
    )


def test_rewrite_custom_sdpa_to_tosa_custom_skips_prefill_like_query_length():
    if shutil.which("glslc") is None:
        pytest.skip("glslc not found")
    edge_model = _to_edge_custom_sdpa(
        (1, 32, 8, 128),
        (1, 8, 512, 128),
        (1, 8, 512, 128),
    )

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP")):
        transformed = edge_model.transform([RewriteCustomSDPAToTosaCustomPass()])

    nodes = list(transformed.exported_program().graph.nodes)
    assert any(node.target == exir_ops.edge.llama.custom_sdpa.default for node in nodes)
    assert not any(
        node.target == exir_ops.backend.tosa.CUSTOM.default for node in nodes
    )


def test_rewrite_custom_sdpa_to_tosa_custom_skips_non_causal_path():
    if shutil.which("glslc") is None:
        pytest.skip("glslc not found")
    edge_model = _to_edge_custom_sdpa(
        (1, 32, 1, 128),
        (1, 8, 512, 128),
        (1, 8, 512, 128),
        is_causal=False,
    )

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP")):
        transformed = edge_model.transform([RewriteCustomSDPAToTosaCustomPass()])

    nodes = list(transformed.exported_program().graph.nodes)
    assert any(node.target == exir_ops.edge.llama.custom_sdpa.default for node in nodes)
    assert not any(
        node.target == exir_ops.backend.tosa.CUSTOM.default for node in nodes
    )


def test_rewrite_custom_sdpa_to_tosa_custom_skips_non_float32_path():
    if shutil.which("glslc") is None:
        pytest.skip("glslc not found")
    edge_model = _to_edge_custom_sdpa(
        (1, 32, 1, 128),
        (1, 8, 512, 128),
        (1, 8, 512, 128),
        dtype=torch.float16,
    )

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP")):
        transformed = edge_model.transform([RewriteCustomSDPAToTosaCustomPass()])

    nodes = list(transformed.exported_program().graph.nodes)
    assert any(node.target == exir_ops.edge.llama.custom_sdpa.default for node in nodes)
    assert not any(
        node.target == exir_ops.backend.tosa.CUSTOM.default for node in nodes
    )


def test_rewrite_custom_sdpa_to_tosa_custom_rejects_non_concrete_shapes():
    class _ShapeLike:
        def __init__(self, shape: tuple[object, ...]) -> None:
            self.shape = shape

        def dim(self) -> int:
            return len(self.shape)

    symbolic_dim = object()
    assert not RewriteCustomSDPAToTosaCustomPass._supports_decode_shape(
        _ShapeLike((1, 32, symbolic_dim, 128)),
        _ShapeLike((1, 8, 512, 128)),
        _ShapeLike((1, 8, 512, 128)),
        layout="BHSD",
    )


def test_vgf_partitioner_delegates_supported_custom_sdpa_decode_path():
    edge_model = _to_edge_custom_sdpa(
        (1, 32, 1, 128),
        (1, 8, 512, 128),
        (1, 8, 512, 128),
    )

    partition_result = VgfPartitioner(VgfCompileSpec("TOSA-1.0+FP")).partition(
        edge_model.exported_program()
    )
    sdpa_node = next(
        node
        for node in partition_result.tagged_exported_program.graph_module.graph.nodes
        if node.target == exir_ops.edge.llama.custom_sdpa.default
    )

    assert sdpa_node.meta["delegation_tag"] in partition_result.partition_tags


def test_vgf_partitioner_rejects_unsupported_custom_sdpa_prefill_path():
    edge_model = _to_edge_custom_sdpa(
        (1, 32, 8, 128),
        (1, 8, 512, 128),
        (1, 8, 512, 128),
    )

    partition_result = VgfPartitioner(VgfCompileSpec("TOSA-1.0+FP")).partition(
        edge_model.exported_program()
    )
    sdpa_node = next(
        node
        for node in partition_result.tagged_exported_program.graph_module.graph.nodes
        if node.target == exir_ops.edge.llama.custom_sdpa.default
    )

    assert "delegation_tag" not in sdpa_node.meta


def test_vgf_partitioner_rejects_custom_sdpa_for_int_only_profile():
    edge_model = _to_edge_custom_sdpa(
        (1, 32, 1, 128),
        (1, 8, 512, 128),
        (1, 8, 512, 128),
    )

    partition_result = VgfPartitioner(VgfCompileSpec("TOSA-1.0+INT")).partition(
        edge_model.exported_program()
    )
    sdpa_node = next(
        node
        for node in partition_result.tagged_exported_program.graph_module.graph.nodes
        if node.target == exir_ops.edge.llama.custom_sdpa.default
    )

    assert "delegation_tag" not in sdpa_node.meta


def test_vgf_backend_preprocess_rewrites_delegated_custom_sdpa_decode_path(
    monkeypatch: pytest.MonkeyPatch,
):
    if shutil.which("glslc") is None:
        pytest.skip("glslc not found")
    edge_model = _to_edge_custom_sdpa(
        (1, 32, 1, 128),
        (1, 8, 512, 128),
        (1, 8, 512, 128),
    )

    partition_result = VgfPartitioner(VgfCompileSpec("TOSA-1.0+FP")).partition(
        edge_model.exported_program()
    )

    monkeypatch.setattr(
        VgfBackend,
        "_compile_tosa_flatbuffer",
        staticmethod(lambda *args, **kwargs: b"vgf"),
    )

    result = VgfBackend.preprocess(
        partition_result.tagged_exported_program,
        VgfCompileSpec("TOSA-1.0+FP")._to_list(),
    )

    assert result.processed_bytes == b"vgf"

    nodes = list(partition_result.tagged_exported_program.graph_module.graph.nodes)
    assert not any(
        node.target == exir_ops.edge.llama.custom_sdpa.default for node in nodes
    )

    custom_node = next(
        node for node in nodes if node.target == exir_ops.backend.tosa.CUSTOM.default
    )
    payload = decode_payload(custom_node.kwargs["implementation_attrs"])
    assert payload["layout"] == "BHSD"
    assert custom_node.kwargs["domain_name"] == CUSTOM_SDPA_DOMAIN_NAME
