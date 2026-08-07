# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Gemma 4-specific, instance-scoped WebGPU partitioning."""

import operator
from collections.abc import Mapping
from functools import lru_cache
from typing import Any, Callable, List, Literal, Optional, Tuple

import executorch.backends.vulkan.custom_ops_lib  # noqa: F401
import executorch.backends.vulkan.patterns as vk_patterns
import executorch.backends.vulkan.utils as vk_utils
import torch

from executorch.backends.vulkan.op_registry import get_op_features, OpFeatures, OpKey
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.backends.vulkan.patterns.rope_hf import (
    create_hf_rotary_emb_single_custom_op,
    HfRotaryEmbeddingSinglePattern,
)
from executorch.examples.models.gemma4.mtp_qat_contract import (
    OFFICIAL_QAT_CENTROID_TOP_K,
    OFFICIAL_QAT_NUM_CENTROIDS,
    OFFICIAL_QAT_SELECTED_TOKEN_COUNT,
    OFFICIAL_QAT_TOKENS_PER_CENTROID,
    validate_qat_token_ordering,
)
from executorch.exir import EdgeCompileConfig, ExportedProgram, to_edge
from executorch.exir.backend.partitioner import Partitioner, PartitionResult
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult

from .webgpu_artifact_manifest import WEBGPU_EXTERNAL_CONSTANTS_MAX_DATA_BYTES

_EXPECTED_GEMMA4_SDPA_COUNT = 35
_EXPECTED_SINGLE_HF_ROPE_COUNT = 20
_EXPECTED_ASSISTANT_SDPA_COUNT = 8
_EXPECTED_ASSISTANT_SINGLE_HF_ROPE_COUNT = 8
_OFFICIAL_TOPK_INPUT_SHAPE: tuple[int, ...] = (1, 1, OFFICIAL_QAT_NUM_CENTROIDS)
_OFFICIAL_TOPK_OUTPUT_SHAPE: tuple[int, ...] = (
    1,
    1,
    OFFICIAL_QAT_CENTROID_TOP_K,
)
_OFFICIAL_SCATTER_OUTPUT_SHAPE: tuple[int, ...] = (
    1,
    1,
    OFFICIAL_QAT_NUM_CENTROIDS * OFFICIAL_QAT_TOKENS_PER_CENTROID,
)
_OFFICIAL_SCATTER_UPDATE_SHAPE: tuple[int, ...] = (
    1,
    1,
    OFFICIAL_QAT_SELECTED_TOKEN_COUNT,
)


def _single_hf_rope_features() -> OpFeatures:
    return get_op_features(exir_ops.edge.et_vk.apply_rotary_emb_hf.default)


def _extra_op_features() -> dict[OpKey, OpFeatures]:
    return {
        exir_ops.edge.et_vk.apply_rotary_emb_hf_single.default: (
            _single_hf_rope_features()
        ),
        exir_ops.edge.et_vk.gemma4_sdpa.default: get_op_features("llama::custom_sdpa"),
    }


def _webgpu_allowlist() -> list[OpKey]:
    return [
        exir_ops.edge.aten.add.Tensor,
        exir_ops.edge.et_vk.rms_norm.default,
        exir_ops.edge.et_vk.apply_rotary_emb_hf.default,
        exir_ops.edge.et_vk.apply_rotary_emb_hf_single.default,
        exir_ops.edge.et_vk.gemma4_sdpa.default,
        exir_ops.edge.aten.mul.Tensor,
        exir_ops.edge.dim_order_ops._clone_dim_order.default,
        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
        exir_ops.edge.aten.view_copy.default,
        exir_ops.edge.aten.select_copy.int,
        exir_ops.edge.aten.sigmoid.default,
        exir_ops.edge.aten.gelu.default,
        exir_ops.edge.aten.clamp.default,
        exir_ops.edge.aten.div.Tensor,
        exir_ops.edge.aten.tanh.default,
        exir_ops.edge.aten.squeeze_copy.dims,
        exir_ops.edge.aten.unsqueeze_copy.default,
        exir_ops.edge.aten.slice_copy.Tensor,
        exir_ops.edge.aten.permute_copy.default,
        exir_ops.edge.aten.cat.default,
        exir_ops.edge.aten.argmax.default,
        exir_ops.edge.aten._assert_scalar.default,
        exir_ops.edge.aten.sym_constrain_range_for_size.default,
        exir_ops.edge.et_vk.select_as_symint.default,
    ]


@lru_cache(maxsize=1)
def _single_hf_rope_patterns() -> List[torch.fx.GraphModule]:
    x = torch.randn(1, 1, 4, 32, dtype=torch.float32)
    freqs_cos = torch.randn(1, 32, dtype=torch.float32)
    freqs_sin = torch.randn(1, 32, dtype=torch.float32)
    edge = to_edge(
        torch.export.export(
            HfRotaryEmbeddingSinglePattern(),
            (x, freqs_cos, freqs_sin),
            strict=True,
        ),
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    return [edge.exported_program().graph_module]


def _replace_single_hf_rope(exported_program: ExportedProgram) -> None:
    graph_module = exported_program.graph_module
    vk_patterns.create_replacement_for_pattern(
        exported_program,
        graph_module,
        _single_hf_rope_patterns(),
        create_hf_rotary_emb_single_custom_op,
    )
    replaced = sum(
        node.target == exir_ops.edge.et_vk.apply_rotary_emb_hf_single.default
        for node in graph_module.graph.nodes
    )
    if replaced != _EXPECTED_SINGLE_HF_ROPE_COUNT:
        raise ValueError(
            "Gemma4 WebGPU expected "
            f"{_EXPECTED_SINGLE_HF_ROPE_COUNT} single-HF-RoPE sites, got {replaced}"
        )


def _rank(node: object) -> Optional[int]:
    if not isinstance(node, torch.fx.Node):
        return None
    value = node.meta.get("val")
    return value.dim() if isinstance(value, torch.Tensor) else None


def _rewrite_gemma4_sdpa(exported_program: ExportedProgram) -> None:
    graph_module = exported_program.graph_module
    rewritten = 0
    for node in graph_module.graph.nodes:
        if node.target != exir_ops.edge.llama.custom_sdpa.default:
            continue
        if len(node.args) != 8 or node.kwargs:
            raise ValueError("Gemma4 custom SDPA must use the exact positional ABI")
        query, key, value, _start_pos, mask, dropout, is_causal, scale = node.args
        if (
            _rank(query) != 4
            or _rank(key) != 4
            or _rank(value) != 4
            or _rank(mask) != 2
            or dropout != 0.0
            or is_causal is not False
            or scale != 1.0
        ):
            raise ValueError("Gemma4 custom SDPA call is not WebGPU-compatible")
        node.target = exir_ops.edge.et_vk.gemma4_sdpa.default
        rewritten += 1

    if rewritten != _EXPECTED_GEMMA4_SDPA_COUNT:
        raise ValueError(
            "Gemma4 WebGPU expected "
            f"{_EXPECTED_GEMMA4_SDPA_COUNT} SDPA sites, got {rewritten}"
        )
    graph_module.recompile()


class _Gemma4WebGPURewritePass(ExportedProgramPassBase):
    """Plain-Gemma edge rewrites, applied before partitioning.

    `to_edge_transform_and_lower` hands partitioners a deep copy and asserts
    the returned graph is identical, so a partitioner cannot rewrite the graph.
    The rewrite set is fixed: single-HF-RoPE then Gemma SDPA. MTP extends the
    model-owned mechanism separately.
    """

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        _replace_single_hf_rope(exported_program)
        _rewrite_gemma4_sdpa(exported_program)
        return ExportedProgramPassResult(exported_program, True)


def _argument(node: torch.fx.Node, index: int, name: str, default: Any) -> Any:
    if len(node.args) > index:
        return node.args[index]
    return node.kwargs.get(name, default)


def _tensor_meta(value: object) -> torch.Tensor | None:
    if not isinstance(value, torch.fx.Node):
        return None
    tensor = value.meta.get("val")
    return tensor if isinstance(tensor, torch.Tensor) else None


def _tensor_shape(value: object) -> tuple[int, ...] | None:
    tensor = _tensor_meta(value)
    return tuple(tensor.shape) if tensor is not None else None


def _require_tensor(
    value: object,
    label: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.fx.Node:
    tensor = _tensor_meta(value)
    if not isinstance(value, torch.fx.Node) or tensor is None:
        raise ValueError(f"MTP residual {label} is not a tensor node")
    if tuple(tensor.shape) != shape or tensor.dtype != dtype:
        raise ValueError(
            f"MTP residual {label} mismatch: "
            f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}"
        )
    return value


def _semantic_call_users(node: torch.fx.Node) -> list[torch.fx.Node]:
    return [
        user
        for user in node.users
        if user.op == "call_function"
        and user.target != torch.ops.aten._assert_tensor_metadata.default
    ]


def _only_call_user(node: torch.fx.Node, target: object, label: str) -> torch.fx.Node:
    users = _semantic_call_users(node)
    if len(users) != 1 or users[0].target != target:
        raise ValueError(f"MTP residual {label} provenance mismatch")
    return users[0]


def _is_integer_tensor(value: torch.Tensor) -> bool:
    return value.dtype in (torch.int32, torch.int64)


def _is_official_qat_topk(node: torch.fx.Node) -> bool:
    input_value = _tensor_meta(node.args[0])
    outputs = node.meta.get("val")
    if (
        input_value is None
        or input_value.dtype != torch.float32
        or tuple(input_value.shape) != _OFFICIAL_TOPK_INPUT_SHAPE
        or not isinstance(outputs, (tuple, list))
        or len(outputs) != 2
        or not all(isinstance(value, torch.Tensor) for value in outputs)
    ):
        return False
    values, indices = outputs
    return bool(
        values.dtype == torch.float32
        and tuple(values.shape) == _OFFICIAL_TOPK_OUTPUT_SHAPE
        and _is_integer_tensor(indices)
        and tuple(indices.shape) == _OFFICIAL_TOPK_OUTPUT_SHAPE
        and _argument(node, 1, "k", None) == OFFICIAL_QAT_CENTROID_TOP_K
        and _argument(node, 2, "dim", -1) == -1
        and _argument(node, 3, "largest", True) is True
        and _argument(node, 4, "sorted", True) is True
    )


def _is_official_qat_unique_scatter(node: torch.fx.Node) -> bool:
    if len(node.args) < 4 or _argument(node, 1, "dim", None) != -1:
        return False
    output = _tensor_meta(node.args[0])
    index = _tensor_meta(node.args[2])
    source = _tensor_meta(node.args[3])
    result = node.meta.get("val")
    return bool(
        output is not None
        and index is not None
        and source is not None
        and isinstance(result, torch.Tensor)
        and output.dtype == torch.float32
        and tuple(output.shape) == _OFFICIAL_SCATTER_OUTPUT_SHAPE
        and _is_integer_tensor(index)
        and tuple(index.shape) == _OFFICIAL_SCATTER_UPDATE_SHAPE
        and source.dtype == torch.float32
        and tuple(source.shape) == _OFFICIAL_SCATTER_UPDATE_SHAPE
        and result.dtype == torch.float32
        and tuple(result.shape) == _OFFICIAL_SCATTER_OUTPUT_SHAPE
    )


def _lifted_tensor_value(
    program: torch.export.ExportedProgram, node: torch.fx.Node
) -> torch.Tensor | None:
    if node.op != "placeholder":
        return None
    signature = program.graph_signature
    target = (
        signature.inputs_to_parameters.get(node.name)
        or signature.inputs_to_buffers.get(node.name)
        or signature.inputs_to_lifted_tensor_constants.get(node.name)
    )
    if target is None:
        return None
    value = program.state_dict.get(target)
    if value is None:
        value = program.constants.get(target)
    return value if isinstance(value, torch.Tensor) else None


def _validate_ordering_constant(
    program: torch.export.ExportedProgram,
    node: torch.fx.Node,
    expected_ordering: torch.Tensor,
) -> None:
    ordering = _lifted_tensor_value(program, node)
    expected = (
        expected_ordering.detach()
        .to(torch.int64)
        .reshape(
            OFFICIAL_QAT_NUM_CENTROIDS,
            OFFICIAL_QAT_TOKENS_PER_CENTROID,
        )
    )
    if (
        ordering is None
        or ordering.dtype != torch.float32
        or tuple(ordering.shape) != tuple(expected.shape)
        or not torch.equal(ordering.detach().to(torch.int64).cpu(), expected.cpu())
    ):
        raise ValueError("MTP residual token-ordering identity mismatch")


def _validate_output_template(
    program: torch.export.ExportedProgram, node: torch.fx.Node
) -> None:
    template = _lifted_tensor_value(program, node)
    if (
        template is None
        or template.dtype != torch.float32
        or tuple(template.shape) != _OFFICIAL_SCATTER_OUTPUT_SHAPE
        or not bool(
            torch.all(template.detach().cpu() == torch.finfo(torch.float32).min)
        )
    ):
        raise ValueError("MTP residual output-template identity mismatch")


def _validate_selected_embedding_chain(
    convert: torch.fx.Node,
) -> tuple[torch.fx.Node, torch.fx.Node]:
    views = [
        user
        for user in _semantic_call_users(convert)
        if user.target == torch.ops.aten.view.default
    ]
    index_view = next(
        (user for user in views if _tensor_shape(user) == (1, 1, 4096)),
        None,
    )
    flat_view = next(
        (user for user in views if _tensor_shape(user) == (4096,)),
        None,
    )
    if index_view is None or flat_view is None or len(views) != 2:
        raise ValueError("MTP residual selected-token view topology mismatch")
    selected = _only_call_user(
        flat_view,
        torch.ops.quantized_decomposed.embedding_4bit.dtype,
        "selected-token embedding",
    )
    if (
        len(selected.args) != 6
        or selected.args[2] is not None
        or selected.args[3:5] != (-8, 7)
        or selected.args[5] is not flat_view
        or selected.kwargs.get("dtype") != torch.float32
    ):
        raise ValueError("MTP residual selected-token embedding ABI mismatch")
    _require_tensor(
        selected,
        "selected-token embedding",
        (OFFICIAL_QAT_SELECTED_TOKEN_COUNT, 256),
        torch.float32,
    )
    return index_view, selected


def _validate_scatter_source_chain(
    scatter: torch.fx.Node,
    selected: torch.fx.Node,
    topk_input: torch.fx.Node,
) -> None:
    source = _require_tensor(
        scatter.args[3],
        "scatter source",
        _OFFICIAL_SCATTER_UPDATE_SHAPE,
        torch.float32,
    )
    if source.target != torch.ops.aten.squeeze.dim or _semantic_call_users(source) != [
        scatter
    ]:
        raise ValueError("MTP residual scatter-source provenance mismatch")
    matmul = source.args[0] if source.args else None
    if (
        not isinstance(matmul, torch.fx.Node)
        or matmul.target != torch.ops.aten.matmul.default
    ):
        raise ValueError("MTP residual scatter-source matmul mismatch")
    selected_view = _only_call_user(
        selected, torch.ops.aten.view.default, "selected-token view"
    )
    selected_transpose = _only_call_user(
        selected_view, torch.ops.aten.transpose.int, "selected-token transpose"
    )
    if len(matmul.args) != 2 or matmul.args[1] is not selected_transpose:
        raise ValueError("MTP residual selected-token matmul linkage mismatch")
    query = matmul.args[0]
    if (
        not isinstance(query, torch.fx.Node)
        or query.target != torch.ops.aten.unsqueeze.default
        or not query.args
        or not topk_input.args
        or query.args[0] is not topk_input.args[0]
    ):
        raise ValueError("MTP residual score/source hidden-state linkage mismatch")


def _certify_scatter_chain(
    program: torch.export.ExportedProgram,
    scatter: torch.fx.Node,
    expected_ordering: torch.Tensor,
) -> tuple[torch.fx.Node, str, str]:
    if not _is_official_qat_unique_scatter(scatter):
        raise ValueError("MTP export found a non-official scatter contract")
    index = _require_tensor(
        scatter.args[2],
        "scatter index",
        _OFFICIAL_SCATTER_UPDATE_SHAPE,
        torch.int64,
    )
    if index.target != torch.ops.aten.view.default or not index.args:
        raise ValueError("MTP residual scatter-index view mismatch")
    convert = index.args[0]
    if (
        not isinstance(convert, torch.fx.Node)
        or convert.target != torch.ops.aten._to_copy.default
        or convert.kwargs != {"dtype": torch.int64}
        or not convert.args
    ):
        raise ValueError("MTP residual token-ordering conversion mismatch")
    embedding = _require_tensor(
        convert.args[0],
        "token-ordering embedding",
        (1, 1, 32, 128),
        torch.float32,
    )
    if embedding.target != torch.ops.aten.embedding.default or len(embedding.args) < 2:
        raise ValueError("MTP residual token-ordering embedding mismatch")
    ordering_node, projection = embedding.args[:2]
    if not isinstance(ordering_node, torch.fx.Node) or not isinstance(
        projection, torch.fx.Node
    ):
        raise ValueError("MTP residual ordering/projection is not a graph node")
    _validate_ordering_constant(program, ordering_node, expected_ordering)
    if projection.target is not operator.getitem or projection.args[1] != 1:
        raise ValueError("MTP residual top-k projection mismatch")
    topk = projection.args[0]
    if not isinstance(topk, torch.fx.Node) or not _is_official_qat_topk(topk):
        raise ValueError("MTP residual top-k contract mismatch")
    if _semantic_call_users(topk) != [projection]:
        raise ValueError("MTP residual top-k consumer mismatch")
    topk_input = topk.args[0]
    if (
        not isinstance(topk_input, torch.fx.Node)
        or topk_input.target != torch.ops.aten.linear.default
        or _semantic_call_users(topk_input) != [topk]
    ):
        raise ValueError("MTP residual top-k producer provenance mismatch")
    index_view, selected = _validate_selected_embedding_chain(convert)
    if index_view is not index:
        raise ValueError("MTP residual scatter-index identity mismatch")
    _validate_scatter_source_chain(scatter, selected, topk_input)
    template = scatter.args[0]
    if not isinstance(template, torch.fx.Node):
        raise ValueError("MTP residual output template is not a graph node")
    _validate_output_template(program, template)
    return topk, ordering_node.name, template.name


def mtp_extra_op_features() -> Mapping[OpKey, OpFeatures]:
    features = _extra_op_features()
    features.update(
        {
            exir_ops.edge.aten.topk.default: OpFeatures(
                inputs_dtypes=[vk_utils.FP_T],
                outputs_dtypes=[vk_utils.FP_T, vk_utils.INT_T],
                inputs_storage=[vk_utils.CONTIGUOUS_BUFFER],
                outputs_storage=[
                    vk_utils.CONTIGUOUS_BUFFER,
                    vk_utils.CONTIGUOUS_BUFFER,
                ],
                supports_resize=True,
                are_node_inputs_supported_fn=_is_official_qat_topk,
            ),
            exir_ops.edge.et_vk.scatter_src_unique.default: OpFeatures(
                inputs_dtypes=[
                    vk_utils.FP_T,
                    vk_utils.NONE_T,
                    vk_utils.INT_T,
                    vk_utils.FP_T,
                ],
                outputs_dtypes=[vk_utils.FP_T],
                inputs_storage=[
                    vk_utils.CONTIGUOUS_BUFFER,
                    vk_utils.NO_STORAGE,
                    vk_utils.CONTIGUOUS_BUFFER,
                    vk_utils.CONTIGUOUS_BUFFER,
                ],
                outputs_storage=[vk_utils.CONTIGUOUS_BUFFER],
                supports_resize=True,
                are_node_inputs_supported_fn=_is_official_qat_unique_scatter,
            ),
        }
    )
    return features


def rewrite_certified_unique_scatter(
    program: torch.export.ExportedProgram,
    token_ordering: torch.Tensor,
    *,
    expected_chains: int = 2,
) -> int:
    validate_qat_token_ordering(token_ordering)
    topk_nodes = [
        node
        for node in program.graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten.topk.default
    ]
    scatter_nodes = [
        node
        for node in program.graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten.scatter.src
    ]
    if (
        expected_chains <= 0
        or len(topk_nodes) != expected_chains
        or len(scatter_nodes) != expected_chains
    ):
        raise ValueError(
            "MTP export residual topology mismatch: "
            f"topk={len(topk_nodes)}, scatter={len(scatter_nodes)}, "
            f"expected={expected_chains}"
        )
    certified = [
        _certify_scatter_chain(program, node, token_ordering) for node in scatter_nodes
    ]
    if (
        {record[0] for record in certified} != set(topk_nodes)
        or len({record[1] for record in certified}) != 1
        or len({record[2] for record in certified}) != 1
    ):
        raise ValueError("MTP export residual-chain identity mismatch")
    for node in scatter_nodes:
        node.target = torch.ops.et_vk.scatter_src_unique.default
        node.meta["gemma4_mtp_unique_scatter_certified"] = True
    program.graph_module.recompile()
    return len(scatter_nodes)


def _node_has_module_fragment(node: torch.fx.Node, fragment: str) -> bool:
    stack = node.meta.get("nn_module_stack") or {}
    for entry in stack.values():
        path = entry[0] if isinstance(entry, tuple) and entry else str(entry)
        if fragment in str(path):
            return True
    return False


def _replace_mtp_single_hf_rope(exported_program: ExportedProgram) -> None:
    graph_module = exported_program.graph_module
    vk_patterns.create_replacement_for_pattern(
        exported_program,
        graph_module,
        _single_hf_rope_patterns(),
        create_hf_rotary_emb_single_custom_op,
    )
    nodes = [
        node
        for node in graph_module.graph.nodes
        if node.target == exir_ops.edge.et_vk.apply_rotary_emb_hf_single.default
    ]
    target_count = sum(_node_has_module_fragment(node, "target") for node in nodes)
    assistant_count = sum(
        _node_has_module_fragment(node, "assistant") for node in nodes
    )
    if (
        target_count != _EXPECTED_SINGLE_HF_ROPE_COUNT
        or assistant_count != _EXPECTED_ASSISTANT_SINGLE_HF_ROPE_COUNT
        or len(nodes) != target_count + assistant_count
    ):
        raise ValueError(
            "Gemma4 MTP single-HF-RoPE scope mismatch: "
            f"target={target_count}, assistant={assistant_count}, total={len(nodes)}"
        )


def _rewrite_mtp_sdpa(exported_program: ExportedProgram) -> None:
    graph_module = exported_program.graph_module
    target_nodes: list[torch.fx.Node] = []
    assistant_nodes: list[torch.fx.Node] = []
    unscoped_nodes: list[torch.fx.Node] = []
    for node in graph_module.graph.nodes:
        if node.target != exir_ops.edge.llama.custom_sdpa.default:
            continue
        if _node_has_module_fragment(node, "target"):
            target_nodes.append(node)
        elif _node_has_module_fragment(node, "assistant"):
            assistant_nodes.append(node)
        else:
            unscoped_nodes.append(node)
        if len(node.args) != 8 or node.kwargs:
            raise ValueError("Gemma4 MTP custom SDPA must use the positional ABI")
        query, key, value, _start_pos, mask, dropout, is_causal, scale = node.args
        if (
            _rank(query) != 4
            or _rank(key) != 4
            or _rank(value) != 4
            or _rank(mask) != 2
            or dropout != 0.0
            or is_causal is not False
            or scale != 1.0
        ):
            raise ValueError("Gemma4 MTP custom SDPA is not WebGPU-compatible")
    if (
        len(target_nodes) != _EXPECTED_GEMMA4_SDPA_COUNT
        or len(assistant_nodes) != _EXPECTED_ASSISTANT_SDPA_COUNT
        or unscoped_nodes
    ):
        raise ValueError(
            "Gemma4 MTP SDPA scope mismatch: "
            f"target={len(target_nodes)}, assistant={len(assistant_nodes)}, "
            f"unscoped={len(unscoped_nodes)}"
        )
    for node in [*target_nodes, *assistant_nodes]:
        node.target = exir_ops.edge.et_vk.gemma4_sdpa.default
    graph_module.recompile()


def _validate_mtp_edge_census(
    exported_program: ExportedProgram,
) -> dict[str, int]:
    counts = {
        "custom_scatter": 0,
        "gemma_sdpa": 0,
        "generic_scatter": 0,
        "legacy_custom_sdpa": 0,
        "topk": 0,
    }
    targets = {
        exir_ops.edge.et_vk.scatter_src_unique.default: "custom_scatter",
        exir_ops.edge.et_vk.gemma4_sdpa.default: "gemma_sdpa",
        exir_ops.edge.aten.scatter.src: "generic_scatter",
        exir_ops.edge.llama.custom_sdpa.default: "legacy_custom_sdpa",
        exir_ops.edge.aten.topk.default: "topk",
    }
    for node in exported_program.graph.nodes:
        label = targets.get(node.target)
        if label is not None:
            counts[label] += 1
    expected = {
        "custom_scatter": 2,
        "gemma_sdpa": 43,
        "generic_scatter": 0,
        "legacy_custom_sdpa": 0,
        "topk": 2,
    }
    if counts != expected:
        raise ValueError(f"Gemma4 MTP edge census mismatch: {counts}")
    return counts


class Gemma4MTPWebGPURewritePass(ExportedProgramPassBase):
    """Fixed MTP edge rewrites that run before metadata-only partitioning."""

    def __init__(self) -> None:
        super().__init__()
        self.census: Optional[dict[str, int]] = None

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        _replace_mtp_single_hf_rope(exported_program)
        _rewrite_mtp_sdpa(exported_program)
        self.census = _validate_mtp_edge_census(exported_program)
        exported_program.graph_module.meta["gemma4MTPEdgeCensus"] = dict(self.census)
        return ExportedProgramPassResult(exported_program, True)


def _mtp_webgpu_allowlist() -> list[OpKey]:
    additions = [
        exir_ops.edge.aten.bmm.default,
        exir_ops.edge.aten.embedding.default,
        exir_ops.edge.aten.mm.default,
        exir_ops.edge.aten.eq.Scalar,
        exir_ops.edge.aten.sub.Tensor,
        exir_ops.edge.aten.where.self,
        exir_ops.edge.aten.topk.default,
        exir_ops.edge.et_vk.scatter_src_unique.default,
    ]
    return list(dict.fromkeys([*_webgpu_allowlist(), *additions]))


class Gemma4WebGPUPartitioner(Partitioner):
    """Vulkan serialization restricted to Gemma 4 WebGPU capabilities."""

    def __init__(
        self,
        text_quantize: str,
        *,
        mode: Literal["plain", "mtp"] = "plain",
        compile_options: dict[str, Any] | None = None,
    ) -> None:
        if mode not in ("plain", "mtp"):
            raise ValueError(f"invalid Gemma4 WebGPU partitioner mode: {mode!r}")
        if "emb8" in text_quantize:
            raise ValueError(
                "WebGPU cannot delegate emb8; use emb4 (for example, 8da4w+emb4)"
            )
        if mode == "mtp" and text_quantize != "8da4w+emb4":
            raise ValueError("Gemma4 MTP WebGPU requires 8da4w+emb4")
        if mode == "plain" and compile_options is not None:
            raise ValueError("plain Gemma4 WebGPU does not accept option overrides")
        options = {
            "external_constants_max_data_bytes": (
                WEBGPU_EXTERNAL_CONSTANTS_MAX_DATA_BYTES
            ),
            "require_dynamic_shapes": True,
            "skip_bool_tensors": mode == "plain",
        }
        if mode == "mtp":
            options["alias_buffer_mutations"] = True
        if compile_options is not None:
            for key, value in compile_options.items():
                if key in options and value != options[key]:
                    raise ValueError(
                        f"Gemma4 MTP WebGPU cannot override {key}: {value!r}"
                    )
                options[key] = value
        self._inner = VulkanPartitioner(
            options,
            operator_allowlist=(
                _webgpu_allowlist() if mode == "plain" else _mtp_webgpu_allowlist()
            ),
            extra_op_features=(
                _extra_op_features() if mode == "plain" else mtp_extra_op_features()
            ),
        )

    def ops_to_not_decompose(self, ep: ExportedProgram) -> Tuple[
        List[torch._ops.OpOverload],
        Optional[Callable[[torch.fx.Node], bool]],
    ]:
        return self._inner.ops_to_not_decompose(ep)

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        # No graph mutation here: to_edge_transform_and_lower asserts the
        # partitioner returns an identical graph. The rewrites run in
        # Gemma4WebGPURewritePass, before partitioning.
        return self._inner.partition(exported_program)


def build_webgpu_partitioner(
    text_quantize: str,
    *,
    mode: Literal["plain", "mtp"] = "plain",
    compile_options: dict[str, Any] | None = None,
) -> Gemma4WebGPUPartitioner:
    return Gemma4WebGPUPartitioner(
        text_quantize,
        mode=mode,
        compile_options=compile_options,
    )


def build_webgpu_transform_passes(
    mode: Literal["plain", "mtp"] = "plain",
) -> List[ExportedProgramPassBase]:
    """Return fixed edge transforms for the requested Gemma4 WebGPU mode."""
    if mode == "plain":
        return [_Gemma4WebGPURewritePass()]
    if mode == "mtp":
        return [Gemma4MTPWebGPURewritePass()]
    raise ValueError(f"invalid Gemma4 WebGPU transform mode: {mode!r}")
