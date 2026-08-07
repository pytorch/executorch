# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Gemma 4-specific, instance-scoped WebGPU partitioning."""

from functools import lru_cache
from typing import Callable, List, Optional, Tuple

import executorch.backends.vulkan.patterns as vk_patterns
import torch

from executorch.backends.vulkan.op_registry import get_op_features, OpFeatures, OpKey
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.backends.vulkan.patterns.rope_hf import (
    create_hf_rotary_emb_single_custom_op,
    HfRotaryEmbeddingSinglePattern,
)
from executorch.exir import EdgeCompileConfig, ExportedProgram, to_edge
from executorch.exir.backend.partitioner import Partitioner, PartitionResult
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult

from .webgpu_artifact_manifest import WEBGPU_EXTERNAL_CONSTANTS_MAX_DATA_BYTES

_EXPECTED_GEMMA4_SDPA_COUNT = 35
_EXPECTED_SINGLE_HF_ROPE_COUNT = 20


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


class Gemma4WebGPUPartitioner(Partitioner):
    """Vulkan serialization restricted to Gemma 4 WebGPU capabilities."""

    def __init__(self, text_quantize: str) -> None:
        if "emb8" in text_quantize:
            raise ValueError(
                "WebGPU cannot delegate emb8; use emb4 (for example, 8da4w+emb4)"
            )
        self._inner = VulkanPartitioner(
            {
                "external_constants_max_data_bytes": (
                    WEBGPU_EXTERNAL_CONSTANTS_MAX_DATA_BYTES
                ),
                "require_dynamic_shapes": True,
                "skip_bool_tensors": True,
            },
            operator_allowlist=_webgpu_allowlist(),
            extra_op_features=_extra_op_features(),
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


def build_webgpu_partitioner(text_quantize: str) -> Gemma4WebGPUPartitioner:
    return Gemma4WebGPUPartitioner(text_quantize)


def build_webgpu_transform_passes() -> List[ExportedProgramPassBase]:
    """Edge transform passes the WebGPU text decoder must run pre-partition."""
    return [_Gemma4WebGPURewritePass()]
