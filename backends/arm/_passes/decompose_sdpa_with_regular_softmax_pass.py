# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.transforms import decompose_sdpa
from executorch.exir.pass_base import ExportPass, PassResult


class DecomposeSDPAWithRegularSoftmaxPass(
    ArmPass, decompose_sdpa.DecomposeScaledDotProductAttention
):
    """Decompose eligible SDPA calls using regular softmax.

    Matches unmasked, noncausal, zero-dropout SDPA calls whose key sequence
    length is statically known to be nonzero. The matched form is conceptually::

        scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
            enable_gqa=enable_gqa,
        )

    Other SDPA options, such as ``scale`` and ``enable_gqa``, are preserved.

    The generated subgraph is approximately::

        scores = (query @ key.transpose(-2, -1)) * scale
        output = softmax(scores, dim=-1) @ value

    The standard SDPA decomposition initially generates ``_safe_softmax``.
    This pass replaces that operator with regular ``softmax`` only in the
    newly generated subgraph. The eligibility checks prevent masks or empty
    key sequences from producing all-negative-infinity score rows. They assume
    such rows are not produced by nonfinite inputs or numerical overflow.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call(
        self, graph_module: torch.fx.GraphModule, allow_non_fake_inputs: bool = True
    ) -> PassResult:
        graph = graph_module.graph
        modified = False
        for node in list(graph.nodes):
            if node.target != torch.ops.aten.scaled_dot_product_attention.default:
                continue
            if not self._is_auto_guard_removal_candidate(node):
                continue

            existing_nodes = set(graph.nodes)
            super()._decompose_sdpa_node(graph_module, node, allow_non_fake_inputs)
            self._remove_safe_softmax_guard(graph, existing_nodes)
            modified = True

        if modified:
            graph.eliminate_dead_code()
            graph_module.recompile()
        return PassResult(graph_module, modified)

    @classmethod
    def _is_auto_guard_removal_candidate(cls, node: torch.fx.Node) -> bool:
        """Return true when SDPA meets automatic removal constraints.

        These structural checks exclude fully masked rows. They assume scores
        do not become all ``-inf`` through nonfinite inputs or overflow.

        """
        canonical_args, _, _ = cls._canonicalize_sdpa_call(node)
        _, key, _, attn_mask, dropout_p, is_causal, _ = canonical_args

        if attn_mask is not None:
            return False
        if cls._extract_arg_value(is_causal) is not False:
            return False
        if cls._extract_arg_value(dropout_p) != 0.0:
            return False
        return cls._has_nonzero_key_sequence_length(key)

    @staticmethod
    def _has_nonzero_key_sequence_length(key: object) -> bool:
        if not isinstance(key, torch.fx.Node):
            return False

        val = key.meta.get("val")
        shape = getattr(val, "shape", None)
        if shape is None or len(shape) < 2:
            return False

        key_sequence_length = shape[-2]
        return isinstance(key_sequence_length, int) and key_sequence_length > 0

    @staticmethod
    def _remove_safe_softmax_guard(
        graph: torch.fx.Graph, existing_nodes: set[torch.fx.Node]
    ) -> None:
        for decomposed_node in graph.nodes:
            if decomposed_node in existing_nodes:
                continue
            if decomposed_node.target == torch.ops.aten._safe_softmax.default:
                decomposed_node.target = torch.ops.aten.softmax.int
