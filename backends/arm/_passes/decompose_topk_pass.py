# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Share the static TopK contract and lower it using ARGMAX and masking."""

import operator
from dataclasses import dataclass
from typing import cast

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    get_first_fake_tensor,
)
from executorch.backends.arm._passes.canonicalize_gather_pass import (
    CanonicalizeGatherPass,
)
from executorch.backends.arm._passes.prepare_gather_indices_pass import (
    PrepareGatherIndicesPass,
)
from executorch.backends.arm.tosa.specification import TosaSpecification
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

TOPK_OPS = (torch.ops.aten.topk.default, exir_ops.edge.aten.topk.default)
_CAST_OPS = (
    torch.ops.dim_order_ops._to_dim_order_copy.default,
    exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
)


@dataclass(frozen=True)
class StaticTopKConfig:
    """Describe a supported last-dimension TopK on a static matrix."""

    tokens: int
    experts: int
    k: int


def _get_score_shape(
    scores: torch.fx.Node,
) -> tuple[tuple[int, int] | None, str | None]:
    value = get_first_fake_tensor(scores)
    if value.ndim != 2:
        return None, "TopK requires rank-2 input."
    if value.dtype not in (torch.float16, torch.float32):
        return None, "TopK requires FP16 or FP32 scores."
    if any(type(size) is not int or size <= 0 for size in value.shape):
        return None, "TopK requires positive static T and E."
    tokens, experts = value.shape
    if experts > torch.iinfo(torch.int32).max:
        return None, "TopK expert dimension exceeds the int32 index-size limit."
    return (tokens, experts), None


def get_static_topk_config(
    node: torch.fx.Node, tosa_spec: TosaSpecification
) -> tuple[StaticTopKConfig | None, str | None]:
    """Check metadata and capabilities, without checking runtime finiteness.

    Args:
        node (torch.fx.Node): ATen or Edge TopK node with valid arguments.
        tosa_spec (TosaSpecification): Target capabilities.

    Returns:
        tuple: ``(config, None)`` for supported TopK, or ``(None, reason)``
            explaining why TopK is unsupported.

    """
    names = ("self", "k", "dim", "largest", "sorted")
    arguments = dict(zip(names, node.args))
    arguments.update(node.kwargs)
    shape, reason = _get_score_shape(cast(torch.fx.Node, arguments["self"]))
    if shape is None:
        return None, reason
    tokens, experts = shape
    k = arguments["k"]
    if type(k) is not int or not 1 <= k <= min(4, experts):
        return None, "TopK requires constant 1 <= K <= min(4, E)."
    dim = arguments.get("dim", -1)
    if type(dim) is not int or dim not in (-1, 1):
        return None, "TopK requires dim=-1 or dim=1."
    if arguments.get("largest", True) is not True:
        return None, "TopK requires largest=True."
    if arguments.get("sorted", True) is not True:
        return None, "TopK requires sorted=True."
    if not tosa_spec.support_float():
        return None, "TopK requires the FP profile."
    if k > 1 and not tosa_spec.support_integer():
        return None, "TopK with K>1 requires the FP and INT profiles."
    if any(
        user.target is not operator.getitem
        or len(user.args) != 2
        or type(user.args[1]) is not int
        or user.args[1] not in (0, 1)
        for user in node.users
    ):
        return None, "TopK requires canonical values/indices tuple extraction."
    return StaticTopKConfig(tokens, experts, k), None


def is_topk_indices_getitem(node: torch.fx.Node) -> bool:
    """Return whether node extracts the indices output of ATen or Edge TopK.

    Args:
        node (torch.fx.Node): Candidate tuple extraction node.

    Returns:
        bool: True for ``operator.getitem(topk, 1)``, selecting the indices
            from TopK's ``(values, indices)`` result.

    """
    return (
        node.target is operator.getitem
        and len(node.args) == 2
        and type(node.args[1]) is int
        and node.args[1] == 1
        and isinstance(node.args[0], torch.fx.Node)
        and node.args[0].target in TOPK_OPS
    )


def is_topk_indices_int32_cast(node: torch.fx.Node) -> bool:
    """Return whether node casts extracted TopK indices to int32.

    The checked node is the final cast in this pattern::

        topk(scores, K) -> getitem(1) -> _to_dim_order_copy(dtype=int32)

    Args:
        node (torch.fx.Node): Candidate cast node.

    Returns:
        bool: True for an ATen or Edge ``_to_dim_order_copy`` to int32
            whose input is a TopK indices extraction.

    """
    return (
        node.target in _CAST_OPS
        and len(node.args) == 1
        and node.kwargs.get("dtype") is torch.int32
        and isinstance(node.args[0], torch.fx.Node)
        and is_topk_indices_getitem(node.args[0])
    )


def topk_indices_only_feed_int32_casts(node: torch.fx.Node) -> bool:
    """Return whether extracted TopK indices feed only int32 casts.

    Expected pattern::

        topk -> getitem(1) -> _to_dim_order_copy(dtype=int32)

    Values consumers are ignored. If the indices output is unused, no
    int32 cast is required.

    Args:
        node (torch.fx.Node): TopK producer with canonical ``getitem`` users.

    Returns:
        bool: True when every immediate indices consumer matches
            ``is_topk_indices_int32_cast``, or the indices are unused.

    """
    for output in node.users:
        if not is_topk_indices_getitem(output):
            continue
        for consumer in output.users:
            if not is_topk_indices_int32_cast(consumer):
                return False
    return True


class DecomposeTopKPass(ArmPass):
    """Lower finite-score TopK using repeated ARGMAX and index masking.

    Runs after Arm partitioning. Every used TopK index extraction must feed
    only explicit int32 casts, matching this pattern::

        topk(scores, K)
            |-- getitem(0) --> value consumers
            `-- getitem(1) --> _to_dim_order_copy(dtype=int32) --> consumers

    Either output may be unused. Before partitioning,
    ``ConvertInt64OutputOpsToInt32Pass`` prepares restoration casts after
    the int32 casts for consumers requiring int64.

    For scores of shape ``[T, E]``, the decomposition is::

        remaining = scores
        selected_indices = []
        for step in range(K):
            index = argmax(remaining, dim=1).reshape(T, 1)
            selected_indices.append(index)
            if step + 1 < K:
                remaining = where(arange(E) == index, -inf, remaining)
        indices = concat(selected_indices, dim=1)  # int32, shape [T, K]
        values = gather(scores, dim=1, index=indices)

    Args:
        tosa_spec (TosaSpecification): Target capabilities.

    """

    _passes_required_after: set[type[ExportPass]] = {
        PrepareGatherIndicesPass,
        CanonicalizeGatherPass,
    }

    def __init__(self, tosa_spec: TosaSpecification) -> None:
        super().__init__()
        self.tosa_spec = tosa_spec

    @staticmethod
    def _build_topk_indices(
        graph: torch.fx.Graph,
        scores: torch.fx.Node,
        config: StaticTopKConfig,
    ) -> torch.fx.Node:
        """Build int32 indices using repeated ARGMAX and cumulative masking.

        Args:
            graph (torch.fx.Graph): Graph with the insertion point set.
            scores (torch.fx.Node): Original score tensor of shape [T, E].
            config (StaticTopKConfig): Validated static TopK parameters.

        Returns:
            torch.fx.Node: Int32 indices with shape [T, K].

        """
        expert_ids: torch.fx.Node | None = None
        minus_inf: torch.fx.Node | None = None
        if config.k > 1:
            scores_tensor = scores.meta["val"]
            expert_ids = create_node(
                graph,
                exir_ops.edge.aten.arange.start_step,
                (0, config.experts, 1),
                {"dtype": torch.int32, "device": scores_tensor.device},
            )
            expert_ids = create_node(
                graph,
                exir_ops.edge.aten.view_copy.default,
                (expert_ids, [1, config.experts]),
            )
            minus_inf = create_node(
                graph,
                exir_ops.edge.aten.full.default,
                ([1, 1], float("-inf")),
                {"dtype": scores_tensor.dtype, "device": scores_tensor.device},
            )
        remaining = scores
        selected_indices: list[torch.fx.Node] = []
        for step in range(config.k):
            # ARGMAX reduces [T, E] to int32 indices [T].
            index = create_node(
                graph, exir_ops.backend.tosa.ARGMAX.default, (remaining, 1)
            )
            index = create_node(
                graph,
                exir_ops.edge.aten.view_copy.default,
                (index, [config.tokens, 1]),
            )
            selected_indices.append(index)
            if step + 1 < config.k:
                assert expert_ids is not None and minus_inf is not None
                # Compare expert IDs [1, E] with selected indices [T, 1]
                # to produce a mask [T, E].
                mask = create_node(
                    graph, exir_ops.edge.aten.eq.Tensor, (expert_ids, index)
                )
                # Mask selected scores in remaining [T, E] with -inf.
                remaining = create_node(
                    graph,
                    exir_ops.edge.aten.where.self,
                    (mask, minus_inf, remaining),
                )
        if config.k == 1:
            indices = selected_indices[0]
        else:
            # Concatenate K selections of shape [T, 1] into [T, K].
            indices = create_node(
                graph, exir_ops.edge.aten.cat.default, (selected_indices, 1)
            )
        return indices

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        """Unroll selection and replace the TopK tuple's live extractions."""
        graph = graph_module.graph
        modified = False
        for topk in list(graph.nodes):
            if topk.target not in TOPK_OPS:
                continue
            config, reason = get_static_topk_config(topk, self.tosa_spec)
            if config is None:
                raise RuntimeError(f"Unsupported delegated TopK: {reason}")
            if not topk_indices_only_feed_int32_casts(topk):
                raise RuntimeError("Delegated TopK requires prepared int32 indices.")
            scores = topk.args[0] if topk.args else topk.kwargs["self"]
            with graph.inserting_before(topk):
                indices = self._build_topk_indices(graph, scores, config)
                # Gather from scores [T, E] using indices [T, K]
                # to produce values [T, K].
                values = create_node(
                    graph, exir_ops.edge.aten.gather.default, (scores, 1, indices)
                )
            # Redirect TopK getitem consumers to the new values and indices.
            for extraction in list(topk.users):
                replacement = values if extraction.args[1] == 0 else indices
                extraction.replace_all_uses_with(replacement)
            modified = True
        if modified:
            graph.eliminate_dead_code()
            graph.lint()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, modified)
