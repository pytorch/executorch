# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import torch
from executorch.backends.nxp.aten_passes.add_simulated_linear_bn_fusion_qat_pass import (
    _get_compute_scale_factor_pattern,
    _get_linear_weight_preprocess_pattern,
    _is_linear,
)
from executorch.backends.nxp.aten_passes.fuse_batch_norm_with_linear_pass import (
    _unwrap_if_fq,
)
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.fx.passes.utils.matcher_utils import InternalMatch, SubgraphMatcher
from torchao.quantization.pt2e.qat_utils import _get_aten_graph_module_for_pattern


def _is_op_node(node: Node, op) -> bool:
    return (
        node is not None
        and hasattr(node, "op")
        and node.op == "call_function"
        and hasattr(node, "target")
        and node.target == op
    )


_is_div = partial(_is_op_node, op=torch.ops.aten.div.Tensor)
_is_add = partial(_is_op_node, op=torch.ops.aten.add.Tensor)
_is_zeros_like = partial(_is_op_node, op=torch.ops.aten.zeros_like)
_is_reshape = partial(_is_op_node, op=torch.ops.aten.reshape)


def _is_denorm_pattern(node: Node) -> bool:
    if not _is_div(node):
        return False

    if not hasattr(node, "users"):
        return False

    div_user_ops = [
        user.target for user in node.users.keys() if hasattr(user, "target")
    ]
    if len(list(div_user_ops)) < 1:
        return False

    if torch.ops.aten.batch_norm.default in div_user_ops:
        return True

    return False


def _remove_pattern_from_graph(graph_module: GraphModule, pattern: GraphModule):
    matcher = SubgraphMatcher(
        pattern.graph,
        match_output=False,
        match_placeholder=False,
        remove_overlapping_matches=True,
        ignore_literals=True,
    )
    matches: list[InternalMatch] = matcher.match(graph_module.graph, node_name_match="")

    for match in matches:
        last_pattern_node = match.anchors[0]
        last_matched_subgraph_node = match.nodes_map[last_pattern_node]
        weight = match.placeholder_nodes[0]

        last_matched_subgraph_node.replace_all_uses_with(weight)

        for node in match.nodes_map.values():
            if node not in match.placeholder_nodes:
                graph_module.graph.erase_node(node)


def _remove_late_bias_pattern(graph_module: GraphModule, bias_node: Node):
    linear_b_users = list(bias_node.users.keys())

    if len(linear_b_users) != 2:
        return

    if _is_zeros_like(linear_b_users[0]):
        zeros_node, maybe_reshape_node = linear_b_users
    elif _is_zeros_like(linear_b_users[1]):
        maybe_reshape_node, zeros_node = linear_b_users
    else:
        return

    if _is_reshape(maybe_reshape_node):
        reshape_node = maybe_reshape_node
        reshape_users = list(reshape_node.users.keys())

        if len(reshape_users) != 1:
            return

        add_node = reshape_users[0]
    else:
        # Handles no reshape node when bias is scalar
        reshape_node = None
        add_node = maybe_reshape_node

    if not _is_add(add_node):
        return

    # Remove zeroed linear bias
    zeros_node.replace_all_uses_with(bias_node)
    graph_module.graph.erase_node(zeros_node)

    # Remove late bias addition
    add_node.replace_all_uses_with(add_node.args[0])
    graph_module.graph.erase_node(add_node)

    if reshape_node:
        graph_module.graph.erase_node(reshape_node)


def _remove_denorm_and_late_bias(graph_module: GraphModule):
    named_modules = dict(graph_module.named_modules(remove_duplicate=False))

    for node in graph_module.graph.nodes:
        if not _is_linear(node):
            continue

        linear_node = node

        if len(linear_node.args) <= 2:
            continue

        linear_bias_fq_or_zeros = _unwrap_if_fq(
            linear_node.args[2], named_modules=named_modules
        )
        has_late_bias = _is_zeros_like(linear_bias_fq_or_zeros)

        if has_late_bias:
            _remove_late_bias_pattern(
                graph_module, bias_node=linear_bias_fq_or_zeros.args[0]
            )

        for user_node in linear_node.users:
            if _is_denorm_pattern(user_node):
                users_ops = [user.target for user in user_node.users.keys()]

                if torch.ops.aten.batch_norm.default in users_ops:
                    user_node.replace_all_uses_with(node)
                    graph_module.graph.erase_node(user_node)
                    break


class RemoveSimulatedLinearBatchNormFusionQATPass(PassBase):
    """
    In order for QAT to work correctly with fused linear + batch norm operators,
    simulated linear + batch norm fusion should be added using AddSimulatedLinearBatchNormFusionQATPass.

    After the QAT training, before inserting QDQ nodes, nodes added by the simulated fusion should be removed.
    This pass removes all artifacts created by AddSimulatedLinearBatchNormFusionQATPass and reverts
    the graph back to the layout before the simulated fusion was applied.
    See `add_simulated_linear_bn_fusion_qat_pass.py` for more details.
    """

    def call(self, graph_module: GraphModule) -> PassResult | None:
        """
        Given a graph of decomposed aten ops, removes nodes corresponding to linear + batch norm fusion.
        """
        is_cuda = False

        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()

        _scale_compute_example_inputs = (
            torch.randn(1),
            torch.randn(1),
        )
        _preprocess_example_inputs = (
            torch.randn(1, 1),
            torch.randn(1),
        )

        scale_pattern = _get_compute_scale_factor_pattern()
        scale_match_pattern = _get_aten_graph_module_for_pattern(
            pattern=scale_pattern,
            example_inputs=_scale_compute_example_inputs,
            is_cuda=is_cuda,
        )

        weight_preprocess_pattern = _get_linear_weight_preprocess_pattern()
        weight_preprocess_pattern = _get_aten_graph_module_for_pattern(
            pattern=weight_preprocess_pattern,
            example_inputs=_preprocess_example_inputs,
            is_cuda=is_cuda,
        )

        _remove_pattern_from_graph(graph_module, pattern=scale_match_pattern)
        _remove_pattern_from_graph(graph_module, pattern=weight_preprocess_pattern)
        _remove_denorm_and_late_bias(graph_module)

        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()

        return PassResult(graph_module, True)
