# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import Any, Dict, List, Optional

import torch
from executorch.backends.samsung.utils.constants import QuantConstants
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch._export.utils import get_buffer
from torch.export import ExportedProgram
from torch.fx import GraphModule, Node
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions


class AnnotateQparamsPass(ExportPass):
    """This parse is to add quantize properties to node need to be quantized.

    Annotate Quant params:
        For src_node->Q->DQ->..., we will add the quant params from Q->DQ node
         to the src_node

    Annotate Requantize:
        For src_node->Q->DQ->Q->DQ->..., if the multiple Q->DQ contains
         different quant params, we will mark the src_node as need requantize,
         and add Q->DQ after removing all the Q->DQs.
    """

    propagate_nodes = {
        exir_ops.edge.aten.view_copy.default,
        exir_ops.edge.aten.permute_copy.default,
        exir_ops.edge.aten.squeeze_copy.default,
        exir_ops.edge.aten.squeeze_copy.dim,
        exir_ops.edge.aten.squeeze_copy.dims,
        exir_ops.edge.aten.slice_copy.Tensor,
        exir_ops.edge.aten.unsqueeze_copy.default,
        exir_ops.edge.aten.concat.default,
        exir_ops.edge.aten.cat.default,
        exir_ops.edge.aten.expand_copy.default,
        exir_ops.edge.aten.split_with_sizes_copy.default,
        exir_ops.edge.aten.clone.default,
        exir_ops.edge.aten.contiguous.default,
    }

    def __init__(self, edge_program: ExportedProgram):
        super().__init__()
        self.edge_program = edge_program

    def _get_last_dqs(self, node: Node) -> List[Node]:
        r"""From one Q-DQ node, find the last DQs in the quantization node chain.


        need to consider such case:
                    /--Q-DQ-node1
            node->Q->DQ--node-node2
                    \--Q-DQ-node3
        This is a dfs implemention, so result will keep sorted
        Args:
            node (Node): Search DQ from this node.

        Returns:
            List[Node]: list of DQ node by original sequence
        """

        def _impl(node: Node, res_list: List[Node]):
            if (
                node.target not in QuantConstants.QUANT_OPS_KEY_MAP
                and node.target not in QuantConstants.DEQUANT_OPS_KEY_MAP
            ):
                return
            for user in node.users.keys():
                if (
                    user.target not in QuantConstants.QUANT_OPS_KEY_MAP
                    and user.target not in QuantConstants.DEQUANT_OPS_KEY_MAP
                ):
                    res_list.append(node)
                else:
                    _impl(user, res_list)

        res_list: List[Node] = []
        for user in node.users:
            _impl(user, res_list)
        return res_list

    def _walk_qdq_chain_to_terminals(self, cur: Node) -> List[Node]:
        r"""Walk forward from a Q/DQ node `cur` through the Q-DQ chain,
        returning every terminal node (last Q/DQ node before a non-Q/DQ
        consumer). Handles fan-out: the SAME quantized tensor is commonly
        dequantized into multiple branches (e.g. one Q feeding two DQs for
        two independent consumers) -- each branch is walked and its own
        terminal is collected, instead of the chain silently stopping at
        `cur` when it has more than one Q/DQ child.

        Mirrors the DFS shape of `_get_last_dqs`, applied starting from a
        single Q/DQ node rather than its non-Q/DQ source.
        """
        next_nodes = [
            u
            for u in cur.users
            if u.target in QuantConstants.QUANT_OPS_KEY_MAP
            or u.target in QuantConstants.DEQUANT_OPS_KEY_MAP
        ]
        if not next_nodes:
            return [cur]
        terminals: List[Node] = []
        for nxt in next_nodes:
            terminals.extend(self._walk_qdq_chain_to_terminals(nxt))
        return terminals

    def _collect_dq_nodes(self, node: Node) -> List[Node]:
        """For each user of `node`, resolve it to a propagate candidate: a
        non-Q user is used directly, a Q user is walked through its Q-DQ
        chain (including fan-out) to find every terminal node."""
        dq_nodes: List[Node] = []
        for user in node.users:
            if user.target not in QuantConstants.QUANT_OPS_KEY_MAP:
                # If user is a direct propagate node (not Q), collect it too
                dq_nodes.append(user)
                continue
            # user is a Q node: walk through the Q-DQ chain (including any
            # fan-out branches) to find every terminal node.
            dq_nodes.extend(self._walk_qdq_chain_to_terminals(user))
        return dq_nodes

    def _is_propagatable(self, candidate: Node) -> bool:
        """True if `candidate` is a SharedQuant propagate node that is safe
        to annotate with quantize_attrs and recurse into: it must be a
        propagate node, and if it has exactly one user, that user must not
        be a Q/DQ (a Q/DQ boundary already carries its own quant params)."""
        if candidate.target not in self.propagate_nodes:
            return False
        if len(candidate.users) == 1:
            only_user = next(iter(candidate.users))
            if (
                only_user.target in QuantConstants.QUANT_OPS_KEY_MAP
                or only_user.target in QuantConstants.DEQUANT_OPS_KEY_MAP
            ):
                return False
        return True

    def _propagate_into_dequant_users(self, dq_node: Node, user_attrs) -> None:
        """dq_node is a DQ: propagate to each of its users that is itself an
        eligible (non-Q/DQ) propagate node."""
        for op_user in dq_node.users:
            if (
                op_user.target in QuantConstants.QUANT_OPS_KEY_MAP
                or op_user.target in QuantConstants.DEQUANT_OPS_KEY_MAP
            ):
                continue
            if not self._is_propagatable(op_user):
                continue
            op_user.meta["quantize_attrs"] = user_attrs
            self._propagate_quant_params(op_user)

    def _propagate_quant_params(self, node: Node):
        assert (
            quantize_attrs := node.meta.get("quantize_attrs")
        ), "Must be annotated node."
        requantize_map: Dict[Node, Node] = node.meta.get("requantize", {})
        # Walk through Q-DQ chains, handling multiple Q-DQ branches.
        # For node->Q->DQ->op1 and node->Q->DQ->op3, we collect all last DQ nodes.
        dq_nodes = self._collect_dq_nodes(node)
        # Case1: ...-q-dq(cur)-propagate_node-node(not q-dq)
        # Case2: propagate_node(propagated)-propagate_node-node(not q-dq)
        for idx, dq_node in enumerate(dq_nodes):
            # For the branch who need to be requantized, we propagate the requantize params
            user_attrs = requantize_map.get(idx, quantize_attrs)
            if dq_node.target in QuantConstants.DEQUANT_OPS_KEY_MAP:
                self._propagate_into_dequant_users(dq_node, user_attrs)
            elif self._is_propagatable(dq_node):
                # dq_node is not a DQ but a propagate node directly connected to source
                dq_node.meta["quantize_attrs"] = user_attrs
                self._propagate_quant_params(dq_node)

    def _backward_propagate(self, node: Node):
        """Walk backward from `node`, copying its quantize_attrs into unannotated
        single-input upstream ops.

        Handles patterns like `DQ → SiLU → chunk → Q` where forward propagation
        cannot populate SiLU's quantize_attrs because SiLU is not in
        `propagate_nodes` (its input/output scales differ in general). But when
        the downstream is a SharedQuant op (chunk/split/view/permute/...) whose
        input scale must equal its output scale, the intermediate op's output
        scale is fully determined by the downstream shared scale, so backward
        propagation is safe.

        Stops at:
          - already-annotated upstream (respect forward pass results)
          - Q/DQ boundaries (scale is defined by the Q/DQ params themselves)
          - non-call_function nodes (placeholders, get_attr, output)
          - multi-input upstream ops (ambiguous which input to follow)
        """
        quant_attrs = node.meta.get("quantize_attrs")
        if not quant_attrs:
            return
        inputs = node.all_input_nodes
        if len(inputs) != 1:
            return
        upstream = inputs[0]
        if upstream.meta.get("quantize_attrs"):
            return
        if upstream.target in QuantConstants.QUANT_OPS_KEY_MAP:
            return
        if upstream.target in QuantConstants.DEQUANT_OPS_KEY_MAP:
            return
        if upstream.op != "call_function":
            return
        upstream.meta["quantize_attrs"] = quant_attrs
        self._backward_propagate(upstream)

    def _propagate_quant_params_backward_all(self, graph_module: GraphModule):
        """For every SharedQuant propagate node with annotated quantize_attrs,
        walk backward and fill in unannotated single-input upstream ops.

        Multi-input propagate ops (cat/concat) are excluded because their
        upstream is ambiguous — different input branches may legitimately have
        different scales, and picking one to backward-propagate would corrupt
        the others.
        """
        single_input_shared = self.propagate_nodes - {
            exir_ops.edge.aten.concat.default,
            exir_ops.edge.aten.cat.default,
        }
        for node in graph_module.graph.nodes:
            if node.target not in single_input_shared:
                continue
            if not node.meta.get("quantize_attrs"):
                continue
            self._backward_propagate(node)

    def _annotate_requantize(self, node: Node):
        assert (
            ori_quant_attrs := node.meta.get("quantize_attrs")
        ), "No quant parameters found"
        list_for_requantize = self._get_last_dqs(node)
        node.meta["requantize"] = node.meta.get("requantize", {})

        # We use index to mark the output to be requantized
        # Because user obj and name may change when we requantize them.

        def _check_same(requant_obj, ori_obj) -> bool:
            if type(requant_obj) != type(ori_obj):  # noqa E721
                # We need actually same type here.
                return False
            if not isinstance(requant_obj, torch.Tensor):
                return requant_obj == ori_obj
            if requant_obj.shape != ori_obj.shape:
                return False
            return bool((requant_obj == ori_obj).all())

        requantize_map: Dict[int, Dict] = node.meta["requantize"]
        for idx, dq in enumerate(list_for_requantize):
            q = dq.all_input_nodes[0]
            if q.target not in QuantConstants.QUANT_OPS_KEY_MAP:
                continue
            key_map = QuantConstants.DEQUANT_OPS_KEY_MAP[dq.target]
            requantize_attrs = self.get_quant_attrs(q, key_map)
            if not all(
                _check_same(ori_quant_attrs[key], requantize_attrs[key])
                for key in key_map.values()
            ):
                if (
                    ori_quant_attrs[QuantConstants.QUANT_KEY.quant_dtype]
                    != requantize_attrs[QuantConstants.QUANT_KEY.quant_dtype]
                ):
                    # For Q-DQ who will change quant dtype, we will insert requantization node
                    requantize_map[idx] = requantize_attrs
                else:
                    node.meta["quantize_attrs"] = requantize_attrs

    def _annotate(self, graph_module: GraphModule):
        for node in graph_module.graph.nodes:
            if key_map := QuantConstants.DEQUANT_OPS_KEY_MAP.get(node.target, None):
                # We will fold node with constant output in the future pass as a constant node
                # example: Constant->Q->DQ->nodeN->Q->DQ, this seq will be folded to one
                # We need to store the q-params from last DQ params for quantizing constant value
                quant_attrs = self.get_quant_attrs(node, key_map)
                if node.args[0].target in QuantConstants.QUANT_OPS_KEY_MAP:
                    node.meta["quantize_attrs"] = quant_attrs
                else:
                    node.args[0].meta["quantize_attrs"] = quant_attrs
                continue
            key_map = QuantConstants.QUANT_OPS_KEY_MAP.get(node.target, None)
            if not key_map:
                continue
            quant_attrs = self.get_quant_attrs(node, key_map)
            if node.args[0].target in QuantConstants.QUANT_OPS_KEY_MAP:
                node.meta["quantize_attrs"] = quant_attrs
                continue
            source_node = node.args[0]
            if source_node.target in (
                *QuantConstants.QUANT_OPS_KEY_MAP,
                *QuantConstants.DEQUANT_OPS_KEY_MAP,
            ):
                # Currently, don't add quant info for d_qd node here.
                continue
            source_node.meta["quantize_attrs"] = quant_attrs
            if source_node.target == operator.getitem:
                source_node.args[0].meta["quantize_attrs"] = quant_attrs
                self._annotate_requantize(source_node.args[0])
            else:
                self._annotate_requantize(source_node)

            self._propagate_quant_params(source_node)

    def _annotate_in_quantize_attrs(self, graph_module: GraphModule):
        # Collects quantize_attrs attributes along with their input index into a list, and stores them
        # in the current node's meta["in_quantize_attrs"]. This information is used later in
        # customized_constant_prop.py.
        for node in graph_module.graph.nodes:
            in_quantize_attrs = []
            for idx, input_node in enumerate(node.all_input_nodes):
                # Check if input is a Dequant node
                if input_node.target not in QuantConstants.DEQUANT_OPS_KEY_MAP:
                    continue
                # Check if Dequant's input is a Quant node
                if input_node.args[0].target in QuantConstants.QUANT_OPS_KEY_MAP:
                    quant_node = input_node.args[0]
                    quant_input_node = quant_node.args[0]
                    if "quantize_attrs" in quant_input_node.meta:
                        in_quantize_attrs.append(
                            (idx, quant_input_node.meta["quantize_attrs"])
                        )
                else:
                    # Const -> Dequant, get quantize_attrs from const node
                    if "quantize_attrs" in input_node.args[0].meta:
                        in_quantize_attrs.append(
                            (idx, input_node.args[0].meta["quantize_attrs"])
                        )
            if in_quantize_attrs:
                node.meta["in_quantize_attrs"] = in_quantize_attrs

    def _annotate_decomposed_mm(self, graph_module: GraphModule):
        partitions = get_source_partitions(
            graph_module.graph,
            [
                "matmul",
                torch.ops.aten.matmul.default,
                operator.matmul,
                torch.matmul,
                torch.bmm,
            ],
        )

        for _, src_partitions in partitions.items():
            for src_partition in src_partitions:
                final_view = src_partition.output_nodes[0]
                if not (quantize_attrs := final_view.meta.get("quantize_attrs")):
                    continue
                for node in src_partition.nodes:
                    if node.target == exir_ops.edge.aten.bmm.default:
                        node.meta["quantize_attrs"] = quantize_attrs
                        break

    def call(self, graph_module: GraphModule):
        self._annotate(graph_module)
        self._propagate_quant_params_backward_all(graph_module)
        self._annotate_decomposed_mm(graph_module)
        self._annotate_in_quantize_attrs(graph_module)
        graph_module.recompile()
        return PassResult(graph_module, True)

    def get_quant_attrs(
        self, quant_node: torch.fx.Node, key_map: Optional[Dict] = None
    ) -> Dict[str, Any]:
        quant_attr_keys = [arg.name for arg in quant_node.target._schema.arguments]
        quant_attrs = dict.fromkeys(quant_attr_keys)
        for key, attr in zip(quant_attr_keys[1:], quant_node.args[1:]):
            # For channel-wise quantization, params are stored by buffer nodes.
            if isinstance(attr, torch.fx.Node):
                attr = get_buffer(self.edge_program, attr)
            quant_attrs[key] = attr
        quant_attrs["target"] = quant_node.target
        if key_map is None:
            return quant_attrs
        miss_attrs = []
        for aten_attr, snc_attr in key_map.items():
            if aten_attr not in quant_attrs:
                miss_attrs.append(aten_attr)
                continue
            attr = quant_attrs[aten_attr]
            quant_attrs.pop(aten_attr)
            quant_attrs[snc_attr] = attr
        assert (
            not miss_attrs
        ), f"Miss quant attrs {miss_attrs} for node {quant_node.name}"
        return quant_attrs
