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

    def _propagate_quant_params(self, node: Node):
        assert (
            quantize_attrs := node.meta.get("quantize_attrs")
        ), "Must be annotated node."
        requantize_map: Dict[Node, Node] = node.meta.get("requantize", {})
        while node.users:
            if len(node.users) != 1:
                break
            user = list(node.users.keys())[0]
            if (
                user.target not in QuantConstants.QUANT_OPS_KEY_MAP
                and user.target not in QuantConstants.DEQUANT_OPS_KEY_MAP
            ):
                break
            node = user
        # Case1: ...-q-dq(cur)-propagate_node-node(not d-dq)
        # Case2: propagate_node(propagateed)-propagate_node-node(not q-dq)
        for idx, user in enumerate(node.users.keys()):
            # For the branch who need to be requantized, we propagate the requantize params
            user_attrs = requantize_map.get(idx, quantize_attrs)
            if user.target not in self.propagate_nodes:
                continue
            if len(user.users) == 1:
                # Possibily no need for checking len(users)>1
                user_of_user = list(user.users)[0]
                # node-q-dq-propagate-q-dq not need for propagatey
                if (
                    user_of_user.target in QuantConstants.QUANT_OPS_KEY_MAP
                    or user_of_user.target in QuantConstants.DEQUANT_OPS_KEY_MAP
                ):
                    continue
            # propagate quant for node-q-dq-propagate_node-node(not qdq)
            user.meta["quantize_attrs"] = user_attrs
            self._propagate_quant_params(user)

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
                requantize_map[idx] = requantize_attrs

    def _annotate(self, graph_module: GraphModule):
        for node in graph_module.graph.nodes:
            key_map = QuantConstants.QUANT_OPS_KEY_MAP.get(node.target, None)
            if not key_map:
                continue
            source_node = node.args[0]
            if source_node.target in (
                *QuantConstants.QUANT_OPS_KEY_MAP,
                *QuantConstants.DEQUANT_OPS_KEY_MAP,
            ):
                # Currently, don't add quant info for d_qd node here.
                continue
            elif source_node.target == operator.getitem:
                source_node = source_node.args[0]
            quant_attrs = self.get_quant_attrs(node, key_map)
            source_node.meta["quantize_attrs"] = quant_attrs
            self._annotate_requantize(source_node)
            self._propagate_quant_params(source_node)

    def call(self, graph_module: GraphModule):
        self._annotate(graph_module)
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
