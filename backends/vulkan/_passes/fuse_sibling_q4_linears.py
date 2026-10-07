# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from collections import defaultdict
from typing import List, Optional, Tuple

import torch

from executorch.backends.transforms.utils import (
    create_constant_placeholder,
    get_param_tensor,
    is_param_node,
)
import executorch.backends.vulkan.utils as utils
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.export.graph_signature import InputKind


class FuseSiblingQ4LinearsPass(ExportPass):
    """
    Fuses dynamically quantized q4gsw linears that read the same input and are no
    wider than their input (e.g. the q, k and v projections of attention) into one
    linear over the concatenated weights that writes each of their outputs. During
    decode each of these GEMVs is too narrow to keep the GPU busy on its own.
    """

    def __init__(self) -> None:
        super().__init__()
        self._exported_program: Optional[ExportedProgram] = None

    def _out_channels(self, node: torch.fx.Node) -> int:
        # Packed weights are [N, K / 2]
        return get_param_tensor(self._exported_program, node.args[3]).shape[0]

    def _output_and_head_dim(
        self, node: torch.fx.Node
    ) -> Optional[Tuple[torch.fx.Node, int]]:
        """
        Returns the node whose value the fused op produces in place of this linear,
        and the head dim that the output channels are reshaped into (0 for none).
        A single reshape into heads that consumes the linear (as for q, k and v) is
        absorbed.
        """
        lead = list(node.args[0].meta["val"].shape[:-1])
        N = self._out_channels(node)
        users = list(node.users)
        candidates = [node]
        if len(users) == 1 and users[0].target == exir_ops.edge.aten.view_copy.default:
            candidates.insert(0, users[0])
        for out in candidates:
            shape = list(out.meta["val"].shape)
            if shape == lead + [N]:
                return out, 0
            if shape[:-2] == lead and shape[-2] * shape[-1] == N:
                return out, shape[-1]
        return None

    def _is_fusable(self, node: torch.fx.Node) -> bool:
        if (
            node.op != "call_function"
            or node.target != exir_ops.edge.et_vk.linear_dq8ca_q4gsw.default
        ):
            return False
        if len(node.args) > 7 and node.args[7] is not None:
            return False
        constants = node.args[3:6]
        for constant in constants:
            if not is_param_node(self._exported_program, constant):
                return False
            if len(constant.users) != 1:
                return False
        N = self._out_channels(node)
        K = node.args[0].meta["val"].shape[-1]
        # Output channel boundaries must fall on the 8-channel blocks of the packed
        # weight, and the weight sums/scales must not have been padded.
        if N % 8 != 0 or N > K:
            return False
        return self._output_and_head_dim(node) is not None

    def _fuse(self, graph_module: torch.fx.GraphModule, nodes: List[torch.fx.Node]):
        ep = self._exported_program
        graph = graph_module.graph

        def concat_constant(arg_idx: int, dim: int, suffix: str) -> torch.fx.Node:
            data = torch.cat(
                [get_param_tensor(ep, n.args[arg_idx]) for n in nodes], dim=dim
            )
            name = utils.get_tensor_name(ep, nodes[0].args[3]) + suffix
            with graph.inserting_before(list(graph.nodes)[0]):
                return create_constant_placeholder(
                    exp_program=ep,
                    graph=graph,
                    kind=InputKind.PARAMETER,
                    name=name.replace(".", "_"),
                    data=data,
                )

        # Weights are [N, K / 2]; weight sums and scales are [num_groups, N].
        weight = concat_constant(3, 0, "_fused")
        weight_sums = concat_constant(4, 1, "_fused_sums")
        weight_scales = concat_constant(5, 1, "_fused_scales")

        outputs, head_dims = zip(*[self._output_and_head_dim(n) for n in nodes])

        first = nodes[0]
        split_sizes = [self._out_channels(n) for n in nodes]
        with graph.inserting_before(first):
            split = graph.create_node(
                "call_function",
                exir_ops.edge.et_vk.linear_dq8ca_q4gsw_split.default,
                args=(
                    first.args[0],
                    first.args[1],
                    first.args[2],
                    weight,
                    weight_sums,
                    weight_scales,
                    first.args[6],
                    split_sizes,
                    list(head_dims),
                ),
            )
            split.meta["val"] = [out.meta["val"] for out in outputs]
            for i, out in enumerate(outputs):
                item = graph.create_node(
                    "call_function", operator.getitem, args=(split, i)
                )
                item.meta["val"] = out.meta["val"]
                out.replace_all_uses_with(item)

        for out, node in zip(outputs, nodes):
            if out is not node:
                graph.erase_node(out)

        old_constants = {c.name for n in nodes for c in n.args[3:6]}
        for node in nodes:
            graph.erase_node(node)
        # The state dict is shared with the other methods of a multi-method program,
        # which may still use these constants, so only this program's inputs go.
        # The input specs are edited in place, as create_constant_placeholder does,
        # since the pass manager maps them onto the new placeholders by position.
        input_specs = ep.graph_signature.input_specs
        input_specs[:] = [s for s in input_specs if s.arg.name not in old_constants]
        for node in list(graph.nodes):
            if node.op == "placeholder" and node.name in old_constants:
                graph.erase_node(node)

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        assert self._exported_program is not None

        siblings = defaultdict(list)
        for node in graph_module.graph.nodes:
            if self._is_fusable(node):
                # Siblings must also share the input qparams (usually None, i.e.
                # computed by the op) and the quantization group size.
                key = (node.args[0], node.args[1], node.args[2], node.args[6])
                siblings[key].append(node)

        # The runtime has dedicated kernels for three outputs (q, k and v); any
        # other count would fall back to copying the chunks out, which costs
        # more dispatches than it saves.
        groups = [nodes for nodes in siblings.values() if len(nodes) == 3]
        for nodes in groups:
            self._fuse(graph_module, nodes)

        if len(groups) == 0:
            return PassResult(graph_module, False)

        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, True)
