# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
from typing import Set, Type

import torch
from executorch.exir.pass_base import ExportPass, PassResult
from torch._decomp import get_decompositions
from torch.fx.experimental.proxy_tensor import make_fx


class DecomposeScaledDotProductAttention(ExportPass):
    """
    Decompose from scaled_dot_product_attention to multiple nodes.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def __init__(self, allow_non_fake_inputs: bool = True) -> None:
        super().__init__()
        # With allow_non_fake_inputs=False, we don't get _unsafe_view ops
        # in the graph, we allow disabling it here.
        self._allow_non_fake_inputs = allow_non_fake_inputs

    def call(
        self, graph_module: torch.fx.GraphModule, allow_non_fake_inputs: bool = True
    ) -> PassResult:
        graph = graph_module.graph
        for node in list(graph.nodes):
            if node.target != torch.ops.aten.scaled_dot_product_attention.default:
                continue
            self._decompose_sdpa_node(graph_module, node, allow_non_fake_inputs)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)

    @staticmethod
    def _extract_input_tensors(node: torch.fx.Node) -> tuple[object, ...]:
        def _extract_arg_value(arg):
            if isinstance(arg, torch.fx.Node):
                if "val" not in arg.meta:
                    raise RuntimeError(
                        f"Missing meta['val'] for SDPA arg node: {arg.name}"
                    )
                return arg.meta["val"]
            return arg

        return tuple(_extract_arg_value(arg) for arg in node.args)

    @staticmethod
    def _copy_decomposed_graph(
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        decomposed_module: torch.fx.GraphModule,
        scale: object,
    ) -> None:
        name_to_input_tensor_map = {}
        for i, arg in enumerate(node.args):
            name_to_input_tensor_map[f"arg{i}_1"] = arg

        decomposed_node_to_subgraph_node: dict[torch.fx.Node, torch.fx.Node] = {}
        last_decomposed_node = None
        for decomposed_node in decomposed_module.graph.nodes:
            if decomposed_node.op == "placeholder":
                decomposed_node_to_subgraph_node[decomposed_node] = (
                    name_to_input_tensor_map[decomposed_node.name]
                )

            if decomposed_node.op == "output":
                last_decomposed_node = decomposed_node.args[0]

        for decomposed_node in decomposed_module.graph.nodes:
            node.meta["nn_module_stack"] = decomposed_node.meta.get("nn_module_stack")
            if decomposed_node.op == "placeholder":
                continue

            if decomposed_node.op == "output" and last_decomposed_node is not None:
                for user in node.users.copy():
                    user.replace_input_with(
                        node,
                        decomposed_node_to_subgraph_node[last_decomposed_node],
                    )
                continue

            if scale is not None and decomposed_node.target in [
                torch.ops.aten.mul.Scalar
            ]:
                new_args = list(decomposed_node.args)
                new_args[1] = math.sqrt(scale)
                decomposed_node.args = tuple(new_args)

            subgraph_node = graph.node_copy(
                decomposed_node,
                arg_transform=lambda x: decomposed_node_to_subgraph_node[x],
            )
            subgraph_node.meta["source_fn_stack"] = [
                (subgraph_node, subgraph_node.target)
            ]
            decomposed_node_to_subgraph_node[decomposed_node] = subgraph_node

    def _decompose_sdpa_node(
        self,
        graph_module: torch.fx.GraphModule,
        node: torch.fx.Node,
        allow_non_fake_inputs: bool,
    ) -> None:
        graph = graph_module.graph

        input_tensors = self._extract_input_tensors(node)
        scale = node.kwargs.get("scale", None)

        def _sdpa_with_gqa(*args, **kwargs):
            # args: (q, k, v, [attn_mask, dropout_p, is_causal, scale])
            q, k, v = args[:3]
            # Shapes: (B, H, T, D)
            Hq = q.shape[1]
            Hk = k.shape[1]
            if Hq != Hk:
                # LLaMA-style GQA: tile K and V heads to match Q
                assert Hq % Hk == 0, f"GQA mismatch: Hq={Hq}, Hk={Hk}"
                r = Hq // Hk
                B, _, Tk, D = k.shape
                k = k.unsqueeze(2).expand(B, Hk, r, Tk, D).reshape(B, Hq, Tk, D)
                v = v.unsqueeze(2).expand(B, Hk, r, Tk, D).reshape(B, Hq, Tk, D)
                args = (q, k, v) + tuple(args[3:])
            return torch.ops.aten.scaled_dot_product_attention.default(*args, **kwargs)

        # refer to pytorch/test/test_decomp.py
        decomposed_module = make_fx(
            _sdpa_with_gqa,
            decomposition_table=get_decompositions(  # pyre-fixme[6]
                [
                    torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default,
                ]
            ),
            tracing_mode="fake",
            _allow_non_fake_inputs=allow_non_fake_inputs,
        )(*input_tensors)

        with graph.inserting_before(node):
            self._copy_decomposed_graph(graph, node, decomposed_module, scale)

            graph.erase_node(node)
