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
from torch.fx.node import Argument


class DecomposeScaledDotProductAttention(ExportPass):
    """
    Decompose from scaled_dot_product_attention to multiple nodes.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()
    _SDPA_OPTIONAL_ARGS = (
        ("attn_mask", None),
        ("dropout_p", 0.0),
        ("is_causal", False),
    )

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
    def _extract_arg_value(arg: object) -> object:
        if isinstance(arg, torch.fx.Node):
            if "val" not in arg.meta:
                raise RuntimeError(f"Missing meta['val'] for SDPA arg node: {arg.name}")
            return arg.meta["val"]
        return arg

    @classmethod
    def _canonicalize_sdpa_call(
        cls, node: torch.fx.Node
    ) -> tuple[tuple[object, ...], object, object]:
        input_args = list(node.args)
        input_kwargs = dict(node.kwargs)

        canonical_args = list(input_args[:3])
        for arg_index, (arg_name, default) in enumerate(
            cls._SDPA_OPTIONAL_ARGS, start=3
        ):
            if len(input_args) > arg_index:
                canonical_args.append(input_args[arg_index])
            else:
                canonical_args.append(input_kwargs.pop(arg_name, default))

        raw_scale = input_kwargs.pop("scale", None)
        canonical_args.append(raw_scale)
        scale = cls._extract_arg_value(raw_scale)
        enable_gqa = cls._extract_arg_value(input_kwargs.pop("enable_gqa", False))
        if input_kwargs:
            raise RuntimeError(
                "Unsupported kwargs for scaled_dot_product_attention: "
                f"{', '.join(sorted(input_kwargs.keys()))}"
            )

        return tuple(canonical_args), scale, enable_gqa

    @staticmethod
    def _copy_decomposed_graph(
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        decomposed_module: torch.fx.GraphModule,
        canonical_inputs: tuple[object, ...],
        scale: object,
    ) -> None:
        decomposed_node_to_subgraph_node: dict[torch.fx.Node, Argument] = {}
        last_decomposed_node: torch.fx.Node | None = None
        placeholder_nodes = [
            decomposed_node
            for decomposed_node in decomposed_module.graph.nodes
            if decomposed_node.op == "placeholder"
        ]
        if len(placeholder_nodes) != len(canonical_inputs):
            raise RuntimeError(
                "Unexpected placeholder count when decomposing "
                "scaled_dot_product_attention"
            )
        for decomposed_node, arg in zip(placeholder_nodes, canonical_inputs):
            decomposed_node_to_subgraph_node[decomposed_node] = arg

        for decomposed_node in decomposed_module.graph.nodes:
            if decomposed_node.op == "output":
                output_arg = decomposed_node.args[0]
                if not isinstance(output_arg, torch.fx.Node):
                    raise RuntimeError(
                        "Unexpected non-node output when decomposing "
                        "scaled_dot_product_attention"
                    )
                last_decomposed_node = output_arg

        for decomposed_node in decomposed_module.graph.nodes:
            decomposed_node.meta["nn_module_stack"] = node.meta.get("nn_module_stack")
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
            subgraph_node.meta["nn_module_stack"] = node.meta.get("nn_module_stack")
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

        canonical_inputs, scale, enable_gqa = self._canonicalize_sdpa_call(node)
        input_tensors = tuple(self._extract_arg_value(arg) for arg in canonical_inputs)

        def _sdpa_with_gqa(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
        ):
            # Shapes: (B, H, T, D)
            Hq = q.shape[1]
            Hk = k.shape[1]
            if Hq != Hk:
                # LLaMA-style GQA: tile K and V heads to match Q
                if not enable_gqa:
                    raise ValueError(
                        "SDPA head mismatch requires enable_gqa=True: "
                        f"Hq={Hq}, Hk={Hk}"
                    )
                if Hq % Hk != 0:
                    raise ValueError(f"GQA mismatch: Hq={Hq}, Hk={Hk}")
                r = Hq // Hk
                B, _, Tk, D = k.shape
                k = k.unsqueeze(2).expand(B, Hk, r, Tk, D).reshape(B, Hq, Tk, D)
                v = v.unsqueeze(2).expand(B, Hk, r, Tk, D).reshape(B, Hq, Tk, D)
            return torch.ops.aten.scaled_dot_product_attention.default(
                q,
                k,
                v,
                attn_mask,
                dropout_p,
                is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )

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
            self._copy_decomposed_graph(
                graph, node, decomposed_module, canonical_inputs, scale
            )

            graph.erase_node(node)
