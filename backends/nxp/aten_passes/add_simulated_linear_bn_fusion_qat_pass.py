# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

import torch
from executorch.backends.nxp.aten_passes.fuse_batch_norm_with_linear_pass import (
    _unwrap_if_fq,
)
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torchao.quantization.pt2e.export_utils import WrapperModule
from torchao.quantization.pt2e.prepare import _is_activation_post_process_node
from torchao.quantization.pt2e.qat_utils import _get_aten_graph_module_for_pattern


_bn_ops = [
    torch.ops.aten.batch_norm.default,
    torch.ops.aten.native_batch_norm.default,
    torch.ops.aten._native_batch_norm_legit_no_training.default,
]


def _get_linear_weight_preprocess_pattern() -> Callable:
    def _preprocess(
        weight: torch.Tensor,
        scale_factor: torch.Tensor,
    ) -> torch.Tensor:
        weight_shape = [1] * len(weight.shape)
        weight_in_channel_axis = 0
        weight_shape[weight_in_channel_axis] = -1
        return weight * scale_factor.reshape(weight_shape)

    return WrapperModule(_preprocess)


def _get_compute_scale_factor_pattern(bn_eps: float = 1e-5) -> Callable:
    def _compute_scale(
        bn_weight: torch.Tensor,
        bn_running_var: torch.Tensor,
    ) -> torch.Tensor:
        running_std = torch.sqrt(bn_running_var + bn_eps)
        scale_factor = bn_weight / running_std
        return scale_factor

    return WrapperModule(_compute_scale)


def _is_batch_norm(node: Node) -> bool:
    return (
        node is not None
        and hasattr(node, "op")
        and node.op == "call_function"
        and hasattr(node, "target")
        and node.target in _bn_ops
    )


def _is_linear(node: Node) -> bool:
    return (
        node is not None
        and hasattr(node, "op")
        and node.op == "call_function"
        and hasattr(node, "target")
        and node.target == torch.ops.aten.linear.default
    )


def _get_input_nodes(graph_module: GraphModule) -> tuple[Node]:
    input_nodes = []
    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            input_nodes.append(node)

    return tuple(input_nodes)


def _get_tensor_shape(node: Node) -> torch.Size | None:
    if hasattr(node, "meta") and "tensor_meta" in node.meta:
        return node.meta["tensor_meta"].shape

    return None


def _get_bn_eps(bn_node: Node) -> float:
    if "eps" in bn_node.kwargs:
        return bn_node.kwargs["eps"]
    elif len(bn_node.args) > 7:
        return bn_node.args[7]
    return 1e-5


def _reinsert_node_before(
    graph_module: GraphModule, node: Node, insertion_point: Node
) -> Node:
    with graph_module.graph.inserting_before(insertion_point):
        node_copy = graph_module.graph.node_copy(node)
        node.replace_all_uses_with(node_copy)
        graph_module.graph.erase_node(node)

    return node_copy


def _insert_linear_output_denorm(
    graph_module: GraphModule, linear_node: Node, scale_factor: Node, bn_node: Node
) -> Node:
    """
    Inserts div node between linear and batch norm layers.
    This denormalizes the linear output by dividing by the scale factor.
    """
    with graph_module.graph.inserting_before(bn_node):
        div_node = graph_module.graph.create_node(
            op="call_function",
            target=torch.ops.aten.div.Tensor,
            args=(linear_node, scale_factor),
        )
        bn_node.update_arg(0, div_node)

    return div_node


def _insert_disabled_linear_bias(
    graph_module: GraphModule, linear_b_node: Node
) -> Node:
    """
    Inserts a zeros_like node to disable the linear layer's bias application.
    Expects the application to be placed after the linear output denormalization.
    """
    with graph_module.graph.inserting_after(linear_b_node):
        zeros_like = graph_module.graph.create_node(
            op="call_function",
            target=torch.ops.aten.zeros_like,
            args=(),
        )
        linear_b_node.replace_all_uses_with(zeros_like)
        zeros_like.insert_arg(0, linear_b_node)

    return zeros_like


def _insert_bias_add_after_denorm(
    graph_module: GraphModule, denorm_output_node: Node
) -> Node:
    """
    Inserts linear bias application after the denormalization node.
    """
    with graph_module.graph.inserting_after(denorm_output_node):
        bias_add_node = graph_module.graph.create_node(
            op="call_function",
            target=torch.ops.aten.add.Tensor,
            args=(),
        )
        denorm_output_node.replace_all_uses_with(bias_add_node)

    return bias_add_node


def _insert_later_bias_application(
    graph_module: GraphModule,
    denorm_output_node: Node,
    linear_node: Node,
    linear_b_node: Node,
):
    """
    Moves bias application from the normalized space to after simulated denormalization.
    """
    _ = _insert_disabled_linear_bias(graph_module, linear_b_node)

    bias_add_node = _insert_bias_add_after_denorm(graph_module, denorm_output_node)

    with graph_module.graph.inserting_before(bias_add_node):
        linear_output_shape = linear_node.meta.get("tensor_meta").shape
        reshape_target_shape = [1] * len(linear_output_shape)

        bias_redirect_node = linear_b_node

        if len(reshape_target_shape) > 1:
            reshape_target_shape[1] = -1
            bias_redirect_node = graph_module.graph.create_node(
                op="call_function",
                target=torch.ops.aten.reshape,
                args=(linear_b_node, reshape_target_shape),
            )

        bias_add_node.insert_arg(0, denorm_output_node)
        bias_add_node.insert_arg(1, bias_redirect_node)


class AddSimulatedLinearBatchNormFusionQATPass(PassBase):
    """
    Batch norm computation can be in some cases fused with preceding linear transformation like conv or linear (see fuse_batch_norm_with_linear_pass.py).
    We cannot do this before the QAT training because we would lose the ability to update the batch norm statistics during QAT training.
    This pass takes inspiration from already existing mechanics of simulated Conv+BN folding present in TorchAO implementation, and applies it to a torch.nn.Linear.
    The implementation can be found in _fuse_conv_bn_qat function [1] that is being used in _prepare_qat_pt2e [2].

    We simulate the fusion by:
    1. Adding computation of the fused scale factor from batch norm parameters to the graph
    2. Adding pre-processing of the linear weights with this scale factor and applying inverse to its output
    3. (Optionally) Moving bias application after the output denormalization to work in the same space it was originally pre-trained in
    4. Keeping the batch norm operator in the graph for statistics tracking during QAT

    Linear weight scaling (necessary reshape nodes omitted for clarity):

                                                 ┌───────────┐                               ┌───────────┐
                                                 │bn_run_var │                               │ bn_weight │
                                                 └─────┬─────┘                               └─────┬─────┘
                                                       │         ┌─────────────────────────────────┤
    ┌─────────────┐              ┌─────┐               │  ┌──────┴──────┐   ┌─────────────┐        │
    │linear_weight│              │  x  │               ├──┤compute_scale│   │linear_weight│        │
    └──────┬──────┘              └──┬──┘               │  └──────┬──────┘   └──────┬──────┘        │
           │                        │                  │         │                 │               │
           │    FQ┌──────────┐FQ    │                  │         │     ┌─────┐     │      ┌─────┐  │
           └──────┤  linear  ├──────┘                  │         ├─────┤ mul ├─────┘      │  x  │  │
                  └─────┬────┘             ──────►     │         │     └──┬──┘            └──┬──┘  │
                        │                              │         │        │                  │     │
     ┌───────────┐      │     ┌───────────┐            │         │        │ FQ┌──────────┐FQ │     │
     │ bn_weight │      │     │bn_run_var │            │         │        └───┤  linear  ├───┘     │
     └─────┬─────┘      │     └─────┬─────┘            │         │            └─────┬────┘         │
           │            │           │                  │         │                  │              │
           │      ┌─────┴────┐      │                  │         │               ┌──┴──┐           │
           └──────┤batch_norm├──────┘                  │         └───────────────┤ div │           │
                  └─────┬────┘                         │                         └──┬──┘           │
                        │                              │                            │              │
                        ▼                              │                      ┌─────┴────┐         │
                                                       └──────────────────────┤batch_norm├─────────┘
                                                                              └─────┬────┘
                                                                                    │
                                                                                    ▼
    Note: compute_scale := (bn_weight / sqrt(bn_run_var + eps))


    Later bias application (necessary reshape nodes omitted for clarity):

    ┌─────────────┐      ┌─────┐    ┌────────────┐         ┌─────────────┐      ┌─────┐    ┌────────────┐
    │linear_weight│      │  x  │    │linear_bias │         │linear_weight│      │  x  │    │linear_bias │
    └──────┬──────┘      └──┬──┘    └──────┬─────┘         └──────┬──────┘      └──┬──┘    └──────┬─────┘
           │                │              │                      │                │              │
           │        FQ┌─────┴────┐FQ       │                      │        FQ┌─────┴────┐FQ       │
           └──────────┤  linear  ├─────────┘                      └──────────┤  linear  ├─ ─ ─ ─ ─┤
                      └─────┬────┘                                           └─────┬────┘  ZEROS  │
                            │                       ───────►                       │              │
     ┌───────────┐          │        ┌───────────┐            ┌───────────┐     ┌──┴──┐           │   ┌───────────┐
     │ bn_weight │          │        │bn_run_var │            │ bn_weight │     │ add ├───────────┘   │bn_run_var │
     └─────┬─────┘          │        └─────┬─────┘            └─────┬─────┘     └──┬──┘               └─────┬─────┘
           │                │              │                        │              │                        │
           │          ┌─────┴────┐         │                        │        ┌─────┴────┐                   │
           └──────────┤batch_norm├─────────┘                        └────────┤batch_norm├───────────────────┘
                      └─────┬────┘                                           └─────┬────┘
                            │                                                      │
                            ▼                                                      ▼

    [1] https://github.com/pytorch/ao/blob/main/torchao/quantization/pt2e/quantize_pt2e.py
    [2] https://github.com/pytorch/ao/blob/main/torchao/quantization/pt2e/qat_utils.py
    """

    def call(self, graph_module: GraphModule) -> PassResult | None:
        """
        Given a graph of decomposed aten ops, adds linear weights normalization and linear output denormalization based on batch norm stats.
        Optionally, applies later Linear bias application if Linear has bias=True selected.

        The normalization follows this equation:
        linear_w_fused = linear_w * (gamma / sqrt(var + eps))

        while `gamma` being the batch norm weights.

        The denormalization is done by dividing the linear layer output by the scale factor:
        y_denorm = y / (gamma / sqrt(var + eps))

        Normalization and denormalization operators should be removed by the
        RemoveSimulatedLinearBatchNormFusionQATPass after the QAT training is complete.
        """
        modified = False

        named_modules = dict(graph_module.named_modules(remove_duplicate=False))

        for node in graph_module.graph.nodes:
            if not _is_batch_norm(node):
                continue

            bn_node = node
            bn_in = bn_node.args[0]

            if not _is_linear(bn_in):
                continue

            linear_node = bn_in

            linear_w_node_or_fq = linear_node.args[1]
            linear_b_node_or_fq = (
                linear_node.args[2] if len(linear_node.args) >= 3 else None
            )
            linear_w_node = _unwrap_if_fq(
                linear_w_node_or_fq, named_modules=named_modules
            )
            linear_b_node = _unwrap_if_fq(
                linear_b_node_or_fq, named_modules=named_modules
            )
            linear_w_is_quantized = linear_w_node_or_fq != linear_w_node

            bn_w_node = bn_node.args[1]
            bn_var_node = bn_node.args[4]

            # BatchNorm(affine=False)
            if bn_w_node is None:
                continue

            # BatchNorm should not have quantized inputs
            if _is_activation_post_process_node(
                bn_w_node, named_modules=named_modules
            ) or _is_activation_post_process_node(
                bn_var_node, named_modules=named_modules
            ):
                continue

            bn_w_shape = _get_tensor_shape(bn_w_node)
            bn_var_shape = _get_tensor_shape(bn_var_node)
            linear_w_shape = _get_tensor_shape(linear_w_node)

            if None in (bn_w_shape, bn_var_shape, linear_w_shape):
                continue

            scale_factor_example_inputs = (
                torch.randn(bn_w_shape),
                torch.randn(bn_var_shape),
            )
            norm_linear_w_example_inputs = (
                torch.randn(linear_w_shape),
                torch.randn(bn_var_shape),
            )

            # Replacement patterns generation
            bn_eps = _get_bn_eps(node)
            bn_scale_factor_fn = _get_compute_scale_factor_pattern(bn_eps=bn_eps)
            norm_linear_w_fn = _get_linear_weight_preprocess_pattern()
            is_cuda = False
            bn_scale_factor_fn = _get_aten_graph_module_for_pattern(
                pattern=bn_scale_factor_fn,
                example_inputs=scale_factor_example_inputs,
                is_cuda=is_cuda,
            )
            normalize_linear_w_fn = _get_aten_graph_module_for_pattern(
                pattern=norm_linear_w_fn,
                example_inputs=norm_linear_w_example_inputs,
                is_cuda=is_cuda,
            )
            bn_w_replacement_node, bn_var_replacement_node = _get_input_nodes(
                bn_scale_factor_fn
            )
            linear_w_replacement_node, scale_factor_out_replacement_node = (
                _get_input_nodes(normalize_linear_w_fn)
            )

            insertion_point = (
                linear_w_node_or_fq if linear_w_is_quantized else linear_node
            )

            # BN var and W node definition needs to be moved before the linear layer to avoid
            # using them before definition.
            bn_var_node = _reinsert_node_before(
                graph_module, node=bn_var_node, insertion_point=insertion_point
            )
            bn_w_node = _reinsert_node_before(
                graph_module, node=bn_w_node, insertion_point=insertion_point
            )

            with graph_module.graph.inserting_before(insertion_point):
                mapping = {
                    bn_var_replacement_node: bn_var_node,
                    bn_w_replacement_node: bn_w_node,
                }
                scale_factor_out_nodes = graph_module.graph.graph_copy(
                    bn_scale_factor_fn.graph, val_map=mapping
                )

                mapping = {
                    linear_w_replacement_node: linear_w_node,
                    scale_factor_out_replacement_node: scale_factor_out_nodes[0],
                }
                output_node = graph_module.graph.graph_copy(
                    normalize_linear_w_fn.graph, val_map=mapping
                )

                if linear_w_is_quantized:
                    linear_w_node_or_fq.update_arg(0, output_node[0])
                else:
                    linear_node.update_arg(1, output_node[0])

            div_node = _insert_linear_output_denorm(
                graph_module,
                linear_node=linear_node,
                scale_factor=scale_factor_out_nodes[0],
                bn_node=bn_node,
            )

            if linear_b_node is not None:
                _insert_later_bias_application(
                    graph_module,
                    denorm_output_node=div_node,
                    linear_node=linear_node,
                    linear_b_node=linear_b_node,
                )

            modified = True

        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()

        return PassResult(graph_module, modified)
