# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import operator
from collections.abc import Collection

import torch
from executorch.backends.transforms.utils import (
    create_constant_placeholder,
    get_param_tensor,
    is_param_node,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import (
    ExportedProgramPassBase,
    ExportedProgramPassResult,
    ExportPass,
    PassResult,
)
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind
from torch.fx import GraphModule, Node
from torch.nn.utils.fusion import fuse_conv_bn_weights


class FuseBatchNormWithConvPass(ExportPass):
    """Fold frozen evaluation BatchNorm into a single-consumer convolution.

    GraphModule callers retain get_attr parameters. Use
    FuseBatchNormWithConvProgramPass for signature-aware lifted constants.
    """

    def __init__(
        self,
        exported_program: ExportedProgram,
        *,
        mutable_buffers: Collection[str] = (),
    ) -> None:
        super().__init__()
        self.exported_program = exported_program
        self.mutable_buffers = frozenset(mutable_buffers)

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = self._fuse_graph(graph_module, lift_constants=False)
        if modified:
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, modified)

    def _fuse_graph(self, graph_module: GraphModule, *, lift_constants: bool) -> bool:
        graph = graph_module.graph
        for node in graph.nodes:
            target = node.target
            if isinstance(target, EdgeOpOverload):
                target = target._op
            if isinstance(target, torch._ops.OpOverload) and target._schema.is_mutable:
                return False
        modified = False
        for conv in list(graph.nodes):
            if (
                conv.target != exir_ops.edge.aten.convolution.default
                or len(conv.users) != 1
            ):
                continue
            bn = next(iter(conv.users))
            if not self.can_fuse(conv, bn, self.exported_program):
                continue
            args = (conv.args[1], conv.args[2], *bn.args[1:5])
            if any(
                isinstance(arg, Node)
                and self.exported_program.graph_signature.inputs_to_buffers.get(
                    arg.name
                )
                in self.mutable_buffers
                for arg in args
            ):
                continue
            weight, bias, scale, shift, mean, var = (
                get_param_tensor(self.exported_program, arg) for arg in args
            )
            with unset_fake_temporarily(), torch.no_grad():
                fused_weight, fused_bias = self._fuse_conv_bn_weights(
                    weight,
                    bias,
                    mean,
                    var,
                    bn.args[-1],
                    scale,
                    shift,
                    transposed=bool(conv.args[6]),
                    groups=conv.args[8],
                )
                weight_node = self._fused_constant(conv, fused_weight, lift_constants)
                bias_node = self._fused_constant(conv, fused_bias, lift_constants)
            conv.args = (conv.args[0], weight_node, bias_node, *conv.args[3:])
            for user in list(bn.users):
                user.replace_all_uses_with(conv)
                graph.erase_node(user)
            graph.erase_node(bn)
            modified = True
        if modified:
            graph.lint()
            graph_module.recompile()
        return modified

    def _fused_constant(self, conv: Node, data: torch.Tensor, lift: bool) -> Node:
        data = data.detach()
        stem = "_fused_bn"
        name = stem
        program = self.exported_program
        graph = conv.graph
        occupied = set(program.state_dict) | set(program.constants)
        occupied.update(graph._graph_namespace._used_names)
        index = 0
        while name in occupied:
            index += 1
            name = f"{stem}_{index}"
        if lift:
            with graph.inserting_before(next(iter(graph.nodes))):
                return create_constant_placeholder(
                    program, graph, name, InputKind.CONSTANT_TENSOR, data
                )
        graph.owning_module.register_parameter(
            name, torch.nn.Parameter(data, requires_grad=False)
        )
        with graph.inserting_before(conv):
            return graph.get_attr(name)

    @staticmethod
    def can_fuse(conv: Node, bn: Node, program: ExportedProgram) -> bool:
        if (
            conv.target != exir_ops.edge.aten.convolution.default
            or len(conv.args) != 9
            or conv.kwargs
            or len(conv.users) != 1
        ):
            return False
        if bn.target == exir_ops.edge.aten.native_batch_norm.default:
            if len(bn.args) != 8 or bn.args[5] is not False:
                return False
        elif (
            bn.target == exir_ops.edge.aten._native_batch_norm_legit_no_training.default
        ):
            if len(bn.args) != 7:
                return False
        else:
            return False
        if (
            bn.kwargs
            or bn.args[0] is not conv
            or not bn.users
            or any(
                user.target is not operator.getitem or user.args != (bn, 0)
                for user in bn.users
            )
            or conv.args[1] is None
            or bn.args[3] is None
            or bn.args[4] is None
        ):
            return False
        signature = program.graph_signature
        for arg in (conv.args[1], conv.args[2], *bn.args[1:5]):
            if arg is None:
                continue
            if not isinstance(arg, Node) or not is_param_node(program, arg):
                return False
            if (
                signature.inputs_to_buffers.get(arg.name)
                in signature.buffers_to_mutate.values()
            ):
                return False
        return True

    @staticmethod
    def _fuse_conv_bn_weights(
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor | None,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        eps: float,
        bn_weight: torch.Tensor | None,
        bn_bias: torch.Tensor | None,
        transposed: bool,
        groups: int,
    ) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
        """
        A transposed conv weight is [in, out/groups, *kernel], so its output
        channels sit on dim 1, split into one block per group along dim 0.
        Regroup it to [in/groups, out, *kernel] so dim 1 holds every output
        channel, fold with transpose=True, then restore the original layout.
        """
        if not transposed or groups == 1:
            return fuse_conv_bn_weights(
                conv_weight,
                conv_bias,
                running_mean,
                running_var,
                eps,
                bn_weight,
                bn_bias,
                transpose=transposed,
            )

        in_channels, out_per_group, *kernel = conv_weight.shape
        grouped_shape = (groups, in_channels // groups, out_per_group, *kernel)
        regrouped = (
            conv_weight.reshape(grouped_shape)
            .transpose(0, 1)
            .reshape(in_channels // groups, groups * out_per_group, *kernel)
        )
        fused_weight, fused_bias = fuse_conv_bn_weights(
            regrouped,
            conv_bias,
            running_mean,
            running_var,
            eps,
            bn_weight,
            bn_bias,
            transpose=True,
        )
        fused_weight = torch.nn.Parameter(
            fused_weight.detach()
            .reshape(in_channels // groups, groups, out_per_group, *kernel)
            .transpose(0, 1)
            .reshape(conv_weight.shape),
            fused_weight.requires_grad,
        )
        return fused_weight, fused_bias


class FuseBatchNormWithConvProgramPass(ExportedProgramPassBase):
    """ExportedProgram adapter for the shared Conv–BatchNorm transformation."""

    def __init__(self, mutable_buffers: Collection[str] = ()) -> None:
        super().__init__()
        self.mutable_buffers = frozenset(mutable_buffers)

    def call(self, program: ExportedProgram) -> ExportedProgramPassResult:
        fusion = FuseBatchNormWithConvPass(
            program, mutable_buffers=self.mutable_buffers
        )
        modified = fusion._fuse_graph(program.graph_module, lift_constants=True)
        return ExportedProgramPassResult(program, modified)
