# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.nxp.backend.edge_helper import (
    try_get_tensor_constant_from_node,
)
from torch._subclasses import FakeTensor, FakeTensorMode
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from torch.export.unflatten import _assign_attr, _AttrKind
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult


Conv1dArgs = tuple[Node, Node, (Node | None), list[int], list[int], list[int], int]
Conv1dTranspArgs = tuple[
    Node, Node, (Node | None), list[int], list[int], list[int], int, list[int]
]


class ConvertConv1dToConv2dPass(PassBase):
    r"""
    The NXP backend supports only 2D convolutions. Rewrite 1D convolutions into an equivalent 2D form by
    inserting a singleton spatial dimension and then remove it again.
    If batch norm is present after the convolution, is is also converted from 1D to 2D.

    Without batch norm:

           x                         W                                x                           W
      [N, C1, H]               [I/O, I/O, k]                     [N, C1, H]                [I/O, I/O, 1, k]
           │                         │                                │                           │
           │                         │                      ┌─────────▼──────────┐                │
           │                         │                      │  unsqueeze(x, -2)  │                │
           │                         │                      └─────────▼──────────┘                │
           │                         │                                │                           │
           │                         │                         [N, C1, 1, H ]                     │
           │                         │                                │                           │
           └────────┐       ┌────────┘                                └──────────┐     ┌──────────┘
                    │       │                                                    │     │
           ┌────────▼───────▼───────┐                                   ┌────────▼─────▼────────┐
           │       convolution      ◄──B [O]        replace             │      convolution      ◄──B [O]
           │   (1D/transposed 1D)   │          ────────────────►        │   (2D/transposed 2D)  │
           └────────────┬───────────┘                with               └───────────┬───────────┘
                        │                                                           │
                        │                                                     [N, C2, 1, H]
                        │                                                           │
                        │                                                 ┌─────────▼──────────┐
                        │                                                 │   squeeze(x, -2)   │
                        │                                                 └─────────┬──────────┘
                        │                                                           │
                        ▼                                                           ▼
                   [N, C2, H]                                                  [N, C2, H]
                        y                                                           y

    With batch norm:

           x                         W                                x                           W
      [N, C1, H]               [I/O, I/O, k]                     [N, C1, H]                [I/O, I/O, 1, k]
           │                         │                                │                           │
           │                         │                      ┌─────────▼──────────┐                │
           │                         │                      │  unsqueeze(x, -2)  │                │
           │                         │                      └─────────▼──────────┘                │
           │                         │                                │                           │
           │                         │                         [N, C1, 1, H]                      │
           │                         │                                │                           │
           └────────┐       ┌────────┘                                └──────────┐     ┌──────────┘
                    │       │                                                    │     │
           ┌────────▼───────▼───────┐                                   ┌────────▼─────▼────────┐
           │       convolution      ◄──B [O]        replace             │      convolution      ◄──B [O]
           │   (1D/transposed 1D)   │          ────────────────►        │   (2D/transposed 2D)  │
           └────────────┬───────────┘                with               └───────────┬───────────┘
                        │                                                           │
                  [N, C2, 1, H]                                               [N, C2, 1, H]
                        │                                                           │
                ┌───────▼───────┐                                           ┌───────▼───────┐
                │   batch_norm  │                                           │   batch_norm  │
                │      (1D)     │                                           │      (2D)     │
                └───────┬───────┘                                           └───────┬───────┘
                        │                                                           │
                        │                                                     [N, C3, 1, H]
                        │                                                           │
                        │                                                   ┌───────▼────────┐
                        │                                                   │   squeeze(-2)  │
                        │                                                   └───────┬────────┘
                        │                                                           │
                        ▼                                                           ▼
                    [N, C3, H]                                                  [N, C3, H]
                        y                                                           y
    """

    @staticmethod
    def _is_conv_1d(node: Node) -> bool:
        return node.target == torch.ops.aten.conv1d.default

    @staticmethod
    def _is_conv_transposed_1d(node: Node) -> bool:
        return node.target == torch.ops.aten.conv_transpose1d.default

    @staticmethod
    def _is_batch_norm(node: Node) -> bool:
        return node.target == torch.ops.aten.batch_norm.default

    @staticmethod
    def _listify(x: int | list[int] | tuple[int]) -> list[int]:
        if isinstance(x, int):
            return [x]

        return list(x)

    def _get_node_shape(self, node: Node):
        node_t = try_get_tensor_constant_from_node(self.graph_module, node)
        if node_t is not None:
            return node_t.shape

        return node.meta["val"].shape if hasattr(node, "meta") else node.shape

    def _get_node_dtype(self, node: Node):
        node_t = try_get_tensor_constant_from_node(self.graph_module, node)

        if node_t is not None:
            return node_t.dtype

        return node.meta["val"].dtype if hasattr(node, "meta") else node.dtype

    def _convert_w_node_to_static_attr(self, node: Node):
        t_node = try_get_tensor_constant_from_node(self.graph_module, node)
        if t_node is None:
            # should not occur
            raise RuntimeError(
                "Node cannot be converted to `get_attr` since it is not static."
            )
        t_node = t_node.unsqueeze(-2)

        t_name = get_new_attr_name_with_prefix(node.name)(self.graph_module)
        _assign_attr(
            torch.nn.Parameter(t_node),
            self.graph_module,
            t_name,
            _AttrKind.PARAMETER,
        )

        get_attr_node = self.graph_module.graph.create_node("get_attr", t_name, (), {})
        fake_mode = node.meta["val"].fake_mode
        get_attr_node.meta["val"] = fake_mode.from_tensor(t_node, static_shapes=True)

        return get_attr_node

    def _create_fake_tensor_for_node_args(
        self, node_args: list[Node | None], mode: FakeTensorMode
    ):
        fake_node_args = [
            (
                FakeTensor.from_tensor(
                    torch.empty(
                        self._get_node_shape(arg), dtype=self._get_node_dtype(arg)
                    ),
                    mode,
                )
                if arg is not None
                else None
            )
            for arg in node_args
        ]

        return fake_node_args

    def _create_batch_norm_2d_node(self, *bn_args):
        bn_target = torch.ops.aten.batch_norm.default
        bn_node = self.graph_module.graph.call_function(bn_target, bn_args)

        bn_node.meta["source_fn_stack"] = [(bn_node.name, bn_target)]

        node_args = bn_args[:5]
        scalar_args = bn_args[5:]

        with FakeTensorMode() as mode:
            fake_node_args = self._create_fake_tensor_for_node_args(node_args, mode)
            output = bn_target(*fake_node_args, *scalar_args)

            bn_node.meta["val"] = FakeTensor.from_tensor(
                torch.empty(output.shape, dtype=output.dtype), mode
            )

        return bn_node

    def _create_some_conv_2d_node(self, target, *conv_args):
        # some_conv_2d_node = could be regular 2d conv or transposed 2d conv
        some_conv_node = self.graph_module.graph.call_function(target, conv_args)
        some_conv_node.meta["source_fn_stack"] = [(some_conv_node.name, target)]

        node_args = conv_args[:3]
        scalar_args = conv_args[3:]

        with FakeTensorMode() as mode:
            fake_node_args = self._create_fake_tensor_for_node_args(node_args, mode)
            output = target(*fake_node_args, *scalar_args)

            some_conv_node.meta["val"] = FakeTensor.from_tensor(
                torch.empty(output.shape, dtype=output.dtype), mode
            )

        return some_conv_node

    def _create_sq_or_unsq_node(self, target, *sq_or_unsq_args) -> Node:
        sq_or_unsq_node = self.graph_module.graph.call_function(target, sq_or_unsq_args)

        sq_or_unsq_node.meta["source_fn_stack"] = [(sq_or_unsq_node.name, target)]
        with FakeTensorMode() as mode:
            inp_node = sq_or_unsq_args[0]
            fake_input = FakeTensor.from_tensor(
                torch.empty(
                    self._get_node_shape(inp_node), dtype=self._get_node_dtype(inp_node)
                ),
                mode,
            )

            output = target(fake_input, *sq_or_unsq_args[1:])
            sq_or_unsq_node.meta["val"] = FakeTensor.from_tensor(
                torch.empty(output.shape, dtype=output.dtype), mode
            )

        return sq_or_unsq_node

    @staticmethod
    def _get_conv_1d_transp_args(node: Node):
        args = node.args
        listify_fn = ConvertConv1dToConv2dPass._listify

        b_node = None if len(args) < 3 else args[2]
        stride = [1] if len(args) < 4 else listify_fn(args[3])
        padding = [0] if len(args) < 5 else listify_fn(args[4])
        output_padding = [0] if len(args) < 6 else listify_fn(args[5])
        groups = 1 if len(args) < 7 else args[6]
        dilation = [1] if len(args) < 8 else listify_fn(args[7])

        return (
            args[0],
            args[1],
            b_node,
            stride,
            padding,
            output_padding,
            groups,
            dilation,
        )

    @staticmethod
    def _get_conv_1d_args(node: Node) -> Conv1dArgs:
        args = node.args
        listify_fn = ConvertConv1dToConv2dPass._listify

        b_node = None if len(args) < 3 else args[2]
        stride = [1] if len(args) < 4 else listify_fn(args[3])
        padding = [0] if len(args) < 5 else listify_fn(args[4])
        dilation = [1] if len(args) < 6 else listify_fn(args[5])
        groups = 1 if len(args) < 7 else args[6]

        return args[0], args[1], b_node, stride, padding, dilation, groups

    def _convert_scalar_1d_args_to_2d(self, old_1d_node: Node):
        if self._is_conv_transposed_1d(old_1d_node):
            _, _, _, stride, pad, output_pad, groups, dil = (
                self._get_conv_1d_transp_args(old_1d_node)
            )

            # conversion of 1d args to 2d, ie. padding with default values
            stride = [1] + stride
            pad = [0] + pad
            output_pad = [0] + output_pad
            dil = [1] + dil

            return stride, pad, output_pad, groups, dil

        else:
            _, _, _, stride, pad, dil, groups = self._get_conv_1d_args(old_1d_node)

            # conversion of 1d args to 2d, ie. padding with default values
            stride = [1] + stride
            pad = [0] + pad
            dil = [1] + dil

            return stride, pad, dil, groups

    def _convert_node_1d_args_to_2d(self, old_1d_node: Node):
        if self._is_conv_transposed_1d(old_1d_node):
            input_node, w_node, b_node, _, _, _, _, _ = self._get_conv_1d_transp_args(
                old_1d_node
            )
        else:
            input_node, w_node, b_node, _, _, _, _ = self._get_conv_1d_args(old_1d_node)

        with self.graph_module.graph.inserting_before(old_1d_node):
            # weights = [i/o, i/o, k] => [i/o, i/o, 1, k] and converted to `get_attr` node
            w_node = self._convert_w_node_to_static_attr(w_node)

            # input = [n, c, h] => [n, c, 1, h]
            unsqueeze_target = torch.ops.aten.unsqueeze.default
            inp_unsq_args = (input_node, -2)
            inp_unsq_node = self._create_sq_or_unsq_node(
                unsqueeze_target, *inp_unsq_args
            )

        return (inp_unsq_node, w_node, b_node)

    def call(self, graph_module: GraphModule) -> PassResult:
        self.graph_module = graph_module
        made_changes = False

        for node in list(graph_module.graph.nodes):
            is_conv_1d = self._is_conv_1d(node)
            is_conv_1d_transp = self._is_conv_transposed_1d(node)

            # some_1d_conv = regular 1d conv or 1d transposed conv
            is_some_1d_conv = is_conv_1d or is_conv_1d_transp
            if not is_some_1d_conv:
                continue

            old_1d_node = node

            # invalid number of args
            if len(old_1d_node.args) < 2:
                continue

            conv_1d_w = old_1d_node.args[1]
            conv_1d_b = old_1d_node.args[2] if len(old_1d_node.args) > 2 else None

            # non-static weights are not supported
            if try_get_tensor_constant_from_node(graph_module, conv_1d_w) is None:
                continue

            # non-static bias is not supported
            if (
                conv_1d_b is not None
                and try_get_tensor_constant_from_node(graph_module, conv_1d_b) is None
            ):
                continue

            # get input, weight and bias arguments for the new 2d conv
            node_args = self._convert_node_1d_args_to_2d(old_1d_node)
            # get stride, padding etc. arguments for the new 2d conv
            scalar_args = self._convert_scalar_1d_args_to_2d(old_1d_node)

            new_2d_target = (
                torch.ops.aten.conv_transpose2d.input
                if is_conv_1d_transp
                else torch.ops.aten.conv2d.default
            )

            # create the new conv 2d and unsqueeze the input and weights
            with self.graph_module.graph.inserting_before(old_1d_node):
                new_2d_args = node_args + scalar_args
                new_2d_node = self._create_some_conv_2d_node(
                    new_2d_target, *new_2d_args
                )

            old_1d_conv_users = list(old_1d_node.users.keys())
            if len(old_1d_conv_users) == 1 and self._is_batch_norm(
                old_1d_conv_users[0]
            ):
                bn_1d_node = old_1d_conv_users[0]

                # also convert batch_norm 1d to 2d
                with self.graph_module.graph.inserting_after(new_2d_node):
                    bn_2d_args = (new_2d_node,) + bn_1d_node.args[1:]
                    bn_2d_node = self._create_batch_norm_2d_node(*bn_2d_args)

                with self.graph_module.graph.inserting_after(bn_2d_node):
                    squeeze_target = torch.ops.aten.squeeze.dim

                    out_sq_args = (bn_2d_node, 2)
                    out_sq_node = self._create_sq_or_unsq_node(
                        squeeze_target, *out_sq_args
                    )

                bn_1d_node.replace_all_uses_with(out_sq_node)
                self.graph_module.graph.erase_node(bn_1d_node)

            else:
                with self.graph_module.graph.inserting_after(new_2d_node):
                    squeeze_target = torch.ops.aten.squeeze.dim

                    out_sq_args = (new_2d_node, -2)
                    out_sq_node = self._create_sq_or_unsq_node(
                        squeeze_target, *out_sq_args
                    )

                old_1d_node.replace_all_uses_with(out_sq_node)

            graph_module.graph.erase_node(old_1d_node)
            made_changes = True

        graph_module.recompile()
        graph_module.graph.eliminate_dead_code()
        return PassResult(graph_module, made_changes)
