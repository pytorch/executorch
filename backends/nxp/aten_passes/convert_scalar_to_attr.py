# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix

from torch.export.unflatten import _assign_attr, _AttrKind
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult


class ConvertScalarToAttrPass(PassBase):
    """
    Convert scalar (Python `int`/`float`) arguments of elementwise operators into
    `get_attr` nodes holding a constant tensor. For example `aten.mul.Tensor(x, 2.0)`
    is rewritten so the scalar `2.0` becomes a `get_attr` node referencing a tensor
    constant with the value `2.0`.

                       x                                                     x     get_attr(2.0)
                       |                                                     |    /
        ┌──────────────▼─────────────┐     replace with     ┌────────────────▼───────────────┐
        |   aten.mul.Tensor(x, 2.0)  |  ─────────────────►  |  aten.mul.Tensor(x, get_attr)  |
        └──────────────┬─────────────┘                      └────────────────┬───────────────┘
                       |                                                     |
                       v                                                     v
                      out                                                   out

    If not done, the scalar arguments prevent from proper QDQ pattern utilization and
    the operator cannot be delegated to Neutron.
    """

    @staticmethod
    def _get_ref_dtype(node: Node) -> torch.dtype:
        # Infer the constant dtype from a sibling tensor input of the operator so
        # the created constant matches the operator's tensor operand.
        # Fall back to the node's own output dtype, then `float32`.
        for arg in node.args:
            if isinstance(arg, Node):
                arg_val = arg.meta.get("val")
                if isinstance(arg_val, torch.Tensor):
                    return arg_val.dtype

        node_val = node.meta.get("val")
        if isinstance(node_val, torch.Tensor):
            return node_val.dtype

        return torch.float32

    def _create_scalar_attr_node(self, node: Node, scalar_value: int | float) -> Node:
        # Create a constant tensor holding the scalar value.
        tensor = torch.tensor(scalar_value, dtype=self._get_ref_dtype(node))
        tensor_name = get_new_attr_name_with_prefix("_scalar_const_")(self.graph_module)

        _assign_attr(
            torch.nn.Parameter(tensor, requires_grad=False),
            self.graph_module,
            tensor_name,
            _AttrKind.PARAMETER,
        )

        fake_mode = node.meta["val"].fake_mode
        with self.graph_module.graph.inserting_before(node):
            get_attr_node = self.graph_module.graph.create_node(
                "get_attr", tensor_name, (), {}
            )
            get_attr_node.meta["val"] = fake_mode.from_tensor(
                tensor, static_shapes=True
            )

        return get_attr_node

    def call(self, graph_module: GraphModule) -> PassResult:
        self.graph_module = graph_module
        made_changes = False

        for node in list(graph_module.graph.nodes):
            if node.op != "call_function":
                continue

            new_args = list(node.args)
            node_changed = False
            for i, arg in enumerate(new_args):
                # Note: `bool` is subclass on `int`.
                if isinstance(arg, bool) or not isinstance(arg, (int, float)):
                    continue

                new_args[i] = self._create_scalar_attr_node(node, arg)
                node_changed = True

            if node_changed:
                node.args = tuple(new_args)
                made_changes = True

        self.graph_module.graph.eliminate_dead_code()
        self.graph_module.recompile()

        return PassResult(graph_module, made_changes)
