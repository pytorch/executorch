# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import executorch.backends.nxp.backend.ir.logger as logger
import torch

from executorch.backends.nxp.backend.edge_helper import (
    try_get_tensor_constant_from_node,
)
from torch._subclasses import FakeTensor, FakeTensorMode
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from torch.export.unflatten import _assign_attr, _AttrKind
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult


class ConvertDivToMulPass(PassBase):
    """
    Replace `aten.div.Tensor` with `aten.mul.Tensor` by multiplying the quotient
    with the reciprocal of the divisor (1 / divisor), when the divisor is known
    at compile time or the divisor is a scalar (one number).

                        x                                                         x
                        │                                                         │
        ┌───────────────▼───────────────┐      replace with      ┌────────────────▼────────────────┐
        │  aten.div.Tensor(x, divisor)  │   ─────────────────►   │ aten.mul.Tensor(x, 1 / divisor) │
        └───────────────┬───────────────┘                        └────────────────┬────────────────┘
                        │                                                         │
                        ▼                                                         ▼
                       out                                                       out
    """

    @staticmethod
    def _is_div_tensor(node_: Node) -> bool:
        return node_.op == "call_function" and node_.target == torch.ops.aten.div.Tensor

    def _create_mul_op_node(self, *mul_args) -> Node:
        mul_target = torch.ops.aten.mul.Tensor
        mul_node = self.graph_module.graph.call_function(mul_target, mul_args)

        mul_node.meta["source_fn_stack"] = [(mul_node.name, mul_target)]

        multiplier_1 = mul_args[0]
        multiplier_2 = mul_args[1]
        mul_1_val = multiplier_1.meta["val"]
        mul_2_val = multiplier_2.meta["val"]
        with FakeTensorMode() as mode:
            fake_input_1 = FakeTensor.from_tensor(
                torch.empty(mul_1_val.shape, dtype=mul_1_val.dtype), mode
            )
            fake_input_2 = FakeTensor.from_tensor(
                torch.empty(mul_2_val.shape, dtype=mul_2_val.dtype), mode
            )

            output_shape = mul_target(fake_input_1, fake_input_2).shape
            mul_node.meta["val"] = FakeTensor.from_tensor(
                torch.empty(output_shape, dtype=mul_1_val.dtype), mode
            )

        return mul_node

    def convert_divisor_to_multiplier(self, div_node: Node) -> torch.Tensor | None:
        if len(div_node.args) <= 1:
            return None

        divisor_raw = div_node.args[1]
        check_zero_atol = 1e-12
        match divisor_raw:
            # the `divisor` is a scalar - represented by a value in the node's `args`
            case int() | float():
                # check for division by zero
                if math.isclose(divisor_raw, 0.0, rel_tol=0.0, abs_tol=check_zero_atol):
                    logger.e(
                        logger.Code.INVALID_INPUT_MODEL,
                        f"Div node could not be converted to mul because the divisor is zero or close to zero with absolute tolerance of {check_zero_atol}.",
                    )
                    return None

                quotient_shape = div_node.all_input_nodes[0].meta["val"].shape
                multiplier_val = 1.0 / divisor_raw

                return torch.full(quotient_shape, multiplier_val, dtype=torch.float32)

            # the `divisor` is a tensor - represented as a node in the graph
            case Node():
                divisor_t = try_get_tensor_constant_from_node(
                    self.graph_module, divisor_raw
                )

                # check if tensor `divisor` is static
                if divisor_t is None:
                    return None

                # check for division by zero
                if (
                    torch.isclose(
                        divisor_t,
                        torch.zeros_like(divisor_t),
                        rtol=0.0,
                        atol=check_zero_atol,
                    )
                    .all()
                    .item()
                ):
                    logger.e(
                        logger.Code.INVALID_INPUT_MODEL,
                        f"Div node could not be converted to mul because the divisor is zero or close to zero with absolute tolerance of {check_zero_atol}.",
                    )
                    return None

                return 1.0 / divisor_t

            case _:
                return None

    def _create_mul_data_node(self, div_node: Node) -> Node | None:
        # get `multiplier` tensor, i.e. multiplier = (1 / divisor)
        multiplier_t = self.convert_divisor_to_multiplier(div_node)
        if multiplier_t is None:
            return None

        # create new node for the `multiplier` tensor and insert it into graph
        tensor_name = get_new_attr_name_with_prefix("multiplier_")(self.graph_module)
        _assign_attr(
            torch.nn.Parameter(multiplier_t),
            self.graph_module,
            tensor_name,
            _AttrKind.PARAMETER,
        )

        fake_mode = div_node.meta["val"].fake_mode
        with self.graph_module.graph.inserting_before(div_node):
            get_attr_node = self.graph_module.graph.create_node(
                "get_attr", tensor_name, (), {}
            )
            get_attr_node.meta["val"] = fake_mode.from_tensor(
                multiplier_t, static_shapes=True
            )
            multiplier_node = get_attr_node

        return multiplier_node

    def _erase_divisor_from_division_node(self, division_node: Node):
        divisor_node = division_node.args[1]

        # if the original `divisor` was a scalar,
        # it was not represented by a node in the graph
        if isinstance(divisor_node, Node):
            self.graph_module.graph.erase_node(divisor_node)

    def call(self, graph_module: GraphModule) -> PassResult:
        self.graph_module = graph_module
        made_changes = False

        for node in list(graph_module.graph.nodes):
            if not self._is_div_tensor(node):
                continue
            div_node = node

            # division can be rewritten as multiplication:
            #     quotient / divisor  →  quotient * (1 / divisor)
            # the term `(1 / divisor)` is referred to as the `multiplier`
            quotient_node = div_node.all_input_nodes[0]
            multiplier_node = self._create_mul_data_node(div_node)

            # cannot create `multiplier` because it is a non-static tensor
            if multiplier_node is None:
                continue

            with self.graph_module.graph.inserting_after(div_node):
                mul_node = self._create_mul_op_node(quotient_node, multiplier_node)

            div_node.replace_all_uses_with(mul_node)
            self.graph_module.graph.erase_node(div_node)
            self._erase_divisor_from_division_node(div_node)

            made_changes = True

        self.graph_module.recompile()
        self.graph_module.graph.eliminate_dead_code()

        return PassResult(graph_module, made_changes)
