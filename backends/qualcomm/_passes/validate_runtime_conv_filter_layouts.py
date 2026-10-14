# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

import torch
from executorch.backends.qualcomm.builders.node_visitor import dq_ops
from executorch.backends.qualcomm.builders.utils import (
    get_custom_op_output_axis_order,
    get_custom_op_output_producer,
    uses_compatible_axis_order,
)

from executorch.backends.qualcomm.utils.constants import (
    QCOM_CUSTOM_OP_INPUT_AXIS_ORDERS,
    QCOM_CUSTOM_OP_OUTPUT_AXIS_ORDERS,
    QCOM_FILTER_AXIS_ORDER_HWIO,
    QCOM_FILTER_AXIS_ORDER_HWOI,
    QCOM_QUANT_ATTRS,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class ValidateRuntimeConvFilterLayouts(ExportPass):
    """Validate custom-op physical layouts and runtime convolution filters.

    Model integrations annotate an op-package output with its physical axis
    order. This pass validates input sources and all annotated output consumers.
    The Edge graph keeps its logical PyTorch filter layout.
    """

    _CONV_TARGET = exir_ops.edge.aten.convolution.default
    _CONV_BACKWARD_TARGET = exir_ops.edge.aten.convolution_backward.default

    def _validate_filter_tensor(self, filter_node: torch.fx.Node) -> None:
        filter_tensor = filter_node.meta.get("val")
        filter_rank = None if filter_tensor is None else filter_tensor.dim()
        if filter_rank != 4:
            raise ValueError(
                f"Runtime Conv filter {filter_node.name} must be rank 4; "
                f"got {filter_rank}."
            )
        if not filter_tensor.is_floating_point():
            raise ValueError(
                f"Runtime Conv filter {filter_node.name} must be floating point; "
                f"got {filter_tensor.dtype}."
            )
        producer_node = get_custom_op_output_producer(filter_node)
        if (
            filter_node.meta.get(QCOM_QUANT_ATTRS) is not None
            or producer_node.meta.get(QCOM_QUANT_ATTRS) is not None
        ):
            raise ValueError(
                f"Runtime Conv filter {filter_node.name} must not be quantized."
            )

    def _validate_filter(
        self,
        node: torch.fx.Node,
        filter_node: torch.fx.Node,
        axis_order: tuple[int, ...],
        supported_axis_orders: tuple[tuple[int, ...], ...],
    ) -> None:
        self._validate_filter_tensor(filter_node)
        if axis_order not in supported_axis_orders:
            raise ValueError(
                f"Runtime Conv filter {filter_node.name} has output axis order "
                f"{axis_order}; {node.target} requires one of {supported_axis_orders}."
            )

    def _validate_output_consumer(
        self,
        node: torch.fx.Node,
        output_node: torch.fx.Node,
        user: torch.fx.Node,
        axis_order: tuple[int, ...],
    ) -> None:
        if user.target == self._CONV_TARGET and user.args[1] is output_node:
            if user.args[6]:
                raise ValueError(
                    "Runtime filters for transposed convolution are not supported."
                )
            self._validate_filter(
                user,
                output_node,
                axis_order,
                (QCOM_FILTER_AXIS_ORDER_HWIO,),
            )
        elif user.target == self._CONV_BACKWARD_TARGET and user.args[2] is output_node:
            self._validate_filter(
                user,
                output_node,
                axis_order,
                (QCOM_FILTER_AXIS_ORDER_HWIO, QCOM_FILTER_AXIS_ORDER_HWOI),
            )
        elif not uses_compatible_axis_order(user, output_node, axis_order):
            raise ValueError(
                f"Custom op {node.name} output {output_node.name} has axis order "
                f"{axis_order} incompatible with consumer {user.name}."
            )

    def _validate_input_sources(self, node: torch.fx.Node) -> None:
        for index, axis_order in node.meta[QCOM_CUSTOM_OP_INPUT_AXIS_ORDERS].items():
            if (
                index < 0
                or index >= len(node.args)
                or not isinstance(node.args[index], torch.fx.Node)
            ):
                raise ValueError(f"Custom op {node.name} has invalid input {index}.")
            input_node = node.args[index]
            if input_node.target in dq_ops:
                input_node = input_node.args[0]
            if input_node.op not in {"placeholder", "get_attr"} and (
                get_custom_op_output_axis_order(input_node) != axis_order
            ):
                raise ValueError(
                    f"Custom op {node.name} input {index} ({input_node.name}) with "
                    f"axis order {axis_order} requires a graph input, a parameter, "
                    "or a custom-op output with the same axis order."
                )

    def _validate_output_users(self, node: torch.fx.Node) -> None:
        output_axis_orders = node.meta[QCOM_CUSTOM_OP_OUTPUT_AXIS_ORDERS]
        outputs = node.meta.get("val")
        output_count = len(outputs) if isinstance(outputs, tuple) else 1
        for index in output_axis_orders:
            if index < 0 or index >= output_count:
                raise ValueError(
                    f"Custom op {node.name} has invalid output {index} axis order."
                )
        output_nodes = {}
        if isinstance(outputs, tuple):
            for user in node.users:
                if user.target is not operator.getitem:
                    raise ValueError(
                        f"Custom op {node.name} has a tuple output consumed directly by "
                        f"{user.name}; tuple outputs must be accessed through operator.getitem."
                    )
                if user.args[1] in output_axis_orders:
                    output_nodes.setdefault(user.args[1], []).append(user)
        else:
            output_nodes[0] = [node]

        for index, axis_order in output_axis_orders.items():
            for output_node in output_nodes.get(index, []):
                for user in output_node.users:
                    self._validate_output_consumer(node, output_node, user, axis_order)

    def call(self, graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            if QCOM_CUSTOM_OP_INPUT_AXIS_ORDERS in node.meta:
                self._validate_input_sources(node)
            if QCOM_CUSTOM_OP_OUTPUT_AXIS_ORDERS in node.meta:
                self._validate_output_users(node)

        return PassResult(graph_module, False)
