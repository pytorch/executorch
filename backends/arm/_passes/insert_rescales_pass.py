# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from copy import copy
from typing import cast, Dict, Optional, Set, Tuple, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import create_node, set_node_arg
from executorch.backends.arm._passes.decompose_sum_pass import DecomposeSumPass
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_output_qparams,
)

from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.backends.arm.constants import DQ_OPS, Q_OPS
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node


class InsertRescalePass(ArmPass):
    """Finds patterns of dq -> q, and replaces them
    with backend dialect tosa::RESCALE op.

    Does not guarantee that the dtypes and zero points are valid
    in TOSA, that is the job of the quantization annotator that
    produced the dq and q nodes. The TOSA constraints are validated
    in the fake implementation of.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def fold_dq_q_to_rescale(self, node: Node, user: Node, graph_module: GraphModule):
        dq_args = QuantArgs.from_operator(node.target, node.args)
        q_args = QuantArgs.from_operator(user.target, user.args)
        new_scale = dq_args.scale / q_args.scale

        with graph_module.graph.inserting_before(node):
            rescale_node = create_node(
                graph_module.graph,
                exir_ops.backend.tosa.RESCALE.default,
                (
                    node.all_input_nodes[0],
                    q_args.dtype,
                    [new_scale],
                    dq_args.zp,
                    q_args.zp,
                ),
            )
            rescale_node.meta = copy(user.meta)
            user.replace_all_uses_with(rescale_node)
            graph_module.graph.erase_node(user)

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        for node in graph_module.graph.nodes:
            node = cast(Node, node)

            if node.target not in DQ_OPS:
                continue
            # Copy users since we remove them while iterating, modyfing the node.users list.
            for user in copy(node.users):
                if user.target in Q_OPS:
                    self.fold_dq_q_to_rescale(node, user, graph_module)
                    modified = True
            if len(node.users) == 0:
                graph_module.graph.erase_node(node)

        graph_module = super().call(graph_module).graph_module
        graph_module.recompile()
        return PassResult(graph_module, modified)


class InsertRescaleInt32Pass(ArmPass):
    """
    Numerous TOSA ops require inputs and outputs to be 32-bit integers in their
    quantized implementations. This pass treats such operator nodes by
    inserting rescale ops before and after them if needed. Note that extra logic
    that handles the scales and zero points must be in place because the affected
    TOSA have naive implementations that do not account for the quantization
    parameters.
    """

    # SUM must be decomposed after this pass to prevent insertion of RESCALE
    # nodes between each subsequent SUM node after decomposition. RESCALE nodes
    # should only be inserted before and after the SUM node prior to its
    # decomposition.
    _passes_required_after: Set[Type[ExportPass]] = {DecomposeSumPass}

    included_targets = [
        exir_ops.edge.aten.abs.default,
        exir_ops.edge.aten.eq.Tensor,
        exir_ops.edge.aten.ge.Tensor,
        exir_ops.edge.aten.gt.Tensor,
        exir_ops.edge.aten.le.Tensor,
        exir_ops.edge.aten.lt.Tensor,
        exir_ops.edge.aten.maximum.default,
        exir_ops.edge.aten.minimum.default,
        exir_ops.edge.aten.mul.Tensor,
        exir_ops.edge.aten.sum.dim_IntList,
    ]

    def _int32_qargs(self, s):
        """Helper creator function for INT32-based QuantArgs"""

        return QuantArgs(
            scale=s,
            zp=0,
            qmin=torch.iinfo(torch.int32).min,
            qmax=torch.iinfo(torch.int32).max,
            dtype=torch.int32,
        )

    def _get_inputs_rescaled_qparams(
        self, target, input_qparams: Dict[int, QuantArgs]
    ) -> Dict[int, QuantArgs]:
        """Get the qparams for the INT32 operands to the op ``target``

        Inputs to the INT32-based operator must be rescaled from INT8 to INT32.
        This function computes the ``QuantArgs`` for each of the operands and returns
        it as a dict, mapping tensor index to ``QuantArgs``.
        """

        if target in [
            exir_ops.edge.aten.abs.default,
            exir_ops.edge.aten.eq.Tensor,
            exir_ops.edge.aten.ge.Tensor,
            exir_ops.edge.aten.gt.Tensor,
            exir_ops.edge.aten.le.Tensor,
            exir_ops.edge.aten.lt.Tensor,
            exir_ops.edge.aten.minimum.default,
            exir_ops.edge.aten.maximum.default,
        ]:
            # For these ops, use the smallest scale among the INT8 operands.
            min_scale = min(
                [qp.get_scale_per_tensor() for qp in input_qparams.values()]
            )
            qparams = {
                i: self._int32_qargs(min_scale) for i in range(len(input_qparams))
            }
        elif target in [
            exir_ops.edge.aten.mul.Tensor,
            exir_ops.edge.aten.sum.dim_IntList,
        ]:
            # The input scales do not need to be adjusted for these ops; they
            # can remain the same.
            qparams = {
                i: self._int32_qargs(input_qparams[i].get_scale_per_tensor())
                for i in range(len(input_qparams))
            }
        else:
            raise ValueError(f"Not a valid target: {target}")

        return qparams

    def _get_output_qparams(
        self, target, inputs_qparams: Dict[int, QuantArgs]
    ) -> Optional[QuantArgs]:
        """Given an op ``target`` and the ``QuantArgs`` for each of its inputs, compute
        the scale of the output based on how the operator itself affects it."""

        if target in [
            exir_ops.edge.aten.abs.default,
            exir_ops.edge.aten.maximum.default,
            exir_ops.edge.aten.minimum.default,
            exir_ops.edge.aten.sum.dim_IntList,
        ]:
            # The op has not altered the scale; the output scale is equal to
            # the operands' scales.
            return self._int32_qargs(inputs_qparams[0].get_scale_per_tensor())
        elif target in [
            exir_ops.edge.aten.eq.Tensor,
            exir_ops.edge.aten.ge.Tensor,
            exir_ops.edge.aten.gt.Tensor,
            exir_ops.edge.aten.le.Tensor,
            exir_ops.edge.aten.lt.Tensor,
        ]:
            # Output is bool for these ops and thus no qparams are present
            return None
        elif target in [exir_ops.edge.aten.mul.Tensor]:
            # Mul will cause the scales to also multiply; refer to the formula
            # where we compute the output scale S_2:
            #
            # (Q_2 - ZP_2) * S_2 == ((Q_0 - ZP_0) * S_0) * ((Q_1 - ZP_1) * S_1)
            #
            # yields:
            #
            # (Q_2 - ZP_2) == (Q_0 - ZP_0) * (Q_1 - ZP_1)
            # S_2 = S_0 * S_1
            output_scale = math.prod(
                (qp.get_scale_per_tensor() for qp in inputs_qparams.values())
            )
            return self._int32_qargs(output_scale)
        else:
            raise ValueError(f"Not a valid target: {target}")

    def _get_rescale_qparams(
        self, target, input_qparams: Dict[int, QuantArgs]
    ) -> Tuple[Dict[int, QuantArgs], Optional[QuantArgs]]:
        """
        Get the quantization parameters of the INT32 inputs/outputs that will
        surround the node after the new RESCALE ops have been inserted.
        """

        inputs_rescaled_qparams = self._get_inputs_rescaled_qparams(
            target, input_qparams
        )
        output_qparams = self._get_output_qparams(target, inputs_rescaled_qparams)

        return (inputs_rescaled_qparams, output_qparams)

    def _rescale_inputs(self, graph, node, rescale_qargs: Dict[int, QuantArgs]) -> bool:
        qargs = node.meta["input_qparams"]

        args_copy = list(node.args)
        seen_args = set()
        modified = False
        for i in qargs:
            qp = qargs[i]
            if qp.dtype not in (torch.int8, torch.int16):
                continue

            arg_node = args_copy[i]
            if arg_node in seen_args:
                continue
            seen_args.add(arg_node)

            with graph.inserting_after(arg_node):
                rescale_node = create_node(
                    graph,
                    exir_ops.backend.tosa.RESCALE.default,
                    (
                        arg_node,
                        torch.int32,
                        [
                            qp.get_scale_per_tensor()
                            / rescale_qargs[i].get_scale_per_tensor()
                        ],  # [Old scale / new scale]
                        qp.get_zp_per_tensor(),  # Old zero point
                        rescale_qargs[i].get_zp_per_tensor(),  # New zero point
                    ),
                    from_node=node,
                )

                node.replace_input_with(arg_node, rescale_node)
                modified = True

        return modified

    def _rescale_outputs(self, graph, node, rescale_qargs: Optional[QuantArgs]) -> bool:
        if "output_qparams" not in node.meta or len(node.meta["output_qparams"]) == 0:
            return False

        qargs = get_output_qparams(node)
        assert len(qargs) == 1
        assert rescale_qargs is not None

        qarg = qargs[0]
        if qarg.dtype not in (torch.int8, torch.int16):
            return False

        users_copy = list(node.users)

        with graph.inserting_after(node):
            rescale_node = create_node(
                graph,
                exir_ops.backend.tosa.RESCALE.default,
                (
                    node,
                    qarg.dtype,
                    [
                        rescale_qargs.get_scale_per_tensor()
                        / qarg.get_scale_per_tensor()
                    ],  # [Old scale / new scale]
                    rescale_qargs.get_zp_per_tensor(),  # Old zero point
                    qarg.get_zp_per_tensor(),  # New zero point
                ),
                from_node=node,
            )

        for user in users_copy:
            user.replace_input_with(node, rescale_node)

        return True

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph

        modified = False
        for node in list(graph.nodes):
            node = cast(Node, node)

            if node.op != "call_function" or node.target not in self.included_targets:
                continue

            if "input_qparams" not in node.meta or len(node.meta["input_qparams"]) == 0:
                continue
            input_qparams = node.meta["input_qparams"]

            inputs_rescale_qargs, output_rescale_qargs = self._get_rescale_qparams(
                node.target, input_qparams
            )

            inputs_was_rescaled = self._rescale_inputs(
                graph, node, inputs_rescale_qargs
            )
            outputs_was_rescaled = False
            if inputs_was_rescaled:
                outputs_was_rescaled = self._rescale_outputs(
                    graph, node, output_rescale_qargs
                )
                modified = True

            # Update node metadata

            if inputs_was_rescaled:
                assert len(inputs_rescale_qargs) == len(node.meta["input_qparams"])
                node.meta["input_qparams"] = inputs_rescale_qargs

            if outputs_was_rescaled:
                assert len(node.meta["output_qparams"]) == 1
                node.meta["output_qparams"] = {0: output_rescale_qargs}

                # If the output type is specified in the node, change it such
                # that it matches the subsequent rescale node(s) that this node
                # now has output edges to.
                if "dtype" in node.kwargs:
                    set_node_arg(node, "dtype", torch.int32)

        if modified:
            # Retrace the graph to update the fake tensor types
            graph_module = super().call(graph_module).graph_module
            graph_module.recompile()

        return PassResult(graph_module, modified)
