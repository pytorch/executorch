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
    """Numerous TOSA ops require inputs and outputs to be 32-bit integers in their
    quantized implementations. This pass treats such operator nodes by
    inserting rescale ops before and after them if needed. Note that extra
    logic that handles the scales and zero points are in place here because the
    affected TOSA ops have naive implementations that do not account for the
    quantization parameters.
    """

    # SUM must be decomposed after this pass to prevent insertion of RESCALE
    # nodes between each subsequent SUM node after decomposition. RESCALE nodes
    # should only be inserted before and after the SUM node prior to its
    # decomposition.
    _passes_required_after: Set[Type[ExportPass]] = {DecomposeSumPass}

    included_targets = [
        exir_ops.edge.aten.abs.default,
        exir_ops.edge.aten.add.Tensor,
        exir_ops.edge.aten.eq.Tensor,
        exir_ops.edge.aten.ge.Tensor,
        exir_ops.edge.aten.gt.Tensor,
        exir_ops.edge.aten.le.Tensor,
        exir_ops.edge.aten.lt.Tensor,
        exir_ops.edge.aten.maximum.default,
        exir_ops.edge.aten.minimum.default,
        exir_ops.edge.aten.mul.Tensor,
        exir_ops.edge.aten.sub.Tensor,
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
            qparams = {i: self._int32_qargs(min_scale) for i in input_qparams.keys()}
        elif target in [
            exir_ops.edge.aten.add.Tensor,
            exir_ops.edge.aten.sub.Tensor,
        ]:
            keys = list(input_qparams)
            if len(keys) < 2:
                raise ValueError(f"Expected two input qparams, got: {input_qparams}.")
            if input_qparams[keys[0]].dtype != input_qparams[keys[1]].dtype:
                raise ValueError(
                    f"Mismatch in dtype args: {input_qparams[keys[0]].dtype} != {input_qparams[keys[1]].dtype}"
                )

            # We are handling two INT8 or two INT16 numbers. For INT8, if the
            # zero point is non-null, the result will be in the range [-255;
            # 255], therefore we need 9 bits for the result. We have a 32-bit
            # accumulator, so we can divide the scale by (1 << 20) which is
            # equivalent to shifting the INT8 operands 20 bits to the left
            # before rescaling them both to 2 * max(lhs, rhs).
            #
            # For INT16, similary logic can be applied, but we instead end up
            # with a left shift of 12.
            lhs_scale, rhs_scale = (
                qp.get_scale_per_tensor() for qp in input_qparams.values()
            )
            max_scale_2x = 2 * max(lhs_scale, rhs_scale)

            # Select shift based on input dtype.
            shift_bits = 12 if input_qparams[keys[0]].dtype == torch.int16 else 20

            scale = max_scale_2x / (1 << shift_bits)
            qparams = {i: self._int32_qargs(scale) for i in input_qparams.keys()}
        elif target in [
            exir_ops.edge.aten.mul.Tensor,
            exir_ops.edge.aten.sum.dim_IntList,
        ]:
            # The input scales do not need to be adjusted for these ops; they
            # can remain the same.
            qparams = {
                i: self._int32_qargs(qp.get_scale_per_tensor())
                for i, qp in input_qparams.items()
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
            exir_ops.edge.aten.add.Tensor,
            exir_ops.edge.aten.sub.Tensor,
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


class InsertControlFlowRescalesPass(ArmPass):
    """The quantization parameters for tensors going into and coming out of a submodule are not guaranteed to
    match the quantization parameters for the corresponding tensors inside the submodule. For example, cond has
    different annotation on input and output, while the entire graph inside the submodule could be using shared
    annotation. This pass solves this by inserting rescales in the beginning and end of the submodule
    that transform the tensor from one set of quantization parameters to another.

    The pass is run by the graph_module containing the control flow operator, but requires that the affected nodes
    inside the submodule have been q-dq folded and have input/output_qparams meta.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def _get_input_nodes(self, graph_module: GraphModule):
        return [node for node in graph_module.graph.nodes if node.op == "placeholder"]

    def _insert_rescale(
        self,
        in_qparams: QuantArgs,
        out_qparams: QuantArgs,
        from_node: Node,
        graph_module: GraphModule,
    ):
        """Insert a rescale into the graph, inheriting meta from `from_node`.
        The node is not connected to anything, that is up to the user."""

        new_scales = [
            in_qparams.get_scale_per_tensor() / out_qparams.get_scale_per_tensor()
        ]

        rescale_node = create_node(
            graph_module.graph,
            exir_ops.backend.tosa.RESCALE.default,
            (
                None,
                out_qparams.dtype,
                new_scales,
                in_qparams.get_zp_per_tensor(),  # Old zero point
                out_qparams.get_zp_per_tensor(),  # New zero point
            ),
            from_node=from_node,
        )
        return rescale_node

    def _rescale_submodule_inputs(
        self, submodule: GraphModule, input_qparams_map: Dict[int, QuantArgs]
    ) -> bool:
        """Insert rescales at the inputs of `submodule` to match the qparams outside the submodule.
        Matching the correct qparams gets a bit tricky:
        Containing module: | submodule:
              ops => cond  | => placeholders => ...

        The dq->q qparam pair we want to convert to a rescale is:
        (input qparams of op, output qparams of placeholder)
        And the rescale is inserted after the placeholder.

        Args:
            submodule: GraphModule: the GraphModule in which to rescale the inputs.
            input_qparams_map: A map of input indexes mapping to QuantArgs. Not guaranteed to contain a mapping
                for every submodule input.
        Returns:
            True if at least one rescale was inserted, False otherwise.
        """

        modified = False
        input_nodes = self._get_input_nodes(submodule)
        for qargs_index in input_qparams_map:
            input_node = input_nodes[qargs_index]
            if len(input_node.users) == 0:
                continue
            if len(out_qparams_map := input_node.meta.get("output_qparams", {})) != 1:
                raise ValueError(
                    f"Expected submodule input {input_node} to have exactly one output qparam, got {out_qparams_map}"
                )
            in_qparams = input_qparams_map[qargs_index]
            out_qparams = cast(QuantArgs, out_qparams_map[0])

            # Remove qparam meta to not confuse folding pass.
            del input_node.meta["output_qparams"]
            if in_qparams == out_qparams:
                continue
            with submodule.graph.inserting_after(input_node):
                modified = True
                rescale_node = self._insert_rescale(
                    in_qparams, out_qparams, input_node, submodule
                )
                input_node.replace_all_uses_with(replace_with=rescale_node)
                rescale_node.update_arg(0, input_node)
        return modified

    def _rescale_submodule_outputs(
        self, submodule: GraphModule, output_qparams_map: Dict[int, QuantArgs]
    ) -> bool:
        """Insert rescales at the outputs of `submodule` to match the qparams outside the submodule.
        Matching the correct qparams gets a bit tricky:
        Submodule:             | Containing module:
        output_nodes => output |=> getitems => ...

        The dq->q qparam pair we want to convert to a rescale is:
        (input qparam of output_node, output qparam of getitem)
        And the rescale is inserted between op and output. Note that the output qparam of op is called input_qargs,
        since the it is the input to the dq-q pair.

        Args:
            submodule: GraphModule: the GraphModule in which to rescale the outputs.
            output_qparams_map: A map of output indexes mapping to QuantArgs. Not guaranteed to contain a mapping
                for every submodule output.
        Returns:
            True if at least one rescale was inserted, False otherwise.
        """

        modified = False
        output_node = submodule.graph.output_node()
        output_args = list(cast(tuple[Node], output_node.args[0]))
        input_qparams_map = cast(
            dict[int, QuantArgs], output_node.meta["input_qparams"]
        )
        for qargs_index in output_qparams_map:
            output_arg_node = output_args[qargs_index]
            in_qparams = input_qparams_map[qargs_index]
            out_qparams = output_qparams_map[qargs_index]
            if in_qparams == out_qparams:
                continue
            with submodule.graph.inserting_before(output_node):
                modified = True
                rescale_node = self._insert_rescale(
                    in_qparams, out_qparams, output_arg_node, submodule
                )
                output_args[qargs_index] = rescale_node
                rescale_node.update_arg(0, output_arg_node)
        output_node.update_arg(0, tuple(output_args))
        # Remove qparam meta to not confuse folding pass.
        del output_node.meta["input_qparams"]
        return modified

    def _get_input_qparams_map(self, node: Node, idx: int):
        input_qparams_meta = cast(
            dict[int, QuantArgs], node.meta.get("input_qparams", None)
        )
        if input_qparams_meta:
            input_qparams = cast(QuantArgs, input_qparams_meta.get(idx, None))
            if not input_qparams:
                raise ValueError(
                    f"Expected entry with key {idx} in input_qparams meta, got {input_qparams_meta}"
                )
            num_inputs = len(cast(list, node.args[idx]))

            # Currently, infra only supports one set of qparams for a list of inputs
            # Map all inputs to the same qparams.
            input_qparams_map = {i: input_qparams for i in range(num_inputs)}
            return input_qparams_map
        return None

    def _get_output_qparams_map(self, node: Node):
        output_qparams_map: dict[int, QuantArgs] = {}
        for getitem_node in node.users:
            idx = cast(int, getitem_node.args[1])
            qparam = getitem_node.meta.get("output_qparams", None)
            if qparam:
                output_qparams_map[idx] = cast(QuantArgs, qparam[0])
        return output_qparams_map

    def _rescale_cond_submodules(self, node: Node, graph_module: GraphModule) -> bool:
        modified = False
        if_graph: GraphModule = cast(GraphModule, graph_module.get_submodule(node.args[1].target))  # type: ignore
        else_graph: GraphModule = cast(GraphModule, graph_module.get_submodule(node.args[2].target))  # type: ignore
        input_qparams_map = self._get_input_qparams_map(node, 3)
        if input_qparams_map:
            modified |= self._rescale_submodule_inputs(if_graph, input_qparams_map)
            modified |= self._rescale_submodule_inputs(else_graph, input_qparams_map)

        output_qparams_map = self._get_output_qparams_map(node)
        if output_qparams_map:
            modified |= self._rescale_submodule_outputs(if_graph, output_qparams_map)
            modified |= self._rescale_submodule_outputs(else_graph, output_qparams_map)
        return modified

    def _rescale_while_submodules(self, node: Node, graph_module: GraphModule):
        modified = False
        cond_graph: GraphModule = cast(GraphModule, graph_module.get_submodule(node.args[0].target))  # type: ignore
        body_graph: GraphModule = cast(GraphModule, graph_module.get_submodule(node.args[1].target))  # type: ignore

        input_qparams_map = self._get_input_qparams_map(node, 2)
        if input_qparams_map:
            modified |= self._rescale_submodule_inputs(cond_graph, input_qparams_map)
            modified |= self._rescale_submodule_inputs(body_graph, input_qparams_map)

        output_qparams_map = self._get_output_qparams_map(node)
        if output_qparams_map:
            modified |= self._rescale_submodule_outputs(body_graph, output_qparams_map)
        return modified

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False

        for node in list(graph_module.graph.nodes):
            node = cast(Node, node)
            if node.op != "call_function":
                continue

            if node.target == torch.ops.higher_order.cond:
                modified = self._rescale_cond_submodules(node, graph_module)
            if node.target == torch.ops.higher_order.while_loop:
                modified = self._rescale_while_submodules(node, graph_module)

        if modified:
            # Retrace the graph to update the fake tensor types
            graph_module = super().call(graph_module).graph_module
            graph_module.recompile()

        return PassResult(graph_module, modified)
