# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm._passes import ArmOpTargetedPass
from executorch.backends.arm._passes.arm_pass_utils import (
    get_strictly_positive_lower_bound,
    is_strictly_positive_tensor_node,
    POW_LOG_POSITIVE_LOWER_BOUND_META,
)
from executorch.backends.arm.tosa.specification import TosaSpecification
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node


aten_pow_tensor_tensor_ops = (torch.ops.aten.pow.Tensor_Tensor,)
edge_pow_tensor_tensor_ops = (exir_ops.edge.aten.pow.Tensor_Tensor,)


class DecomposePowTensorTensorPass(ArmOpTargetedPass):
    """Decompose Tensor/Tensor pow for INT-only TOSA lowering.

    For a strictly positive base:

        pow(x, y) == exp(y * log(x))

    The decomposition runs during transform-for-annotation so LOG, MUL and EXP
    can receive independent observers/quantization parameters. It intentionally
    does not run for FP or mixed INT+FP profiles, where native TOSA POW can be
    used.

    Positivity is proven on the *completed input FX graph before retracing*.
    This is important because prior TFA passes can materialize scalar constants
    as get_attr nodes. During ExportPass retracing the new graph is incomplete
    and does not yet have an owning GraphModule, so resolving those constants in
    call_operator() is unreliable.

    If strict positivity cannot be proven structurally, POW is left unchanged.
    Calibration/example values are never used as a runtime positivity proof.

    """

    _passes_required_after: set[type[ExportPass]] = set()

    target_ops = (
        *aten_pow_tensor_tensor_ops,
        *edge_pow_tensor_tensor_ops,
    )

    check_allowed_to_transform = True

    def __init__(
        self,
        tosa_spec: TosaSpecification,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tosa_spec = tosa_spec

    def should_run_pass(self, graph_module: GraphModule) -> bool:
        # Native TOSA POW is floating-point. Only decompose for a pure integer
        # profile such as the Ethos-U55 INT path.
        return (
            self.tosa_spec.support_integer()
            and not self.tosa_spec.support_float()
            and super().should_run_pass(graph_module)
        )

    @staticmethod
    def _decomposition_ops(op):
        if op in aten_pow_tensor_tensor_ops:
            return (
                torch.ops.aten.log.default,
                torch.ops.aten.mul.Tensor,
                torch.ops.aten.exp.default,
            )
        if op in edge_pow_tensor_tensor_ops:
            return (
                exir_ops.edge.aten.log.default,
                exir_ops.edge.aten.mul.Tensor,
                exir_ops.edge.aten.exp.default,
            )
        raise AssertionError(f"Unexpected pow op: {op}")

    @staticmethod
    def _set_log_meta(log_node: Node, pow_node: Node, base: Node) -> None:
        """Copy POW metadata while keeping LOG metadata base-shaped."""
        log_node.meta = dict(pow_node.meta)

        # Tensor/Tensor POW broadcasts, while LOG is shape-preserving on base.
        # Never leave POW output shape metadata attached to LOG.
        for key in ("val", "tensor_meta", "example_value"):
            if key in base.meta:
                log_node.meta[key] = base.meta[key]
            else:
                log_node.meta.pop(key, None)

    def _rewrite_graph(self, graph_module: GraphModule) -> bool:
        graph = graph_module.graph
        modified = False

        for node in list(graph.nodes):
            if (
                node.op != "call_function"
                or node.target not in self.target_ops
                or not self.allowed_to_transform(node.meta)
                or len(node.args) < 2
            ):
                continue

            base = node.args[0]
            exponent = node.args[1]
            if not isinstance(base, Node) or not is_strictly_positive_tensor_node(base):
                continue

            # The source graph proves base > 0, but the eventual LOG
            # input qparams are not known yet. Carry a concrete lower
            # bound forward so INT lowering can validate QDQ(base) > 0.
            positive_lower_bound = get_strictly_positive_lower_bound(base)
            if positive_lower_bound is None:
                continue

            log_op, mul_op, exp_op = self._decomposition_ops(node.target)

            with graph.inserting_before(node):
                log_base = graph.create_node(
                    "call_function",
                    log_op,
                    (base,),
                    {},
                )
                self._set_log_meta(log_base, node, base)
                custom_meta = dict(log_base.meta.get("custom", {}))
                custom_meta[POW_LOG_POSITIVE_LOWER_BOUND_META] = positive_lower_bound
                log_base.meta["custom"] = custom_meta
                log_base.meta[POW_LOG_POSITIVE_LOWER_BOUND_META] = positive_lower_bound

                scaled_log = graph.create_node(
                    "call_function",
                    mul_op,
                    (log_base, exponent),
                    {},
                )
                scaled_log.meta = dict(node.meta)

                exp_result = graph.create_node(
                    "call_function",
                    exp_op,
                    (scaled_log,),
                    {},
                )
                exp_result.meta = dict(node.meta)

            node.replace_all_uses_with(exp_result)
            graph.erase_node(node)
            modified = True

        if modified:
            graph.eliminate_dead_code()
            graph.lint()
            graph_module.recompile()

        return modified

    def _rewrite_graph_module_tree(self, graph_module: GraphModule) -> bool:
        modified = self._rewrite_graph(graph_module)
        for child in graph_module.children():
            if isinstance(child, GraphModule):
                modified |= self._rewrite_graph_module_tree(child)
        return modified

    def call(self, graph_module: GraphModule) -> PassResult:
        # Perform the graph-dependent proof/rewrite before ExportPass starts
        # rebuilding the graph. At this point get_attr nodes still resolve through
        # graph.owning_module and compile-time scalar constants are observable.
        if not self._rewrite_graph_module_tree(graph_module):
            return PassResult(graph_module, False)

        # Retrace only after the semantic decision has been made. This refreshes
        # FakeTensor metadata for LOG/MUL/EXP without attempting to rediscover
        # constants from a partially constructed tracer graph.
        return super().call(graph_module)
