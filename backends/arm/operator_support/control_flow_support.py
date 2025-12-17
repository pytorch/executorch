# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import typing
from typing import cast

import torch
import torch.fx as fx

from executorch.backends.arm._passes.arm_pass_utils import is_submodule_node
from executorch.backends.arm.constants import DQ_OPS, Q_OPS
from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.arm.tosa.specification import Tosa_1_00
from executorch.exir import ExportedProgram
from executorch.exir.backend.utils import WhyNoPartitionReporter

from torch.fx.passes.operator_support import OperatorSupportBase


def _fully_partitioned(submodule: fx.GraphModule) -> bool:
    partition_tag = None
    for submodule_node in submodule.graph.nodes:
        if submodule_node.op == "call_function":
            # Input Q ops and output DQ ops will be de-tagged even if the submodule is fully supported.
            if (
                submodule_node.target in Q_OPS
                and list(submodule_node.all_input_nodes)[0].op == "placeholder"
            ):
                continue
            if (
                submodule_node.target in DQ_OPS
                and list(submodule_node.users)[0].op == "output"
            ):
                continue
            if "delegation_tag" not in submodule_node.meta:
                return False
            if partition_tag is None:
                partition_tag = submodule_node.meta["delegation_tag"]
            elif submodule_node.meta["delegation_tag"] != partition_tag:
                return False
    return True


def _submodules_fully_partitioned(
    node: fx.Node, exported_program: ExportedProgram
) -> bool:
    """Returns whether the submodule arguments to a cond node were fully partitioned.
    Updates "val" meta of the submodules if they are.
    """
    match node.target:
        case torch.ops.higher_order.cond:
            submodule_args = node.args[1:3]
        case torch.ops.higher_order.while_loop:
            submodule_args = node.args[0:2]
        case _:
            raise ValueError(f"Unexpected target: {node.target}")
    cond_submodules = (
        (
            exported_program.graph_module.get_submodule(
                str(cast(torch.fx.Node, submodule_node).target)
            ),
            cast(torch.fx.Node, submodule_node),
        )
        for submodule_node in submodule_args
    )
    for submodule, submodule_node in cond_submodules:
        submodule = cast(torch.fx.GraphModule, submodule)

        if _fully_partitioned(submodule):
            submodule_node.meta["val"] = submodule.graph.output_node().meta["val"]
        else:
            return False
    return True


def _tosa_spec_supports_cf(tosa_spec: TosaSpecification) -> bool:
    if not isinstance(tosa_spec, Tosa_1_00):
        return False
    return tosa_spec.support_extension("cf")


class ControlFlowSubmoduleSupported(OperatorSupportBase):
    """Check whether control flow submodule args should be partitioned.
    Applies control-flow extension constraints before allowing delegation."""

    def __init__(
        self,
        exported_program: ExportedProgram,
        tosa_spec: TosaSpecification,
        reporter: WhyNoPartitionReporter,
    ):
        self.exported_program = exported_program
        self.reporter = reporter
        self.tosa_spec = tosa_spec
        super().__init__()

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        if is_submodule_node(node):
            if not _tosa_spec_supports_cf(self.tosa_spec):
                self.reporter.report_reject(
                    node,
                    f"TOSA spec {self.tosa_spec} does not support control flow extension.",
                )
                return False
            for user in node.users:
                if user.target not in ControlFlowOpSupported._targeted_ops:
                    self.reporter.report_reject(
                        node, f"Submodule had unsupported user {user}"
                    )
                    return False
                if not _submodules_fully_partitioned(user, self.exported_program):
                    self.reporter.report_reject(
                        node, "One submodule was not fully partitioned"
                    )
                    return False
            return True
        return False


class ControlFlowOpSupported(OperatorSupportBase):
    """Check whether control flow ops should be partitioned.
    Applies control-flow extension constraints before allowing delegation."""

    _targeted_ops = {
        torch.ops.higher_order.cond,
        torch.ops.higher_order.while_loop,
    }

    def __init__(
        self,
        exported_program: ExportedProgram,
        tosa_spec: TosaSpecification,
        reporter: WhyNoPartitionReporter,
    ):
        self.exported_program = exported_program
        self.reporter = reporter
        self.tosa_spec = tosa_spec
        super().__init__()

    def is_node_supported(
        self, submodules: typing.Mapping[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        if node.target in self._targeted_ops:
            if not _tosa_spec_supports_cf(self.tosa_spec):
                self.reporter.report_reject(
                    node,
                    f"TOSA spec {self.tosa_spec} does not support control flow extension.",
                )
                return False

            if not _submodules_fully_partitioned(node, self.exported_program):
                self.reporter.report_reject(
                    node, "Submodule was not fully partitioned."
                )
                return False
            return True

        return False
