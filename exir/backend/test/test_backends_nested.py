# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
import unittest
from typing import Dict, final, List

import executorch.exir as exir

import torch

from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.canonical_partitioners.pattern_op_partitioner import (
    generate_pattern_op_partitions,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)

from executorch.exir.backend.test.op_partitioner_demo import (
    AddOperatorSupport,
    MatmulOperatorSupport,
)
from executorch.exir.delegate import executorch_call_delegate

from executorch.exir.graph_module import _get_submodule, get_control_flow_submodules
from executorch.exir.lowered_backend_module import get_lowered_submodules
from functorch.experimental import control_flow
from torch.export import ExportedProgram
from torch.fx.passes.operator_support import any_chain, OperatorSupportBase


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, pred1, pred2, y):
        def true_fn(x, pred2):
            def true_nested(y):
                y = y + y
                y = torch.mm(y, y)
                return y

            def false_nested(y):
                return torch.mm(y, y)

            z = control_flow.cond(pred2, true_nested, false_nested, [x])
            return x + z

        def false_fn(x, _pred2):
            return torch.mm(x, x)

        x = x.cos()
        x = x + y
        y = control_flow.cond(pred1, true_fn, false_fn, [x, pred2])
        return y.sin()

    def get_example_inputs(self):
        return (
            torch.ones(2, 2),
            torch.tensor([False]),
            torch.Tensor([False]),
            torch.ones(2, 2),
        )


@final
class Backend2Demo(BackendDetails):
    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        processed_bytes = "Backend2::"
        for node in edge_program.graph.nodes:
            if node.op == "call_function":
                processed_bytes += f"{node.target.__name__};"
        return PreprocessResult(
            processed_bytes=bytes(processed_bytes, encoding="utf8"),
        )


@final
class Backend2PartitionerDemo(Partitioner):
    """
    Partitions all add/mul nodes regardless of order for Backend2
    """

    def __init__(self) -> None:
        self.op_support = any_chain(AddOperatorSupport(), MatmulOperatorSupport())
        self.delegation_spec = DelegationSpec("Backend2Demo", [])
        self.partition_tags = {}

    def _partition_graph_module(
        self, edge_graph_module: torch.fx.GraphModule
    ) -> Dict[str, DelegationSpec]:
        partition_tags: Dict[str, DelegationSpec] = {}
        partition_list = generate_pattern_op_partitions(
            edge_graph_module, op_support=self.op_support
        )

        for _, submodule, _ in get_control_flow_submodules(edge_graph_module):
            submodule_partition_tags = self._partition_graph_module(submodule)
            partition_tags.update(submodule_partition_tags)

        for partition in partition_list:
            for node in partition.nodes:
                delegation_tag = f"backend2_tag{partition.id}"
                node.meta["delegation_tag"] = delegation_tag
                partition_tags[delegation_tag] = self.delegation_spec
        return partition_tags

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        partition_tags = self._partition_graph_module(exported_program.graph_module)
        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )


@final
class Backend1Demo(BackendDetails):
    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        assert isinstance(edge_program, ExportedProgram)
        partitioned_module = to_backend(edge_program, Backend2PartitionerDemo())

        def process(gm):
            processed_bytes = ""
            for node in gm.graph.nodes:
                if node.op == "call_function":
                    if node.target is torch.ops.higher_order.cond:
                        _, true_gm, _ = _get_submodule(gm, node, 1)
                        _, false_gm, _ = _get_submodule(gm, node, 2)
                        processed_bytes += f"{node.target.__name__}({process(true_gm)},{process(false_gm)});"
                    elif node.target is operator.getitem:
                        continue
                    elif node.target is executorch_call_delegate:
                        _, lowered, _ = _get_submodule(gm, node, 0)
                        processed_bytes += f"call_delegate({lowered.processed_bytes});"
                    else:
                        processed_bytes += f"{node.target.__name__};"
            return processed_bytes

        processed_bytes = f"Backend1::({process(partitioned_module.graph_module)})"
        return PreprocessResult(
            processed_bytes=bytes(processed_bytes, encoding="utf8"),
        )


class CondOperatorSupport(OperatorSupportBase):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        return node.op == "call_function" and node.target is torch.ops.higher_order.cond


@final
class Backend1PartitionerDemo(Partitioner):
    """
    Partitions all add/mul/cond nodes regardless of order. Since we're
    partitioning the cond ops, we do not need to go into those submodules.
    """

    def __init__(self) -> None:
        self.op_support = any_chain(
            AddOperatorSupport(), MatmulOperatorSupport(), CondOperatorSupport()
        )
        self.delegation_spec = DelegationSpec("Backend1Demo", [])

    def _partition_graph_module(
        self, edge_graph_module: torch.fx.GraphModule
    ) -> Dict[str, DelegationSpec]:
        partition_tags: Dict[str, DelegationSpec] = {}
        partition_list = generate_pattern_op_partitions(
            edge_graph_module, op_support=self.op_support
        )

        for _, submodule, node in get_control_flow_submodules(edge_graph_module):
            # Don't partition the cond submodules because we are lowering the
            # entire cond node, including it's submodules.
            if node.target is not control_flow.cond:
                self._partition_graph_module(submodule)

        for partition in partition_list:
            for node in partition.nodes:
                delegation_tag = f"backend1_tag{partition.id}"
                if (
                    node.op == "call_function"
                    and node.target is torch.ops.higher_order.cond
                ):
                    # Tag the arguments that take in the submodules to cond
                    node.args[1].meta["delegation_tag"] = delegation_tag
                    node.args[2].meta["delegation_tag"] = delegation_tag
                node.meta["delegation_tag"] = delegation_tag
                partition_tags[delegation_tag] = self.delegation_spec
        return partition_tags

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        partition_tags = self._partition_graph_module(exported_program.graph_module)
        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )


class TestNestedBackends(unittest.TestCase):
    def test(self) -> None:
        """
        Partitions the cond ops into the delegate
        """

        m = M()
        orig_res = m(*m.get_example_inputs())
        orig = exir.capture(
            m,
            m.get_example_inputs(),
            exir.CaptureConfig(),
        ).to_edge(exir.EdgeCompileConfig(_check_ir_validity=False))

        partitioned = orig
        partitioned.exported_program = to_backend(
            orig.exported_program, Backend1PartitionerDemo()
        )

        new_res = partitioned(*m.get_example_inputs())[0]
        self.assertTrue(torch.allclose(orig_res, new_res))

        # The toplevel module should have lowered the cond and add op
        toplevel_lowered = get_lowered_submodules(
            partitioned.exported_program.graph_module
        )
        self.assertEqual(len(toplevel_lowered), 1)
        toplevel_lowered = toplevel_lowered[0][1]
        self.maxDiff = None
        self.assertEqual(
            str(toplevel_lowered.processed_bytes),
            (
                'b"Backend1::('
                + "call_delegate(b'Backend2::aten.add.Tensor;');"
                + "cond("
                # True function of toplevel cond (nested cond)
                + "cond(call_delegate(b'Backend2::aten.add.Tensor;aten.mm.default;');,call_delegate(b'Backend2::aten.mm.default;'););"
                # True function of toplevel cond (delegated add)
                + "call_delegate(b'Backend2::aten.add.Tensor;');,"
                # False function of toplevel cond
                + "call_delegate(b'Backend2::aten.mm.default;'););)\""
            ),
        )
