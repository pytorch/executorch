# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import MappingProxyType

import torch

from executorch import exir
from executorch.exir.backend.backend_details import CompileSpec, ExportedProgram
from executorch.exir.backend.canonical_partitioners.pattern_op_partitioner import (
    generate_pattern_op_partitions,
)

from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.test.demos.rpc.executor_backend_partitioner import (
    AnyOperatorSupport,
)
from executorch.exir.backend.test.demos.rpc.executor_backend_preprocess import (
    ExecutorBackend,
)
from executorch.exir.backend.utils import get_delegates

from executorch.exir.dialects._ops import ops as exir_ops

from executorch.exir.tests.models import MLP
from torch._export import capture_pre_autograd_graph
from torch._export.utils import is_buffer, is_param
from torch.export import export
from torch.fx.passes.operator_support import any_chain


class TestPartitioner(unittest.TestCase):
    def test_partitioner_with_spec(self):
        # Create a custom partitioner with spec and check the spec can be accessed by not mutable.
        class PartitionerWithSpec(Partitioner):
            def __init__(self, spec) -> None:
                super().__init__(spec)
                self.op_support = any_chain(AnyOperatorSupport())
                self.delegation_spec = DelegationSpec(
                    ExecutorBackend.__name__,
                    [CompileSpec(key, value) for key, value in self.spec.items()],
                )

            def partition(
                self, edge_exported_program: ExportedProgram
            ) -> PartitionResult:
                partition_tags = {}
                partition_list = generate_pattern_op_partitions(
                    edge_exported_program.graph_module, op_support=self.op_support
                )
                for partition in partition_list:
                    for node in partition.nodes:
                        delegation_tag = f"tag{partition.id}"
                        node.meta["delegation_tag"] = delegation_tag
                        partition_tags[delegation_tag] = self.delegation_spec

                return PartitionResult(
                    tagged_exported_program=edge_exported_program,
                    partition_tags=partition_tags,
                )

        mlp = MLP()
        example_inputs = mlp.get_random_inputs()
        model = capture_pre_autograd_graph(mlp, example_inputs)
        aten = export(model, example_inputs)
        spec_key = "path"
        spec_value = "/a/b/c/d"
        spec = MappingProxyType({spec_key: spec_value})
        my_partitioner = PartitionerWithSpec(spec)
        edge = exir.to_edge(aten).to_backend(my_partitioner)

        lowered_module_nodes = get_delegates(edge.exported_program().graph)

        self.assertEqual(len(lowered_module_nodes), 1)
        # Check the lowered module has correct compile spec
        for lower_module_node in lowered_module_nodes:
            lower_module = getattr(
                edge.exported_program().graph_module, lower_module_node.name
            )
            self.assertEqual(lower_module.compile_specs[0].key, spec_key)
            self.assertEqual(lower_module.compile_specs[0].value, spec_value)

        # Check the custom partitioner has the correct spec
        self.assertEqual(my_partitioner.spec[spec_key], spec_value)

        with self.assertRaisesRegex(
            TypeError,
            "'mappingproxy' object does not support item assignment",
        ):
            my_partitioner.spec[spec_key] = "new_value"

        with self.assertRaisesRegex(
            AttributeError,
            "can't set attribute 'spec'",
        ):
            my_partitioner.spec = {"new_key": "new_value"}

    def test_bad_partitioner_tagged_output(self):
        # Create a bad partitioner to tag output, which is not allowed.
        class PartitionerTagOutput(Partitioner):
            def __init__(self) -> None:
                super().__init__()
                self.delegation_spec = DelegationSpec(
                    ExecutorBackend.__name__,
                    [CompileSpec(key, value) for key, value in self.spec.items()],
                )

            def partition(
                self, edge_exported_program: ExportedProgram
            ) -> PartitionResult:
                partition_tags = {}
                for node in edge_exported_program.graph.nodes:
                    if node.op == "output":
                        delegation_tag = "tag0"
                        node.meta["delegation_tag"] = delegation_tag
                        partition_tags[delegation_tag] = self.delegation_spec

                return PartitionResult(
                    tagged_exported_program=edge_exported_program,
                    partition_tags=partition_tags,
                )

        mlp = MLP()
        example_inputs = mlp.get_random_inputs()
        model = capture_pre_autograd_graph(mlp, example_inputs)
        aten = export(model, example_inputs)
        edge = exir.to_edge(aten)

        with self.assertRaisesRegex(
            RuntimeError,
            "output node output should not be tagged",
        ):
            _ = edge.to_backend(PartitionerTagOutput())

    def test_bad_partitioner_tagged_model_input(self):
        # Create a bad partitioner to tag an input that is neither params nor buffer, which is not allowed.
        class PartitionerTagInput(Partitioner):
            def __init__(self) -> None:
                super().__init__()
                self.delegation_spec = DelegationSpec(
                    ExecutorBackend.__name__,
                    [CompileSpec(key, value) for key, value in self.spec.items()],
                )

            def partition(
                self, edge_exported_program: ExportedProgram
            ) -> PartitionResult:
                partition_tags = {}
                for node in edge_exported_program.graph.nodes:
                    if node.op == "placeholder":
                        if not is_param(edge_exported_program, node) and not is_buffer(
                            edge_exported_program, node
                        ):
                            delegation_tag = "tag_" + str(node.meta["debug_handle"])
                            node.meta["delegation_tag"] = delegation_tag
                            partition_tags[delegation_tag] = self.delegation_spec

                return PartitionResult(
                    tagged_exported_program=edge_exported_program,
                    partition_tags=partition_tags,
                )

        mlp = MLP()
        example_inputs = mlp.get_random_inputs()
        model = capture_pre_autograd_graph(mlp, example_inputs)
        edge = exir.to_edge(export(model, example_inputs))

        with self.assertRaisesRegex(
            RuntimeError,
            "placeholder node for non-params, non-buffer, and non-tensor constants should not be tagged",
        ):
            _ = edge.to_backend(PartitionerTagInput())

    class AddConst(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.const1 = torch.ones(2, 2)
            self.register_buffer("const2", torch.ones(2, 2), persistent=False)
            self.register_parameter("const3", torch.nn.Parameter(torch.ones(2, 2)))

        def forward(self, x):
            return x + self.const1 + self.const2 + self.const3

    def test_partitioner_not_tag_data(self):
        """
        We test here that when partitioners do not explicitly tag constant data nodes,
        then the partitioned ExportedProgram will not own the data. Instead the owning program
        will still own the constant data and instead feed it as inputs to the partitioned
        program
        """

        class PartitionerNoTagData(Partitioner):
            def __init__(self):
                super().__init__()
                self.delegation_spec = DelegationSpec(
                    ExecutorBackend.__name__,
                    [CompileSpec(key, value) for key, value in self.spec.items()],
                )

            def partition(
                self, edge_exported_program: ExportedProgram
            ) -> PartitionResult:
                partition_tags = {}
                for node in edge_exported_program.graph.nodes:
                    if node.op == "call_function" and node.target in [
                        exir_ops.edge.aten.add.Tensor
                    ]:
                        delegation_tag = "tag0"
                        node.meta["delegation_tag"] = delegation_tag
                        partition_tags[delegation_tag] = self.delegation_spec

                return PartitionResult(
                    tagged_exported_program=edge_exported_program,
                    partition_tags=partition_tags,
                )

        model = capture_pre_autograd_graph(self.AddConst(), (torch.ones(2, 2),))
        edge = exir.to_edge(export(model, (torch.ones(2, 2),)))
        delegated = edge.to_backend(PartitionerNoTagData())

        # Check Owning Program still owns all constant data
        owning_program = delegated.exported_program()
        self.assertEqual(len(owning_program.state_dict), 3)
        self.assertEqual(len(owning_program.graph_signature.buffers), 2)
        self.assertEqual(len(owning_program.graph_signature.parameters), 1)

        # Check Lowered Module Exported Program does not have any constant data
        lowered_module_nodes = get_delegates(delegated.exported_program().graph)
        self.assertEqual(len(lowered_module_nodes), 1)
        lowered_module_node = lowered_module_nodes[0]

        # get call delegate node
        call_delegate_node = list(lowered_module_node.users.keys())[0]
        # 5 args to lowered module are: delegated_payload, x, const1, const2, const3
        self.assertEqual(len(call_delegate_node.args), 5)
        lower_module = getattr(
            delegated.exported_program().graph_module, lowered_module_node.name
        )
        delegated_ep = lower_module.original_module
        self.assertEqual(len(delegated_ep.state_dict), 0)
        self.assertEqual(len(delegated_ep.graph_signature.buffers), 0)
        self.assertEqual(len(delegated_ep.graph_signature.parameters), 0)

        # check exported program is still runnable
        output = delegated.exported_program().module()(torch.ones(2, 2))
        reference_output = model(torch.ones(2, 2))
        self.assertTrue(torch.allclose(reference_output, output))

    def test_partitioner_tag_data(self):
        """
        We test here that when partitioners explicitly tag constant data nodes,
        then the partitioned ExportedProgram will own the data, and the data will
        be removed from the owning program.
        """

        class PartitionerTagData(Partitioner):
            def __init__(self):
                super().__init__()
                self.delegation_spec = DelegationSpec(
                    ExecutorBackend.__name__,
                    [CompileSpec(key, value) for key, value in self.spec.items()],
                )

            def partition(
                self, edge_exported_program: ExportedProgram
            ) -> PartitionResult:
                partition_tags = {}
                for node in edge_exported_program.graph.nodes:
                    if node.op == "call_function" and node.target in [
                        exir_ops.edge.aten.add.Tensor
                    ]:
                        delegation_tag = "tag0"
                        node.meta["delegation_tag"] = delegation_tag
                        partition_tags[delegation_tag] = self.delegation_spec

                    if node.op == "placeholder" and (
                        is_param(edge_exported_program, node)
                        or is_buffer(edge_exported_program, node)
                    ):
                        delegation_tag = "tag0"
                        node.meta["delegation_tag"] = delegation_tag
                        partition_tags[delegation_tag] = self.delegation_spec

                return PartitionResult(
                    tagged_exported_program=edge_exported_program,
                    partition_tags=partition_tags,
                )

        model = capture_pre_autograd_graph(self.AddConst(), (torch.ones(2, 2),))
        edge = exir.to_edge(export(model, (torch.ones(2, 2),)))
        delegated = edge.to_backend(PartitionerTagData())

        # Check Owning Program still owns all constant data
        owning_program = delegated.exported_program()
        self.assertEqual(len(owning_program.state_dict), 0)
        self.assertEqual(len(owning_program.graph_signature.buffers), 0)
        self.assertEqual(len(owning_program.graph_signature.parameters), 0)

        # Check Lowered Module Exported Program does not have any constant data
        lowered_module_nodes = get_delegates(delegated.exported_program().graph)
        self.assertEqual(len(lowered_module_nodes), 1)
        lowered_module_node = lowered_module_nodes[0]

        # get call delegate node
        call_delegate_node = list(lowered_module_node.users.keys())[0]
        # 5 args to lowered module are: delegated_payload, x
        self.assertEqual(len(call_delegate_node.args), 2)
        lower_module = getattr(
            delegated.exported_program().graph_module, lowered_module_node.name
        )
        delegated_ep = lower_module.original_module
        self.assertEqual(len(delegated_ep.state_dict), 3)
        self.assertEqual(len(delegated_ep.graph_signature.buffers), 2)
        self.assertEqual(len(delegated_ep.graph_signature.parameters), 1)

        # check exported program is still runnable
        output = delegated.exported_program().module()(torch.ones(2, 2))
        reference_output = model(torch.ones(2, 2))
        self.assertTrue(torch.allclose(reference_output, output))

    def test_partitioner_tag_only_params(self):
        """
        We test here that when partitioners explicitly tag constant data nodes,
        then the partitioned ExportedProgram will own the data, and the data will
        be removed from the owning program.
        """

        class PartitionerTagData(Partitioner):
            def __init__(self):
                super().__init__()
                self.delegation_spec = DelegationSpec(
                    ExecutorBackend.__name__,
                    [CompileSpec(key, value) for key, value in self.spec.items()],
                )

            def partition(
                self, edge_exported_program: ExportedProgram
            ) -> PartitionResult:
                partition_tags = {}
                for node in edge_exported_program.graph.nodes:
                    if node.op == "call_function" and node.target in [
                        exir_ops.edge.aten.add.Tensor
                    ]:
                        delegation_tag = "tag0"
                        node.meta["delegation_tag"] = delegation_tag
                        partition_tags[delegation_tag] = self.delegation_spec

                    if node.op == "placeholder" and (
                        is_param(edge_exported_program, node)
                    ):
                        delegation_tag = "tag0"
                        node.meta["delegation_tag"] = delegation_tag
                        partition_tags[delegation_tag] = self.delegation_spec

                return PartitionResult(
                    tagged_exported_program=edge_exported_program,
                    partition_tags=partition_tags,
                )

        model = capture_pre_autograd_graph(self.AddConst(), (torch.ones(2, 2),))
        edge = exir.to_edge(export(model, (torch.ones(2, 2),)))
        delegated = edge.to_backend(PartitionerTagData())

        # Check Owning Program still owns only buffers
        owning_program = delegated.exported_program()
        self.assertEqual(len(owning_program.state_dict), 2)
        self.assertEqual(len(owning_program.graph_signature.buffers), 2)
        self.assertEqual(len(owning_program.graph_signature.parameters), 0)

        # Check Lowered Module Exported Program does not own any buffers
        lowered_module_nodes = get_delegates(delegated.exported_program().graph)
        self.assertEqual(len(lowered_module_nodes), 1)
        lowered_module_node = lowered_module_nodes[0]

        # get call delegate node
        call_delegate_node = list(lowered_module_node.users.keys())[0]
        # 5 args to lowered module are: delegated_payload, x, buffer1, buffer2
        self.assertEqual(len(call_delegate_node.args), 4)
        lower_module = getattr(
            delegated.exported_program().graph_module, lowered_module_node.name
        )
        delegated_ep = lower_module.original_module
        self.assertEqual(len(delegated_ep.state_dict), 1)
        self.assertEqual(len(delegated_ep.graph_signature.buffers), 0)
        self.assertEqual(len(delegated_ep.graph_signature.parameters), 1)

        # check exported program is still runnable
        output = delegated.exported_program().module()(torch.ones(2, 2))
        reference_output = model(torch.ones(2, 2))
        self.assertTrue(torch.allclose(reference_output, output))

    def test_partitioner_splits_constant_data(self):
        """
        We test that we throw an error when constant data users are split
        between different delegated payloads or owning program.
        """

        class ReuseConstData(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.const = torch.ones(2, 2)

            def forward(self, x):
                y = x + self.const
                z = x - self.const
                return y, z

        class PartitionerTagData(Partitioner):
            def __init__(self):
                super().__init__()
                self.delegation_spec = DelegationSpec(
                    ExecutorBackend.__name__,
                    [CompileSpec(key, value) for key, value in self.spec.items()],
                )

            def partition(
                self, edge_exported_program: ExportedProgram
            ) -> PartitionResult:
                partition_tags = {}
                for node in edge_exported_program.graph.nodes:
                    if node.op == "call_function" and node.target in [
                        exir_ops.edge.aten.add.Tensor
                    ]:
                        delegation_tag = "tag0"
                        node.meta["delegation_tag"] = delegation_tag
                        partition_tags[delegation_tag] = self.delegation_spec

                    if node.op == "placeholder" and (
                        is_param(edge_exported_program, node)
                        or is_buffer(edge_exported_program, node)
                    ):
                        delegation_tag = "tag0"
                        node.meta["delegation_tag"] = delegation_tag
                        partition_tags[delegation_tag] = self.delegation_spec

                return PartitionResult(
                    tagged_exported_program=edge_exported_program,
                    partition_tags=partition_tags,
                )

        model = capture_pre_autograd_graph(ReuseConstData(), (torch.ones(2, 2),))
        edge = exir.to_edge(export(model, (torch.ones(2, 2),)))
        with self.assertRaises(RuntimeError) as error:
            _ = edge.to_backend(PartitionerTagData())

        self.assertEqual(
            "constant data node (arg0_1) is tagged with (tag0) but has user (aten_sub_tensor) which has tag (None)",
            str(error.exception),
        )
