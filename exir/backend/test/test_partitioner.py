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
from executorch.exir.backend.test.op_partitioner_demo import (
    AddAttributePartitionerDemo,
    AllNodesPartitionerDemo,
)
from executorch.exir.backend.utils import get_delegates, tag_constant_data

from executorch.exir.dialects._ops import ops as exir_ops

from executorch.exir.tests.models import MLP
from executorch.extension.pybindings.portable_lib import (  # @manual=//executorch/extension/pybindings:portable_lib
    _load_for_executorch_from_buffer,
)
from executorch.extension.pytree import tree_flatten
from torch._export.utils import is_buffer, is_lifted_tensor_constant, is_param
from torch.export import export, export_for_training
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
        model = export_for_training(mlp, example_inputs).module()
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
        model = export_for_training(mlp, example_inputs).module()
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
        model = export_for_training(mlp, example_inputs).module()
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

        model = export_for_training(self.AddConst(), (torch.ones(2, 2),)).module()
        edge = exir.to_edge(export(model, (torch.ones(2, 2),)))
        delegated = edge.to_backend(PartitionerNoTagData())

        # Check Owning Program still owns all constant data
        owning_program = delegated.exported_program()
        self.assertEqual(
            len(owning_program.state_dict) + len(owning_program.constants), 3
        )
        self.assertEqual(
            len(owning_program.graph_signature.buffers)
            + len(owning_program.graph_signature.lifted_tensor_constants),
            2,
        )
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
                        or is_lifted_tensor_constant(edge_exported_program, node)
                    ):
                        delegation_tag = "tag0"
                        node.meta["delegation_tag"] = delegation_tag
                        partition_tags[delegation_tag] = self.delegation_spec

                return PartitionResult(
                    tagged_exported_program=edge_exported_program,
                    partition_tags=partition_tags,
                )

        model = export_for_training(self.AddConst(), (torch.ones(2, 2),)).module()
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
        self.assertEqual(len(delegated_ep.state_dict) + len(delegated_ep.constants), 3)
        self.assertEqual(
            len(delegated_ep.graph_signature.buffers)
            + len(delegated_ep.graph_signature.lifted_tensor_constants),
            2,
        )
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

        model = export_for_training(self.AddConst(), (torch.ones(2, 2),)).module()
        edge = exir.to_edge(export(model, (torch.ones(2, 2),)))
        delegated = edge.to_backend(PartitionerTagData())

        # Check Owning Program still owns only buffers
        owning_program = delegated.exported_program()
        self.assertEqual(
            len(owning_program.state_dict) + len(owning_program.constants), 2
        )
        self.assertEqual(
            len(owning_program.graph_signature.buffers)
            + len(owning_program.graph_signature.lifted_tensor_constants),
            2,
        )
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

        inputs = (torch.ones(2, 2),)
        model = export_for_training(ReuseConstData(), (torch.ones(2, 2),)).module()
        edge = exir.to_edge(export(model, (torch.ones(2, 2),)))
        exec_prog = edge.to_backend(PartitionerTagData()).to_executorch()
        executorch_module = _load_for_executorch_from_buffer(exec_prog.buffer)
        inputs_flattened, _ = tree_flatten(inputs)

        # Send the input from server executor to client executor, and receive the result from client executor
        _ = executorch_module.run_method("forward", inputs)

    def test_partitioner_alert_split_constant_data(self):
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
                        or is_lifted_tensor_constant(edge_exported_program, node)
                    ):
                        delegation_tag = "tag0"
                        node.meta["delegation_tag"] = delegation_tag
                        node.meta["no_copy"] = True
                        partition_tags[delegation_tag] = self.delegation_spec

                return PartitionResult(
                    tagged_exported_program=edge_exported_program,
                    partition_tags=partition_tags,
                )

        model = export_for_training(ReuseConstData(), (torch.ones(2, 2),)).module()
        edge = exir.to_edge(export(model, (torch.ones(2, 2),)))
        with self.assertRaises(RuntimeError) as error:
            _ = edge.to_backend(PartitionerTagData())

        self.assertTrue(
            "is tagged with (tag0) but has user (aten_sub_tensor) which has tag (None)"
            in str(error.exception),
        )

    def test_not_delegate_mutable_buffers(self) -> None:
        """
        A test case to check the mutated buffer is not delegated. We'll need to add a test case
        to consider when the delegate can consume the mutable buffer.
        """

        class MutableStateModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("my_state", torch.zeros(1))

            def forward(self, x):
                y = x + self.my_state
                self.my_state.add_(1)
                return y

        edge = exir.to_edge(
            torch.export.export(
                MutableStateModule(),
                (torch.zeros(1),),
            )
        )
        self.assertGreater(
            len(edge.exported_program().graph_signature.buffers_to_mutate),
            0,
            "The test case should at leaset one mutable buffer",
        )

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
                tag_constant_data(edge_exported_program)
                return PartitionResult(
                    tagged_exported_program=edge_exported_program,
                    partition_tags=partition_tags,
                )

        # Check the edge program inital buffers_to_mutate
        mutate_op = "aten_add_tensor_1"
        self.assertEqual(
            edge.exported_program().graph_signature.buffers_to_mutate[mutate_op],
            "my_state",
        )
        edge = edge.to_backend(PartitionerTagData())
        # After to_backend, add is delegated and is no longer in buffers_to_mutate.
        self.assertNotIn(
            mutate_op,
            edge.exported_program().graph_signature.buffers_to_mutate,
        )

        mutate_op = "getitem_1"
        # Ensure the mutated buffer is not delegated, and the new mutate node is getitem (from call_delegate)
        self.assertEqual(
            edge.exported_program().graph_signature.buffers_to_mutate[mutate_op],
            "my_state",
        )
        # Check the copy_ node is inserted
        edge = edge.to_executorch()
        copy_node = [
            node
            for node in edge.exported_program().graph.nodes
            if node.op == "call_function"
            and node.target == torch.ops.aten.copy_.default
        ]
        self.assertEqual(len(copy_node), 1)

    def test_buffer_mutation1(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("b", torch.ones(3, 3))

            def forward(self, x):
                self.b.add_(x)
                return x + self.b

        model_inputs = (torch.ones(3, 3),)
        orig_res = TestModule()(*model_inputs)
        edge_program = exir.to_edge(torch.export.export(TestModule(), model_inputs))
        lowered = edge_program.to_backend(AddAttributePartitionerDemo())

        self.assertTrue(
            torch.allclose(lowered.exported_program().module()(*model_inputs), orig_res)
        )

        self.assertEqual(
            len(lowered.exported_program().graph_signature.buffers_to_mutate),
            0,
        )
        lowered_module_nodes = get_delegates(lowered.exported_program().graph)
        self.assertEqual(len(lowered_module_nodes), 1)
        lowered_module_node = lowered_module_nodes[0]

        # get call delegate node
        call_delegate_node = list(lowered_module_node.users.keys())[0]
        self.assertEqual(len(call_delegate_node.args), 2)

        lower_module = getattr(
            lowered.exported_program().graph_module, lowered_module_node.name
        )
        delegated_ep = lower_module.original_module

        self.assertEqual(len(delegated_ep.state_dict), 1)
        self.assertEqual(len(delegated_ep.graph_signature.buffers_to_mutate), 1)
        self.assertEqual(len(delegated_ep.graph_signature.buffers), 1)

    def test_buffer_mutation_llama_repro(self):
        SHAPE = (2, 3)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(SHAPE, dtype=torch.float32))

            def forward(self, q, k_val, input_pos):
                q_T = q.transpose(0, 1)
                k = torch.ops.aten.index_put_(self.cache, [input_pos, None], k_val)
                attn = k.mm(q_T)
                return attn

        q = torch.rand(1, 3)
        k = torch.rand(1, 3)
        example_inputs = (q, k, torch.tensor([1, 1]))

        model = Model()
        model.eval()

        exir_program_aten = torch.export.export(model, example_inputs)
        exir_program_aten.module()(*example_inputs)
        edge_program_manager = exir.to_edge(exir_program_aten)
        lowered = edge_program_manager.to_backend(AllNodesPartitionerDemo())

        self.assertEqual(
            len(lowered.exported_program().graph_signature.buffers_to_mutate),
            0,
        )
        lowered_module_nodes = get_delegates(lowered.exported_program().graph)
        self.assertEqual(len(lowered_module_nodes), 1)
        lowered_module_node = lowered_module_nodes[0]

        # get call delegate node
        call_delegate_node = list(lowered_module_node.users.keys())[0]
        self.assertEqual(len(call_delegate_node.args), 4)

        lower_module = getattr(
            lowered.exported_program().graph_module, lowered_module_node.name
        )
        delegated_ep = lower_module.original_module

        self.assertEqual(len(delegated_ep.state_dict), 1)
        self.assertEqual(len(delegated_ep.graph_signature.buffers_to_mutate), 1)
        self.assertEqual(len(delegated_ep.graph_signature.buffers), 1)

    def test_buffer_mutation_unsupported(self):
        SHAPE = (2, 3)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state_1", torch.zeros(SHAPE, dtype=torch.float32))

            def forward(self, x):
                add = self.state_1.add_(x)
                return add

        model = Model()
        model.eval()

        example_inputs = (torch.randn(SHAPE),)
        exir_program_aten = torch.export.export(model, example_inputs)
        edge_program_manager = exir.to_edge(exir_program_aten)
        with self.assertRaises(AssertionError):
            edge_program_manager.to_backend(AddAttributePartitionerDemo())
