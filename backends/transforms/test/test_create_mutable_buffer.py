# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

import executorch
import torch
from executorch.backends.transforms.utils import create_mutable_buffer
from executorch.exir import ExecutorchBackendConfig, to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.extension.pybindings.portable_lib import (  # @manual
    _load_for_executorch_from_buffer,
    Verification,
)
from torch.export import export
from torch.utils._pytree import tree_flatten


class TestMutableBufferCreation(unittest.TestCase):
    """
    Test suite for the create_mutable_buffer utility function.
    """

    def test_create_mutable_buffer(self):
        """
        Tests the utility function create_mutable_buffer which creates a mutable buffer
        that can be modified during execution.
        """

        class EmptyNetwork(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

            test_data: torch.Tensor = (torch.zeros(1),)

        module = EmptyNetwork()
        exported_program = export(module, args=module.test_data, strict=True)
        exported_program = to_edge(exported_program).exported_program()
        graph = exported_program.graph_module.graph

        assert len(graph.nodes) == 2
        assert exported_program.module()(torch.zeros(1)) == 0
        assert len(exported_program.graph_signature.input_specs) == 1
        assert len(exported_program.graph_signature.output_specs) == 1
        assert len(exported_program.state_dict) == 0

        buffer_name = "b_test_mutable_buffer"
        target_name = buffer_name[2:]  # Remove the "b_" prefix
        initial_data = torch.ones(1) * 2  # Initialize with value 2

        # Create a mutable buffer using create_mutable_buffer
        buffer_node = create_mutable_buffer(
            exp_program=exported_program,
            name=buffer_name,
            data=initial_data,
        )
        assert "val" in buffer_node.meta

        # Verify the buffer was created correctly
        input_node = list(graph.nodes)[
            1
        ]  # Original input node (buffer_node is now first)

        # Create an add operation that uses the mutable buffer
        with graph.inserting_after(input_node):
            graph.create_node(
                "call_function",
                exir_ops.edge.aten.add.Tensor,
                args=(input_node, buffer_node),
                kwargs={},
            )

        # We should now have four nodes: buffer, input, add, output
        assert len(graph.nodes) == 4

        assert target_name in exported_program.state_dict
        assert torch.equal(exported_program.state_dict[target_name], initial_data)

        # Check that buffer is properly referenced in graph signature
        assert buffer_name in exported_program.graph_signature.inputs_to_buffers
        assert (
            exported_program.graph_signature.inputs_to_buffers[buffer_name]
            == target_name
        )
        assert (
            exported_program.graph_signature.buffers_to_mutate[buffer_name]
            == target_name
        )


class TestRegisterMutableBufferPass(unittest.TestCase):
    """
    This test is to test the `create_mutable_buffer` utility.
    """

    class HelperPass(ExportPass):
        def __init__(
            self,
            exported_program: torch.export.ExportedProgram,
            buf_data: torch.Tensor,
        ):
            super().__init__()
            self.registered_buffers = set()
            self.exported_program = exported_program
            self.buf_data = buf_data

        def call(self, graph_module: torch.fx.GraphModule):
            """
            This pass will register a mutable buffer for the add op(s) in the graph.
            It will insert a new index_put_ op to the graph to update the buffer using the output of the add op.
            And adjust the output of the graph to return the buffer or the index_put_ op.
            """
            modified = False

            assert (
                len(graph_module.graph.output_node().args[0]) == 1
            ), "For this pass, expecting only one output i.e. add"

            for node in graph_module.graph.nodes:
                if (
                    node.op == "call_function"
                    and node.target == exir_ops.edge.aten.add.Tensor
                ):
                    # To match what we do in export.
                    suffix = (
                        ""
                        if len(self.registered_buffers) == 0
                        else f"{len(self.registered_buffers) + 1}"
                    )
                    buffer_name = f"b_my_buffer{suffix}"
                    self.registered_buffers.add(buffer_name)

                    # Utility under test!
                    buf_node = create_mutable_buffer(
                        self.exported_program,
                        data=self.buf_data,
                        name=buffer_name,
                    )

                    # Assuming `indices` is always available
                    indices = [
                        node
                        for node in graph_module.graph.nodes
                        if node.op == "placeholder" and node.name == "indices"
                    ]
                    with graph_module.graph.inserting_after(node):
                        index_put_node = graph_module.graph.create_node(
                            "call_function",
                            exir_ops.edge.aten.index_put_.default,
                            (
                                buf_node,
                                indices,
                                node,
                            ),
                            {},
                        )

                        # Replace the old add output with index_put_node
                        output_node = graph_module.graph.output_node()
                        outputs = list(output_node.args[0])
                        if node in outputs:
                            outputs[-1] = (
                                index_put_node  # Replace the old add output with index_put_node
                            )
                            output_node.args = (outputs,)

                            # update the output node name in the graph signature
                            graph_signature = self.exported_program.graph_signature
                            graph_signature.replace_all_uses(
                                node.name, index_put_node.name
                            )
                            self.exported_program._graph_signature = graph_signature

                    modified = True

            if modified:
                graph_module = super().call(graph_module).graph_module
                graph_module.graph.lint()
                graph_module.graph.eliminate_dead_code()
                graph_module.recompile()
            return PassResult(graph_module, modified)

    def _test_edge_pass(self, model, example_inputs, num_lifted_args=1):
        exported = export(model, example_inputs)

        edge_program = to_edge(
            exported,
            # for torch._export.verifier.SpecViolationError:
            #   Operator torch._ops.aten.index_put_.default is not in Core ATen opset
            compile_config=executorch.exir.EdgeCompileConfig(_check_ir_validity=False),
        )
        # Tensor data for the new buffer(s)
        buffer_tensor = torch.zeros(2)

        transformed_example_inputs = (
            *(buffer_tensor for i in range(num_lifted_args)),
            *example_inputs,
        )
        transformed_ep = edge_program.transform(
            passes=[
                TestRegisterMutableBufferPass.HelperPass(
                    edge_program.exported_program(), buffer_tensor
                )
            ]
        )
        transformed_edge_gm = transformed_ep.exported_program().graph_module
        # Make sure it works
        transformed_edge_gm(*transformed_example_inputs)

        # Explicitly passing inplace_pass to make sure it works with our manually inserted buffer and index_put_ node(s).
        executorch_program_manager = edge_program.to_executorch(
            ExecutorchBackendConfig(
                emit_mutable_buffer_names=True, run_reinplace_pass=True
            )
        )
        return executorch_program_manager

    def _test_eager(self, model, example_inputs, num_lifted_args=1):
        exported = export(model, example_inputs, strict=True)

        edge_program = executorch.exir.to_edge(exported)
        edge_gm = edge_program.exported_program().graph_module

        # Adding buffer as an extra pos[0] arg
        buffer_tensor = torch.zeros(2)
        edge_example_inputs = (
            *(buffer_tensor for i in range(num_lifted_args)),
            *example_inputs,
        )
        # Make sure it works
        edge_gm(*edge_example_inputs)

        executorch_program_manager = edge_program.to_executorch(
            ExecutorchBackendConfig(
                emit_mutable_buffer_names=True, run_reinplace_pass=True
            )
        )
        return executorch_program_manager

    def _compare_outputs(self, et_ep1, et_ep2, example_inputs):

        def run(et_pm, inputs):
            buffer = et_pm.buffer
            inputs_flattened, _ = tree_flatten(inputs)
            executorch_module = _load_for_executorch_from_buffer(
                buffer, program_verification=Verification.Minimal
            )
            executorch_output = copy.deepcopy(
                executorch_module.run_method("forward", tuple(inputs_flattened))
            )
            return executorch_output

        # compare the outputs of the two programs
        output1 = run(et_ep1, example_inputs)
        output2 = run(et_ep2, example_inputs)
        assert len(output1) == len(output2)
        for o1, o2 in zip(output1, output2):
            self.assertTrue(torch.allclose(o1, o2))

    def _compare_ep_state_dict(self, et_ep1, et_ep2):
        # compare the state dict of the two programs
        state_dict1 = et_ep1.exported_program().state_dict
        state_dict2 = et_ep2.exported_program().state_dict
        self.assertEqual(len(state_dict1), len(state_dict2))
        # a bit fragile comparing the names, but the util tries to match the names i.e. `b_my_buffer` and `b_my_buffer2`
        for k, _ in state_dict1.items():
            self.assertTrue(
                k in state_dict2, f"{state_dict1.keys()} != {state_dict2.keys()} @ {k}"
            )

    def _compare_signatures(self, et_ep1, et_ep2):
        # compare the graph signatures
        def _input_spec_compare(input_spec1, input_spec2):
            self.assertEqual(
                input_spec1.kind,
                input_spec2.kind,
                f"{input_spec1.kind} != {input_spec2.kind}",
            )
            self.assertEqual(input_spec1.arg, input_spec2.arg)
            self.assertEqual(
                input_spec1.target,
                input_spec2.target,
                f"{input_spec1.target} != {input_spec2.target}",
            )
            self.assertEqual(
                input_spec1.persistent,
                input_spec2.persistent,
                f"{input_spec1.persistent} != {input_spec2.persistent}",
            )

        def _output_spec_compare(output_spec1, output_spec2):
            self.assertEqual(
                output_spec1.kind,
                output_spec2.kind,
                f"{output_spec1.kind} != {output_spec2.kind}",
            )
            # TODO: Look into why the output names are different,
            # and not updated by the buffer_write_back_pass when the buffer
            # is inserted via a pass and used by an inplace op.
            # self.assertEqual(output_spec1.arg, output_spec2.arg)
            self.assertEqual(
                output_spec1.target,
                output_spec2.target,
                f"{output_spec1.target} != {output_spec2.target}",
            )

        graph_signature1 = et_ep1.exported_program().graph_signature
        graph_signature2 = et_ep2.exported_program().graph_signature

        # compare input spec order, kind and targets
        self.assertEqual(
            len(graph_signature1.input_specs), len(graph_signature2.input_specs)
        )
        for i1, i2 in zip(graph_signature1.input_specs, graph_signature2.input_specs):
            _input_spec_compare(i1, i2)

        # compare output spec order, kind and targets
        self.assertEqual(
            len(graph_signature1.output_specs), len(graph_signature2.output_specs)
        )
        for o1, o2 in zip(graph_signature1.output_specs, graph_signature2.output_specs):
            _output_spec_compare(o1, o2)

    def _compare_plan_ops(self, et1, et2):
        operators1 = et1.executorch_program.execution_plan[0].operators
        operators2 = et2.executorch_program.execution_plan[0].operators
        for op1, op2 in zip(operators1, operators2):
            self.assertEqual(op1.name, op2.name, f"{op1.name} != {op2.name}")
            self.assertEqual(
                op1.overload, op2.overload, f"{op1.overload} != {op2.overload}"
            )

    def compare(self, et1, et2, example_inputs):
        self._compare_signatures(et1, et2)
        self._compare_ep_state_dict(et1, et2)
        self._compare_plan_ops(et1, et2)
        self._compare_outputs(et1, et2, example_inputs)

    def test_basic(self):
        class CustomModuleGraph(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, indices):
                output = x + x
                return output

        class CustomModuleSrc(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("my_buffer", torch.zeros(2))

            def forward(self, x, indices):
                output = x + x
                self.my_buffer.index_put_((indices,), output)
                return output

        example_inputs = (torch.ones(2), torch.tensor([0, 1]))

        with torch.no_grad():
            graph_model = CustomModuleGraph().eval()
            et_1 = self._test_edge_pass(graph_model, example_inputs)

            src_model = CustomModuleSrc().eval()
            et_2 = self._test_eager(src_model, example_inputs)

        self.compare(et_1, et_2, example_inputs)

    def test_basic_with_param(self):
        input_tensor = torch.ones(2)

        class CustomModuleGraph(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_parameter("my_param", torch.nn.Parameter(input_tensor))

            def forward(self, x, indices):
                output = x + self.my_param
                return output

        class CustomModuleSrc(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("my_buffer", torch.zeros(2))
                self.register_parameter("my_param", torch.nn.Parameter(input_tensor))

            def forward(self, x, indices):
                output = x + self.my_param
                self.my_buffer.index_put_((indices,), output)
                return output

        example_inputs = (input_tensor, torch.tensor([0, 1]))
        with torch.no_grad():
            graph_model = CustomModuleGraph().eval()
            et1 = self._test_edge_pass(graph_model, example_inputs, num_lifted_args=2)

            src_model = CustomModuleSrc().eval()
            et2 = self._test_eager(src_model, example_inputs, num_lifted_args=2)

        self.compare(et1, et2, example_inputs)

    def test_basic_with_constant(self):
        input_tensor = torch.ones(2) * 2

        class CustomModuleGraph(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.constant = input_tensor

            def forward(self, x, indices):
                output = x + self.constant
                return output

        class CustomModuleSrc(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("my_buffer", torch.zeros(2))
                self.constant = input_tensor

            def forward(self, x, indices):
                output = x + self.constant
                self.my_buffer.index_put_((indices,), output)
                return output

        example_inputs = (torch.ones(2), torch.tensor([0, 1]))
        with torch.no_grad():
            graph_model = CustomModuleGraph().eval()
            et1 = self._test_edge_pass(graph_model, example_inputs, num_lifted_args=2)

            src_model = CustomModuleSrc().eval()
            et2 = self._test_eager(src_model, example_inputs, num_lifted_args=2)

        self.compare(et1, et2, example_inputs)

    def test_single(self):
        class CustomModuleGraph(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, indices):
                output = x + x
                return output

        class CustomModuleSrc(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("my_buffer", torch.zeros(2))

            def forward(self, x, indices):
                output = x + x
                self.my_buffer.index_put_((indices,), output)
                return self.my_buffer

        example_inputs = (torch.ones(2), torch.tensor([0, 1]))

        graph_model = CustomModuleGraph().eval()
        et1 = self._test_edge_pass(graph_model, example_inputs)

        src_model = CustomModuleSrc().eval()
        et2 = self._test_eager(src_model, example_inputs)

        self.compare(et1, et2, example_inputs)

    def test_double(self):
        class CustomModuleGraph(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, indices):
                output = x + x
                output = output + x
                return output

        class CustomModuleSrc(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("my_buffer", torch.zeros(2))
                self.register_buffer("my_buffer2", torch.zeros(2))

            def forward(self, x, indices):
                output = x + x
                self.my_buffer.index_put_((indices,), output)

                output = output + x
                self.my_buffer2.index_put_((indices,), output)

                return output

        example_inputs = (torch.ones(2), torch.tensor([0, 1]))

        graph_model = CustomModuleGraph().eval()
        et1 = self._test_edge_pass(graph_model, example_inputs, num_lifted_args=2)

        src_model = CustomModuleSrc().eval()
        et2 = self._test_eager(src_model, example_inputs, num_lifted_args=2)

        self.compare(et1, et2, example_inputs)
