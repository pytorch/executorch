# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import io
import operator
import unittest
from unittest.mock import patch

import executorch.exir.tests.models as models

import torch
from executorch import exir
from executorch.exir import EdgeProgramManager, to_edge, to_edge_transform_and_lower
from executorch.exir._serialize import _PTEFile, _serialize_pte_binary
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.test.backend_with_compiler_demo import (
    BackendWithCompilerDemo,
)
from executorch.exir.backend.test.demo_backend import DemoBackend
from executorch.exir.backend.test.op_partitioner_demo import AddMulPartitionerDemo
from executorch.exir.schema import DelegateCall, DelegateScratch, Program
from executorch.exir.tensor import make_allocation_info

from executorch.extension.pybindings.portable_lib import (  # @manual
    _load_for_executorch_from_buffer,
)
from torch.export import export


class TestBackendAPI(unittest.TestCase):
    def validate_lowered_module_program(self, program: Program) -> None:
        """
        For any program emitted from lowered_backend_module, we expect only one delegate call
        """
        # there should only be one instruction
        self.assertEqual(
            len(program.execution_plan[0].chains[0].instructions),
            1,
        )

        # the only instruction should be a delegate call
        self.assertTrue(
            isinstance(
                program.execution_plan[0].chains[0].instructions[0].instr_args,
                DelegateCall,
            )
        )

    def get_program_from_wrapped_module(
        self, lowered_module, example_inputs, edge_compile_config
    ):
        class WrappedModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.one_module = lowered_module

            def forward(self, *args):
                return self.one_module(*args)

        return (
            to_edge(
                export(WrappedModule(), example_inputs, strict=True),
                compile_config=edge_compile_config,
            )
            .to_executorch()
            .executorch_program
        )

    def test_emit_lowered_backend_module_end_to_end(self):
        class SinModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.sin(x)

        sin_module = SinModule()
        model_inputs = (torch.ones(1),)
        expected_res = sin_module(*model_inputs)
        edgeir_m = to_edge(
            export(sin_module, model_inputs, strict=True),
            compile_config=exir.EdgeCompileConfig(_check_ir_validity=False),
        )
        max_value = model_inputs[0].shape[0]
        compile_specs = [CompileSpec("max_value", bytes([max_value]))]
        lowered_sin_module = to_backend(
            BackendWithCompilerDemo.__name__, edgeir_m.exported_program(), compile_specs
        )

        new_res = lowered_sin_module(*model_inputs)

        self.assertTrue(torch.allclose(new_res[0], expected_res))
        program = lowered_sin_module.program()
        self.validate_lowered_module_program(program)
        buff = lowered_sin_module.buffer()

        executorch_module = _load_for_executorch_from_buffer(buff)
        model_inputs = torch.ones(1)
        model_outputs = executorch_module.forward([model_inputs])
        self.assertEqual(
            model_inputs,
            torch.ones(1),
        )
        expected_res = 0.8333 * torch.ones(1)

        self.assertTrue(
            torch.allclose(model_outputs[0], expected_res, atol=1e-03, rtol=1e-03)
        )

    def _repeated_sin_calls(self, backend_id, compile_specs, num_calls):
        """Lowers sin once and returns it with a module that calls it num_calls times."""

        class SinModule(torch.nn.Module):
            def forward(self, x):
                return torch.sin(x)

        model_inputs = (torch.ones(1),)
        lowered_sin_module = to_backend(
            backend_id,
            to_edge(export(SinModule(), model_inputs, strict=True)).exported_program(),
            compile_specs,
        )

        class RepeatedCalls(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sin = lowered_sin_module

            def forward(self, x):
                for _ in range(num_calls):
                    x = self.sin(x)
                return x

        edge = to_edge(export(RepeatedCalls(), model_inputs, strict=True))
        return lowered_sin_module, edge, model_inputs

    def _delegate_indices(self, plan):
        return [
            instruction.instr_args.delegate_index
            for instruction in plan.chains[0].instructions
            if isinstance(instruction.instr_args, DelegateCall)
        ]

    def test_calls_of_one_lowered_module_share_one_delegate(self):
        lowered_sin_module, edge, _ = self._repeated_sin_calls(
            BackendWithCompilerDemo.__name__,
            [CompileSpec("max_value", bytes([1]))],
            num_calls=3,
        )
        self.assertTrue(copy.deepcopy(lowered_sin_module).share_handle_across_calls)

        plan = edge.to_executorch().executorch_program.execution_plan[0]
        self.assertEqual(len(plan.delegates), 1)
        self.assertEqual(self._delegate_indices(plan), [0, 0, 0])
        args = [
            tuple(instruction.instr_args.args)
            for instruction in plan.chains[0].instructions
            if isinstance(instruction.instr_args, DelegateCall)
        ]
        self.assertEqual(len(set(args)), 3)

    def test_shared_delegate_matches_separate_calls_at_runtime(self):
        lowered_sin_module, edge, model_inputs = self._repeated_sin_calls(
            BackendWithCompilerDemo.__name__,
            [CompileSpec("max_value", bytes([1]))],
            num_calls=3,
        )
        program = edge.to_executorch()

        expected = model_inputs[0]
        single_call = _load_for_executorch_from_buffer(lowered_sin_module.buffer())
        for _ in range(3):
            expected = single_call.forward([expected])[0]
        outputs = _load_for_executorch_from_buffer(program.buffer).forward(
            list(model_inputs)
        )
        self.assertTrue(torch.equal(outputs[0], expected))

    def test_sharing_survives_a_serde_round_trip(self):
        _, edge, _ = self._repeated_sin_calls(
            BackendWithCompilerDemo.__name__,
            [CompileSpec("max_value", bytes([1]))],
            num_calls=2,
        )
        buffer = io.BytesIO()
        exir.save(edge.exported_program(), buffer)
        buffer.seek(0)

        plan = (
            EdgeProgramManager(exir.load(buffer))
            .to_executorch()
            .executorch_program.execution_plan[0]
        )
        self.assertEqual(self._delegate_indices(plan), [0, 0])

    def test_runtime_rejects_a_shared_delegate_with_scratch(self):
        _, edge, model_inputs = self._repeated_sin_calls(
            BackendWithCompilerDemo.__name__,
            [CompileSpec("max_value", bytes([1]))],
            num_calls=2,
        )

        def delegate_calls(program):
            return [
                instruction.instr_args
                for instruction in program.execution_plan[0].chains[0].instructions
                if isinstance(instruction.instr_args, DelegateCall)
            ]

        program = edge.to_executorch().executorch_program
        scratch = [DelegateScratch(allocation=make_allocation_info(1, 0), size=4)]

        def run(program):
            return _load_for_executorch_from_buffer(
                bytes(_serialize_pte_binary(pte_file=_PTEFile(program=program)))
            ).forward(list(model_inputs))

        # The same scratch on two separate delegates is valid, so the errors
        # below come from the sharing alone.
        separate = copy.deepcopy(program)
        delegates = separate.execution_plan[0].delegates
        delegates.append(copy.deepcopy(delegates[0]))
        delegate_calls(separate)[1].delegate_index = 1
        for call in delegate_calls(separate):
            call.scratch = scratch
        run(separate)

        # The emitter never writes these, but another producer could.
        for calls_with_scratch in ([0, 1], [0], [1]):
            shared = copy.deepcopy(program)
            for index in calls_with_scratch:
                delegate_calls(shared)[index].scratch = scratch
            with self.assertRaisesRegex(RuntimeError, "error: 0x23"):
                run(shared)

    def test_partitioned_module_shares_one_delegate_across_calls(self):
        class AddModule(torch.nn.Module):
            def forward(self, x):
                return x + x

        model_inputs = (torch.ones(2, 2),)
        add = (
            to_edge_transform_and_lower(
                export(AddModule(), model_inputs, strict=True),
                partitioner=[AddMulPartitionerDemo()],
            )
            .exported_program()
            .module()
        )

        class TwoCalls(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.add = add

            def forward(self, x):
                return self.add(self.add(x))

        plan = (
            to_edge(export(TwoCalls(), model_inputs, strict=False))
            .to_executorch()
            .executorch_program.execution_plan[0]
        )
        self.assertEqual(self._delegate_indices(plan), [0, 0])

    def test_each_call_uses_the_delegate_of_its_own_module(self):
        class SinModule(torch.nn.Module):
            def forward(self, x):
                return torch.sin(x)

        model_inputs = (torch.ones(1),)
        edge_sin = to_edge(export(SinModule(), model_inputs, strict=True))
        compile_specs = [CompileSpec("max_value", bytes([1]))]
        first, second = (
            to_backend(
                BackendWithCompilerDemo.__name__,
                edge_sin.exported_program(),
                compile_specs,
            )
            for _ in range(2)
        )

        class TwoModules(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.first = first
                self.second = second

            def forward(self, x):
                return self.first(self.second(self.first(x)))

        program = (
            to_edge(export(TwoModules(), model_inputs, strict=True))
            .to_executorch()
            .executorch_program
        )
        # Identical blobs are stored once, but separate modules keep separate
        # delegates, and a repeated call goes back to its own module's delegate.
        self.assertEqual(len(program.backend_delegate_data), 1)
        self.assertEqual(self._delegate_indices(program.execution_plan[0]), [0, 1, 0])

    def test_calls_keep_their_own_delegate_unless_the_backend_allows_sharing(self):
        _, edge, _ = self._repeated_sin_calls(DemoBackend.__name__, [], num_calls=2)

        plan = edge.to_executorch().executorch_program.execution_plan[0]
        self.assertEqual(len(plan.delegates), 2)
        self.assertEqual(self._delegate_indices(plan), [0, 1])

    def test_emit_lowered_backend_module(self):
        module_list = [
            models.Emformer(),
            models.Repeat(),
            models.ElementwiseAdd(),
            models.MLP(),
            models.ModelWithUnusedArg(),
        ]

        edge_compile_config = exir.EdgeCompileConfig(_check_ir_validity=False)

        for model in module_list:
            model_inputs = model.get_random_inputs()

            edgeir_m = to_edge(
                export(model, model_inputs, strict=True),
                compile_config=edge_compile_config,
            )
            lowered_model = to_backend(
                DemoBackend.__name__, edgeir_m.exported_program(), []
            )
            program = lowered_model.program()
            reference_program = self.get_program_from_wrapped_module(
                lowered_model, model_inputs, edge_compile_config
            )

            # Check program is fairly equal to the reference program
            self.assertEqual(
                len(program.execution_plan[0].chains[0].instructions),
                len(reference_program.execution_plan[0].chains[0].instructions),
            )

            self.assertEqual(
                len(program.execution_plan[0].values),
                len(reference_program.execution_plan[0].values),
            )

            self.assertEqual(
                len(program.execution_plan[0].inputs),
                len(reference_program.execution_plan[0].inputs),
            )

            self.assertEqual(
                len(program.execution_plan[0].outputs),
                len(reference_program.execution_plan[0].outputs),
            )

            # Ensure we can get the buffer
            _ = lowered_model.buffer()
            self.validate_lowered_module_program(program)

    def test_emit_nested_lowered_backend_module(self):
        module_list = [
            models.Emformer(),
            models.Repeat(),
            models.ElementwiseAdd(),
            models.MLP(),
            models.ModelWithUnusedArg(),
        ]

        edge_compile_config = exir.EdgeCompileConfig(_check_ir_validity=False)

        for model in module_list:
            model_inputs = model.get_random_inputs()

            edgeir_m = to_edge(
                export(model, model_inputs, strict=True),
                compile_config=edge_compile_config,
            )
            lowered_module = to_backend(
                DemoBackend.__name__, edgeir_m.exported_program(), []
            )

            # This module will include one operator and two delegate call
            class WrappedModule(torch.nn.Module):
                def __init__(self, lowered_module):
                    super().__init__()
                    self.one_module = lowered_module

                def forward(self, *args):
                    return self.one_module(*args)

            wrapped_module = WrappedModule(lowered_module)
            wrapped_module_edge = to_edge(
                export(wrapped_module, model_inputs, strict=True),
                compile_config=edge_compile_config,
            )

            nested_lowered_model = to_backend(
                DemoBackend.__name__, wrapped_module_edge.exported_program(), []
            )

            program = nested_lowered_model.program()
            self.validate_lowered_module_program(program)

    def test_arrange_graph_outputs_reorders_mutations_before_user_outputs(self):
        """
        Directly test that arrange_graph_outputs correctly reorders a
        submodule's output tuple so that BUFFER_MUTATION outputs come before
        USER_OUTPUT outputs, and that getitem indices in the parent graph are
        remapped accordingly.
        """
        from executorch.exir.lowered_backend_module import arrange_graph_outputs
        from torch.export.exported_program import OutputKind, OutputSpec, TensorArgument

        # Build a submodule graph with 3 outputs in order:
        #   [user_out_0, buffer_mut_1, user_out_2]
        # The expected reordering is:
        #   [buffer_mut_1, user_out_0, user_out_2]
        sub_graph = torch.fx.Graph()
        x = sub_graph.placeholder("x")
        buf = sub_graph.placeholder("buf")
        add_node = sub_graph.call_function(torch.ops.aten.add.Tensor, (x, x))
        mul_node = sub_graph.call_function(torch.ops.aten.mul.Tensor, (buf, x))
        sub_node = sub_graph.call_function(torch.ops.aten.sub.Tensor, (x, x))
        # Output order: user, mutation, user
        sub_graph.output((add_node, mul_node, sub_node))
        sub_gm = torch.fx.GraphModule({}, sub_graph)

        output_specs = [
            OutputSpec(
                kind=OutputKind.USER_OUTPUT,
                arg=TensorArgument(name="add"),
                target=None,
            ),
            OutputSpec(
                kind=OutputKind.BUFFER_MUTATION,
                arg=TensorArgument(name="mul"),
                target="buf",
            ),
            OutputSpec(
                kind=OutputKind.USER_OUTPUT,
                arg=TensorArgument(name="sub"),
                target=None,
            ),
        ]

        # Build a parent graph with a call_module node and getitem users
        parent_graph = torch.fx.Graph()
        px = parent_graph.placeholder("x")
        call_mod = parent_graph.call_module("sub_mod", (px,))
        gi0 = parent_graph.call_function(operator.getitem, (call_mod, 0))
        gi1 = parent_graph.call_function(operator.getitem, (call_mod, 1))
        gi2 = parent_graph.call_function(operator.getitem, (call_mod, 2))
        parent_graph.output((gi0, gi1, gi2))

        # Run arrange_graph_outputs
        arrange_graph_outputs(sub_gm, output_specs, call_mod)

        # Verify output_specs are reordered: mutation first
        self.assertEqual(output_specs[0].kind, OutputKind.BUFFER_MUTATION)
        self.assertEqual(output_specs[1].kind, OutputKind.USER_OUTPUT)
        self.assertEqual(output_specs[2].kind, OutputKind.USER_OUTPUT)
        self.assertEqual(output_specs[0].target, "buf")

        # Verify the submodule graph output tuple is reordered
        output_node = None
        for node in sub_gm.graph.nodes:
            if node.op == "output":
                output_node = node
                break
        reordered = list(output_node.args[0])
        self.assertIs(reordered[0], mul_node)  # buffer mutation first
        self.assertIs(reordered[1], add_node)  # then user outputs
        self.assertIs(reordered[2], sub_node)

        # Verify getitem indices were remapped:
        #   old 0 (user) -> new 1
        #   old 1 (mutation) -> new 0
        #   old 2 (user) -> new 2 (unchanged)
        self.assertEqual(gi0.args[1], 1)
        self.assertEqual(gi1.args[1], 0)
        self.assertEqual(gi2.args[1], 2)

    def test_to_backend_copies_the_program_through_the_backend_hook(self):
        # A backend overrides copy_exported_program_for_preprocess to avoid copying
        # large constants, so to_backend must not deep-copy the program around it.
        model = models.MLP()
        edge_program = to_edge(
            export(model, model.get_random_inputs(), strict=True)
        ).exported_program()

        with patch.object(
            DemoBackend,
            "copy_exported_program_for_preprocess",
            return_value=edge_program,
        ) as copy_hook, patch.object(
            DemoBackend, "preprocess", wraps=DemoBackend.preprocess
        ) as preprocess:
            to_backend(DemoBackend.__name__, edge_program, [])

        copy_hook.assert_called_once_with(edge_program, [])
        self.assertIs(preprocess.call_args.args[0], edge_program)
