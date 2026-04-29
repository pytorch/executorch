# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
import unittest

import executorch.exir.tests.models as models

import torch
from executorch import exir
from executorch.exir import to_edge
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.test.backend_with_compiler_demo import (
    BackendWithCompilerDemo,
)
from executorch.exir.backend.test.demo_backend import DemoBackend
from executorch.exir.schema import DelegateCall, Program

from executorch.extension.pybindings.portable_lib import (  # @manual
    _get_registered_backend_names,
    _load_for_executorch_from_buffer,
)
from torch.export import export


def _has_backend_with_compiler_demo() -> bool:
    """Check if BackendWithCompilerDemo is linked into the portable runtime."""
    try:
        return "BackendWithCompilerDemo" in _get_registered_backend_names()
    except Exception:
        return False


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

    @unittest.skipUnless(
        _has_backend_with_compiler_demo(),
        "BackendWithCompilerDemo not registered (build with EXECUTORCH_BUILD_TESTS=ON)",
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
            compile_config=exir.EdgeCompileConfig(
                _check_ir_validity=False, _use_edge_ops=True
            ),
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

    def test_emit_lowered_backend_module(self):
        module_list = [
            models.Emformer(),
            models.Repeat(),
            models.ElementwiseAdd(),
            models.MLP(),
            models.ModelWithUnusedArg(),
        ]

        edge_compile_config = exir.EdgeCompileConfig(
            _check_ir_validity=False, _use_edge_ops=True
        )

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

        edge_compile_config = exir.EdgeCompileConfig(
            _check_ir_validity=False, _use_edge_ops=True
        )

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
