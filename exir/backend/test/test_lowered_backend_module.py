# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
from executorch.exir.backend.test.qnn_backend_demo import QnnBackend
from executorch.exir.schema import DelegateCall, Program

from executorch.extension.pybindings.portable_lib import (  # @manual
    _load_for_executorch_from_buffer,
)
from hypothesis import given, settings, strategies as st
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
                export(WrappedModule(), example_inputs),
                compile_config=edge_compile_config,
            )
            .to_executorch()
            .executorch_program
        )

    @settings(deadline=500000)
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
            export(
                sin_module,
                model_inputs,
            ),
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

    @given(
        unlift=st.booleans(),  # verify both lifted and unlifted graph
    )
    @settings(deadline=500000)
    def test_emit_lowered_backend_module(self, unlift):
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
                export(model, model_inputs), compile_config=edge_compile_config
            )
            lowered_model = to_backend(
                QnnBackend.__name__, edgeir_m.exported_program(), []
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

    @given(
        unlift=st.booleans(),  # verify both lifted and unlifted graph
    )
    @settings(deadline=500000)
    def test_emit_nested_lowered_backend_module(self, unlift):
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
                export(model, model_inputs), compile_config=edge_compile_config
            )
            lowered_module = to_backend(
                QnnBackend.__name__, edgeir_m.exported_program(), []
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
                export(wrapped_module, model_inputs), compile_config=edge_compile_config
            )

            nested_lowered_model = to_backend(
                QnnBackend.__name__, wrapped_module_edge.exported_program(), []
            )

            program = nested_lowered_model.program()
            self.validate_lowered_module_program(program)
