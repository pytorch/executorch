# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.exir import to_edge
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.test.backend_with_compiler_demo import (
    BackendWithCompilerDemo,
)

from executorch.extension.pybindings.aten_lib import (  # @manual
    _load_for_executorch_from_buffer,
)

from executorch.extension.pytree import tree_flatten
from torch.export import export


class TestDelegateAtenMode(unittest.TestCase):
    def test_add_xnnpack_and_dqlinear_qnn(self):
        class AddMulModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, x, b):
                y = torch.mm(a, x)
                z = torch.add(y, b)
                return z

        add_mul_module = AddMulModule()
        model_inputs = (torch.ones(2, 2), 2 * torch.ones(2, 2), 3 * torch.ones(2, 2))
        edge_graph_module = to_edge(export(add_mul_module, model_inputs))
        max_value = model_inputs[0].shape[0]
        compile_specs = [CompileSpec("max_value", bytes([max_value]))]
        lowered_add_mul = to_backend(
            BackendWithCompilerDemo.__name__,
            edge_graph_module.exported_program(),
            compile_specs,
        )

        class CompositeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lowered_add_mul = lowered_add_mul

            def forward(self, a, x, b):
                return self.lowered_add_mul(a, x, b)

        composite_model = CompositeModule()

        composite_model(*model_inputs)

        exec_prog = to_edge(export(composite_model, model_inputs)).to_executorch()

        buff = exec_prog.buffer

        executorch_module = _load_for_executorch_from_buffer(buff)
        inputs_flattened, _ = tree_flatten(model_inputs)
        model_output = executorch_module.run_method("forward", tuple(inputs_flattened))
        ref_output = add_mul_module(*model_inputs)

        self.assertTrue(
            torch.allclose(model_output[0], ref_output, atol=1e-03, rtol=1e-03)
        )
