# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.exir import to_edge
from executorch.exir._serialize import _serialize_pte_binary
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.test.backend_with_compiler_demo import (
    BackendWithCompilerDemo,
)

from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,  # @manual
)
from torch.export import export


class TestCompatibility(unittest.TestCase):
    def test_compatibility_in_runtime(self):
        class SinModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.sin(x)

        sin_module = SinModule()
        model_inputs = (torch.ones(1),)
        edgeir_m = to_edge(export(sin_module, model_inputs))
        max_value = model_inputs[0].shape[0]
        compile_specs = [CompileSpec("max_value", bytes([max_value]))]
        lowered_sin_module = to_backend(
            BackendWithCompilerDemo.__name__, edgeir_m.exported_program(), compile_specs
        )
        buff = lowered_sin_module.buffer()

        # The demo backend works well
        executorch_module = _load_for_executorch_from_buffer(buff)
        model_inputs = torch.ones(1)
        _ = executorch_module.forward([model_inputs])

        prog = lowered_sin_module.program()
        # Rewrite the delegate version number from 0 to 1.
        prog.backend_delegate_data[0].data = bytes(
            "1version:1#op:demo::aten.sin.default, numel:1, dtype:torch.float32<debug_handle>1#",
            encoding="utf8",
        )

        # Generate the .pte file with the wrong version.
        buff = bytes(
            _serialize_pte_binary(
                program=prog,
            )
        )

        # Throw runtime error with error code 0x30, meaning delegate is incompatible.
        with self.assertRaisesRegex(
            RuntimeError,
            "loading method forward failed with error 0x30",
        ):
            executorch_module = _load_for_executorch_from_buffer(buff)
