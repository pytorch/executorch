# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import executorch.exir as exir

import torch
from executorch.backends.fb.qnnpack.partition.qnnpack_partitioner import (
    QnnpackPartitioner,
)
from executorch.backends.fb.qnnpack.qnnpack_preprocess import QnnpackBackend
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackFloatingPointPartitioner,
)

# import the xnnpack backend implementation
from executorch.backends.xnnpack.xnnpack_preprocess import XnnpackBackend

from executorch.exir import CaptureConfig
from executorch.exir.backend.backend_api import to_backend, validation_disabled
from executorch.exir.passes.spec_prop_pass import SpecPropPass

from executorch.extension.pybindings.portable_lib import (  # @manual
    _load_for_executorch_from_buffer,
)
from executorch.extension.pytree import tree_flatten
from torch.ao.quantization.backend_config.executorch import (
    get_executorch_backend_config,
)
from torch.ao.quantization.observer import (
    default_dynamic_quant_observer,
    default_per_channel_weight_observer,
)
from torch.ao.quantization.qconfig_mapping import QConfig, QConfigMapping
from torch.ao.quantization.quantize_fx import (
    _convert_to_reference_decomposed_fx,
    prepare_fx,
)


class TestXnnQnnBackends(unittest.TestCase):
    def test_add_xnnpack_and_dqlinear_qnn(self):
        qconfig_mapping = QConfigMapping().set_object_type(
            torch.nn.Linear,
            QConfig(
                activation=default_dynamic_quant_observer,
                weight=default_per_channel_weight_observer,
            ),
        )
        in_size = 1
        in_features = 3
        out_features = 4

        class LinearAndAdd(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features)

            def forward(self, x, y):
                return self.linear(x) + y

        linear_and_add_mod = LinearAndAdd()

        example_inputs = (
            torch.ones(in_size, in_features, dtype=torch.float),
            torch.ones(in_size, out_features, dtype=torch.float),
        )

        prepared_mod = prepare_fx(
            linear_and_add_mod,
            qconfig_mapping,
            example_inputs,
            backend_config=get_executorch_backend_config(),
        )

        converted_mod: torch.fx.GraphModule = _convert_to_reference_decomposed_fx(
            prepared_mod
        )

        # Step 2: EXIR capturing
        capture_config = CaptureConfig(enable_dynamic_shape=False)
        captured_mod = exir.capture(
            converted_mod, example_inputs, config=capture_config
        ).to_edge(
            exir.EdgeCompileConfig(
                _check_ir_validity=False,
            )
        )

        # Step 3.1: Lower dynamic quant linear to qnnpack
        with validation_disabled():
            module_with_qnnpack_delegate = captured_mod
            module_with_qnnpack_delegate.exported_program = to_backend(
                captured_mod.exported_program, QnnpackPartitioner()
            )

        # Step 3.2: Lower add to xnnpack
        with validation_disabled():
            module_with_xnn_and_qnn = module_with_qnnpack_delegate
            module_with_xnn_and_qnn.exported_program = to_backend(
                module_with_qnnpack_delegate.exported_program,
                XnnpackFloatingPointPartitioner(),
            )

        program_with_delegates = module_with_xnn_and_qnn.to_executorch(
            exir.ExecutorchBackendConfig(passes=[SpecPropPass()]),
        )
        # The first delegate backend is Qnnpack
        self.assertEqual(
            program_with_delegates.program.execution_plan[0].delegates[0].id,
            QnnpackBackend.__name__,
        )
        # The second delegate backend is Xnnpack
        self.assertEqual(
            program_with_delegates.program.execution_plan[0].delegates[1].id,
            XnnpackBackend.__name__,
        )

        executorch_module = _load_for_executorch_from_buffer(
            program_with_delegates.buffer
        )
        inputs_flattened, _ = tree_flatten(example_inputs)

        model_output = executorch_module.run_method("forward", tuple(inputs_flattened))
        ref_output = captured_mod(*example_inputs)

        # Compare the result from executor and eager mode direclty
        self.assertTrue(
            torch.allclose(model_output[0], ref_output, atol=1e-03, rtol=1e-03)
        )
