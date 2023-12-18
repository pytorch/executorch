# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Optional

import torch
from executorch import exir
from executorch.backends.xnnpack.passes import XNNPACKPassManager

from executorch.backends.xnnpack.utils.configs import get_xnnpack_capture_config
from executorch.backends.xnnpack.utils.utils import capture_graph_for_xnnpack
from executorch.exir.backend.canonical_partitioners.duplicate_dequant_node_pass import (
    DuplicateDequantNodePass,
)
from executorch.exir.pass_base import ExportPass
from torch.ao.quantization.backend_config.executorch import (
    get_executorch_backend_config,
)
from torch.ao.quantization.qconfig_mapping import _get_symmetric_qnnpack_qconfig_mapping
from torch.ao.quantization.quantize_fx import (
    _convert_to_reference_decomposed_fx,
    prepare_fx,
)

from torch.testing import FileCheck


class TestXNNPackPasses(unittest.TestCase):
    class ReusedInput(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(1, 1, 1)
            self.conv2 = torch.nn.Conv2d(1, 1, 1)

        def forward(self, x):
            return self.conv1(x) + self.conv2(x)

    def capture_and_test_pass(
        self,
        module,
        example_inputs,
        passes,
        expected_copies=None,
        expected_node: str = "executorch_exir_dialects_edge__ops_aten__to_copy_default",
        enable_aot: Optional[bool] = None,
        unlift: Optional[bool] = None,
        atol: float = 1e-08,
        rtol: float = 1e-05,
    ):
        """
        Captures the [module] and runs the given graph [passes]. It checks
        that the number of _to_copy.default ops in the newly created graph is
        equal to the number of [expected_copies]
        """
        exported_program = capture_graph_for_xnnpack(
            module, example_inputs, enable_aot=enable_aot, unlift=unlift
        )

        # Temp hack - to be removed soon
        if all(type(pass_) == type(ExportPass) for pass_ in passes):
            new_exported_program = XNNPACKPassManager(
                exported_program.exported_program, passes
            ).transform()
        else:
            new_exported_program = exported_program.transform(*passes).exported_program

        if expected_copies is not None:
            FileCheck().check_count(
                expected_node,
                expected_copies,
                exactly=True,
            ).run(new_exported_program.graph_module.code)

        old_results = exported_program(*example_inputs)
        new_results = new_exported_program(*example_inputs)

        assert len(old_results) == len(new_results), "Mismatch in length of results!"
        for i in range(len(old_results)):
            self.assertTrue(
                torch.allclose(old_results[i], new_results[i], rtol=rtol, atol=atol)
            )
        return new_exported_program

    # TODO T154127848: Move this out of XNNPACK dir and into cannonical_partitioner dir
    def test_duplicate_dequant_node_pass(self) -> None:
        passes = [DuplicateDequantNodePass()]

        model = self.ReusedInput()

        sample_input = (torch.randn(1, 1, 2, 2),)
        prepared = prepare_fx(
            model,
            _get_symmetric_qnnpack_qconfig_mapping(),
            sample_input,
            backend_config=get_executorch_backend_config(),
        )

        converted = _convert_to_reference_decomposed_fx(
            prepared, backend_config=get_executorch_backend_config()
        )

        prepass_ep = exir.capture(
            converted,
            sample_input,
            get_xnnpack_capture_config(),
        ).to_edge(
            exir.EdgeCompileConfig(
                _check_ir_validity=False,
            )
        )

        FileCheck().check_count(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default",
            6,
            exactly=True,
        ).run(prepass_ep.exported_program.graph_module.code)

        postpass_ep = prepass_ep.transform(*passes)

        duplicated = postpass_ep(sample_input[0])[0]
        non_duplicated = prepass_ep(sample_input[0])[0]

        self.assertTrue(torch.allclose(duplicated, non_duplicated))

        FileCheck().check_count(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default",
            7,
            exactly=True,
        ).run(postpass_ep.exported_program.graph_module.code)
