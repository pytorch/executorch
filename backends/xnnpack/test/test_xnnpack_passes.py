# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Optional, Tuple

import torch
from executorch import exir
from executorch.backends.xnnpack.passes import XNNPACKPassManager
from executorch.backends.xnnpack.passes.convert_to_linear import ConvertToLinearPass
from executorch.backends.xnnpack.passes.remove_getitem_op import RemoveGetItemPass
from executorch.backends.xnnpack.passes.tag_implicit_q_dq_pass import TagImplicitQDqPass

from executorch.backends.xnnpack.utils.configs import get_xnnpack_capture_config
from executorch.backends.xnnpack.utils.utils import capture_graph_for_xnnpack
from executorch.exir.backend.canonical_partitioners.duplicate_dequant_node_pass import (
    DuplicateDequantNodePass,
)
from executorch.exir.dialects._ops import ops as exir_ops
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

    def test_max_pool2d_remove_getitem(self) -> None:
        passes = [RemoveGetItemPass()]

        class MaxPool2dModule(torch.nn.Module):
            def __init__(
                self,
                kernel_size=3,
                stride=1,
                padding=0,
                dilation=1,
            ):
                super().__init__()
                self.max_pool2d_module = torch.nn.MaxPool2d(
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                )

            def forward(self, x):
                return self.max_pool2d_module(x)

        maxpool2d_module = MaxPool2dModule(3, 1, 0, 1)
        model_inputs = (torch.randn(4, 3, 24, 24),)

        edge_ep = capture_graph_for_xnnpack(maxpool2d_module.eval(), model_inputs)
        new_ep = edge_ep.transform(*passes)
        result1 = edge_ep(model_inputs[0])[0]
        result2 = new_ep(model_inputs[0])[0]

        # Filecheck exir_ops.edge.aten.max_pool2d.default node.
        FileCheck().check_count(
            "executorch_exir_dialects_edge__ops_aten_max_pool2d_default",
            1,
            exactly=True,
        ).run(new_ep.exported_program.graph_module.code)

        self.assertTrue(torch.allclose(result1, result2))

    def test_max_remove_getitem(self) -> None:
        passes = [RemoveGetItemPass()]

        class MaxModule(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()

            def forward(self, x):
                max_vals, indices = torch.max(x, dim=2, keepdim=True)
                return max_vals

        max_module = MaxModule()
        model_inputs = (torch.randn(4, 3, 24, 24),)

        edge_ep = capture_graph_for_xnnpack(max_module.eval(), model_inputs)

        new_ep = edge_ep.transform(*passes)
        result1 = edge_ep(model_inputs[0])[0]
        result2 = new_ep(model_inputs[0])[0]

        # Filecheck exir_ops.edge.aten.amax.default node.
        FileCheck().check_count(
            "executorch_exir_dialects_edge__ops_aten_amax_default", 1, exactly=True
        ).run(new_ep.exported_program.graph_module.code)

        self.assertTrue(torch.allclose(result1, result2))

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

    def test_convert_to_linear(self):
        in_sizes = [1, 4, 4]
        input_sizes = [4, 37, 17]
        output_sizes = [4, 17, 37]
        bias_vals = [True, True, False]

        for enable_aot, unlift in [(False, None), (True, True), (True, False)]:
            for i, _ in enumerate(in_sizes):
                in_size = int(in_sizes[i])
                input_size = int(input_sizes[i])
                output_size = int(output_sizes[i])
                linear = torch.nn.Linear(
                    input_size, output_size, bias=bias_vals[i]
                ).eval()
                example_input = (torch.randn(in_size, input_size),)

                self.capture_and_test_pass(
                    linear,
                    example_input,
                    [ConvertToLinearPass],
                    1,
                    expected_node="executorch_exir_dialects_edge__ops_aten_linear_default",
                    enable_aot=enable_aot,
                    unlift=unlift,
                )

    def test_tag_implicit_q_dq_pass(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default(
                    x, 0.12345, 0, -127, 127, torch.int8
                )
                x = exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default(
                    x, 0.12345, 0, -127, 127, torch.int8
                )
                x = torch.add(x, x)
                x = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default(
                    x, 0.12345, 0, -127, 127, torch.int8
                )
                x = exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default(
                    x, 0.12345, 0, -127, 127, torch.int8
                )
                x = torch.mul(x, x)
                x = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default(
                    x, 0.12345, 0, -127, 127, torch.int8
                )
                x = exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default(
                    x, 0.12345, 0, -127, 127, torch.int8
                )
                x = torch.add(x, x)
                x = torch.mul(x, x)
                return x

        test_model = TestModule()
        test_model.eval()

        sample_inputs = (torch.randn(2, 3),)

        for enable_aot, unlift in [(False, None), (True, True), (True, False)]:
            tag_pass = [TagImplicitQDqPass]
            edge_program = self.capture_and_test_pass(
                test_model,
                sample_inputs,
                tag_pass,
                enable_aot=enable_aot,
                unlift=unlift,
            )
            tagged_graph = edge_program.graph_module.graph

            # The six tagged nodes are:
            # 1) The dq of the first add input
            # 2) The dq of the second add input
            # 3) The q of the add output
            # 4) The dq of the first mul input
            # 5) The dq of the second mul input
            # 6) The q of the mul output
            self.assertEqual(
                sum(
                    node.meta.get(TagImplicitQDqPass.IS_IMPLICIT_Q_DQ_TAG, False)
                    for node in tagged_graph.nodes
                ),
                6,
            )
