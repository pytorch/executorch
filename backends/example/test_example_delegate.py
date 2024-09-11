# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

import torch
from executorch import exir
from executorch.backends.example.example_partitioner import ExamplePartitioner
from executorch.backends.example.example_quantizer import ExampleQuantizer
from executorch.exir import to_edge

from executorch.exir.backend.canonical_partitioners.duplicate_dequant_node_pass import (
    DuplicateDequantNodePass,
)
from executorch.exir.delegate import executorch_call_delegate

from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.export import export

from torchvision.models.quantization import mobilenet_v2


class TestExampleDelegate(unittest.TestCase):
    def test_delegate_linear(self):
        class Conv2dModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv2d = torch.nn.Conv2d(16, 33, 3)

            def forward(self, arg):
                return self.conv2d(arg)

            @staticmethod
            def get_example_inputs():
                return (torch.randn(20, 16, 50, 100),)

        model = Conv2dModule()
        example_inputs = Conv2dModule.get_example_inputs()
        EDGE_COMPILE_CONFIG = exir.EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,  # TODO(T182928844): Delegate dim order op to backend.
        )

        m = model.eval()
        m = torch.export.export_for_training(m, copy.deepcopy(example_inputs)).module()
        # print("original model:", m)
        quantizer = ExampleQuantizer()
        # quantizer = XNNPACKQuantizer()
        # if we set is_per_channel to True, we also need to add out_variant of quantize_per_channel/dequantize_per_channel
        # operator_config = get_symmetric_quantization_config(is_per_channel=False)
        # quantizer.set_global(operator_config)
        m = prepare_pt2e(m, quantizer)
        # calibration
        m(*example_inputs)
        m = convert_pt2e(m)

        quantized_gm = m
        exported_program = to_edge(
            export(quantized_gm, copy.deepcopy(example_inputs)),
            compile_config=EDGE_COMPILE_CONFIG,
        )

        lowered_export_program = exported_program.to_backend(
            ExamplePartitioner(),
        )

        print("After lowering to qnn backend: ")
        lowered_export_program.exported_program().graph.print_tabular()

    def test_delegate_mobilenet_v2(self):
        model = mobilenet_v2(num_classes=3)
        model.eval()
        example_inputs = (torch.rand(1, 3, 320, 240),)

        EDGE_COMPILE_CONFIG = exir.EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,  # TODO(T182928844): Delegate dim order op to backend.
        )

        m = model.eval()
        m = torch.export.export_for_training(m, copy.deepcopy(example_inputs)).module()
        quantizer = ExampleQuantizer()

        m = prepare_pt2e(m, quantizer)
        # calibration
        m(*example_inputs)
        m = convert_pt2e(m)

        quantized_gm = m
        exported_program = to_edge(
            export(quantized_gm, copy.deepcopy(example_inputs)),
            compile_config=EDGE_COMPILE_CONFIG,
        )

        lowered_export_program = exported_program.transform(
            [DuplicateDequantNodePass()]
        ).to_backend(
            ExamplePartitioner(),
        )

        lowered_export_program.exported_program().graph.print_tabular()

        call_deleage_node = [
            node
            for node in lowered_export_program.exported_program().graph.nodes
            if node.target == executorch_call_delegate
        ]
        self.assertEqual(len(call_deleage_node), 1)
