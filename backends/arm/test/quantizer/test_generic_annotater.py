# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import unittest

import torch
from executorch.backends.arm.quantizer.arm_quantizer_utils import is_annotated
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions


class SingleOpModel(torch.nn.Module):
    def __init__(self, op, example_input, **op_kwargs) -> None:
        super().__init__()
        self.op = op
        self._example_input = example_input
        self.op_kwargs = op_kwargs

    def forward(self, x):
        return self.op(x, **self.op_kwargs)

    def example_inputs(self):
        return self._example_input


class TestGenericAnnotator(unittest.TestCase):
    def check_annotation(self, model):
        tester = ArmTester(
            model,
            model.example_inputs(),
            common.get_tosa_compile_spec("TOSA-0.80.0+BI"),
        )
        quant_model = tester.quantize().get_artifact()
        partitions = get_source_partitions(quant_model.graph, [model.op])
        partitions = list(itertools.chain.from_iterable(partitions.values()))

        assert len(partitions) == 1
        partition = partitions[0]
        assert all(is_annotated(node) for node in partition.nodes)

    def test_squeeze(self):
        self.check_annotation(SingleOpModel(torch.squeeze, (torch.rand(8, 8, 1),)))
        self.check_annotation(SingleOpModel(torch.squeeze_copy, (torch.rand(8, 8, 1),)))

    def test_unsqueeze(self):
        self.check_annotation(
            SingleOpModel(torch.unsqueeze, (torch.rand(8, 8),), dim=0)
        )
        self.check_annotation(
            SingleOpModel(torch.unsqueeze_copy, (torch.rand(8, 8),), dim=0)
        )

    def test_reshape(self):
        self.check_annotation(
            SingleOpModel(torch.reshape, (torch.randn(8, 8),), shape=(64,)),
        )

    def test_view(self):
        self.check_annotation(
            SingleOpModel(torch.view_copy, (torch.randn(4, 4),), size=(2, 8)),
        )

    def test_slice(self):
        self.check_annotation(
            SingleOpModel(torch.slice_copy, (torch.randn(3, 4),)),
        )

    def test_transpose(self):
        self.check_annotation(
            SingleOpModel(torch.transpose, (torch.randn(2, 3),), dim0=0, dim1=1),
        )
        self.check_annotation(
            SingleOpModel(torch.transpose_copy, (torch.randn(2, 3),), dim0=0, dim1=1),
        )

    def test_tile(self):
        self.check_annotation(
            SingleOpModel(torch.tile, (torch.randn(4, 4),), dims=(2,)),
        )

    def test_flip(self):
        self.check_annotation(
            SingleOpModel(torch.flip, (torch.randn(2, 4),), dims=(0, 1)),
        )

    def test_concat(self):
        self.check_annotation(
            SingleOpModel(
                torch.concatenate, ((torch.randn(2, 3), torch.randn(2, 3)),), dim=0
            ),
        )
