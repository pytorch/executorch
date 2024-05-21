# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import tempfile
import unittest

import torch
from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.arm_tester import ArmTester

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        self.inputs = (torch.randn(5, 10, 25, in_features),)
        self.fc = torch.nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )

    def get_inputs(self):
        return self.inputs

    def forward(self, x):
        return self.fc(x)


class TestDumpPartitionedArtifact(unittest.TestCase):
    def _tosa_MI_pipeline(self, module: torch.nn.Module, dump_file=None):
        (
            ArmTester(
                module,
                inputs=module.get_inputs(),
                compile_spec=common.get_tosa_compile_spec(),
            )
            .export()
            .to_edge()
            .partition()
            .dump_artifact(dump_file)
            .dump_artifact()
        )

    def _tosa_BI_pipeline(self, module: torch.nn.Module, dump_file=None):
        (
            ArmTester(
                module,
                inputs=module.get_inputs(),
                compile_spec=common.get_tosa_compile_spec(),
            )
            .quantize()
            .export()
            .to_edge()
            .partition()
            .dump_artifact(dump_file)
            .dump_artifact()
        )

    def _is_tosa_marker_in_file(self, tmp_file):
        for line in open(tmp_file).readlines():
            if "'name': 'main'" in line:
                return True
        return False

    def test_MI_artifact(self):
        model = Linear(20, 30)
        tmp_file = os.path.join(tempfile.mkdtemp(), "tosa_dump_MI.txt")
        self._tosa_MI_pipeline(model, dump_file=tmp_file)
        assert os.path.exists(tmp_file), f"File {tmp_file} was not created"
        if self._is_tosa_marker_in_file(tmp_file):
            return  # Implicit pass test
        self.fail("File does not contain TOSA dump!")

    def test_BI_artifact(self):
        model = Linear(20, 30)
        tmp_file = os.path.join(tempfile.mkdtemp(), "tosa_dump_BI.txt")
        self._tosa_BI_pipeline(model, dump_file=tmp_file)
        assert os.path.exists(tmp_file), f"File {tmp_file} was not created"
        if self._is_tosa_marker_in_file(tmp_file):
            return  # Implicit pass test
        self.fail("File does not contain TOSA dump!")


class TestNumericalDiffPrints(unittest.TestCase):
    def test_numerical_diff_prints(self):
        model = Linear(20, 30)
        tester = (
            ArmTester(
                model,
                inputs=model.get_inputs(),
                compile_spec=common.get_tosa_compile_spec(),
            )
            .quantize()
            .export()
            .to_edge()
            .partition()
            .to_executorch()
        )
        if common.TOSA_REF_MODEL_INSTALLED:
            # We expect an assertion error here. Any other issues will cause the
            # test to fail. Likewise the test will fail if the assertion error is
            # not present.
            try:
                # Tolerate 0 difference => we want to trigger a numerical diff
                tester.run_method_and_compare_outputs(atol=0, rtol=0, qtol=0)
            except AssertionError:
                pass  # Implicit pass test
            else:
                self.fail()
        else:
            logger.warning(
                "TOSA ref model tool not installed, skip numerical correctness tests"
            )
