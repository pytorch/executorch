# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import tempfile

from typing import Tuple

import pytest

import torch
from executorch.backends.arm.test import common, conftest
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
)


input_t1 = Tuple[torch.Tensor]  # Input x


class Linear(torch.nn.Module):
    inputs = {
        "randn": (torch.randn(5, 10, 25, 3),),
    }

    def __init__(
        self,
    ):
        super().__init__()
        self.fc = torch.nn.Linear(
            in_features=3,
            out_features=5,
            bias=True,
        )

    def forward(self, x):
        return self.fc(x)


"""Tests dumping the partition artifact in ArmTester. Both to file and to stdout."""


def _tosa_MI_pipeline(module: torch.nn.Module, test_data: input_t1, dump_file=None):

    pipeline = TosaPipelineMI[input_t1](module, test_data, [], [])
    pipeline.dump_artifact("to_edge_transform_and_lower")
    pipeline.dump_artifact("to_edge_transform_and_lower", suffix=dump_file)
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()


def _tosa_BI_pipeline(module: torch.nn.Module, test_data: input_t1, dump_file=None):

    pipeline = TosaPipelineBI[input_t1](module, test_data, [], [])
    pipeline.dump_artifact("to_edge_transform_and_lower")
    pipeline.dump_artifact("to_edge_transform_and_lower", suffix=dump_file)
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()


def _is_tosa_marker_in_file(tmp_file):
    for line in open(tmp_file).readlines():
        if "'name': 'main'" in line:
            return True
    return False


@common.parametrize("test_data", Linear.inputs)
def test_MI_artifact(test_data: input_t1):
    model = Linear()
    tmp_file = common.get_time_formatted_path(
        tempfile.mkdtemp(), test_MI_artifact.__name__
    )
    _tosa_MI_pipeline(model, test_data, dump_file=tmp_file)
    assert os.path.exists(tmp_file), f"File {tmp_file} was not created"
    if _is_tosa_marker_in_file(tmp_file):
        return  # Implicit pass test
    pytest.fail("File does not contain TOSA dump!")


@common.parametrize("test_data", Linear.inputs)
def test_BI_artifact(test_data: input_t1):
    model = Linear()
    tmp_file = common.get_time_formatted_path(
        tempfile.mkdtemp(), test_BI_artifact.__name__
    )
    _tosa_BI_pipeline(model, test_data, dump_file=tmp_file)
    assert os.path.exists(tmp_file), f"File {tmp_file} was not created"
    if _is_tosa_marker_in_file(tmp_file):
        return  # Implicit pass test
    pytest.fail("File does not contain TOSA dump!")


"""Tests trigging the exception printout from the ArmTester's run and compare function."""


@common.parametrize("test_data", Linear.inputs)
def test_numerical_diff_print(test_data: input_t1):
    pipeline = TosaPipelineMI[input_t1](
        Linear(),
        test_data,
        [],
        [],
        custom_path="diff_print_test",
    )
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()
    tester = pipeline.tester
    # We expect an assertion error here. Any other issues will cause the
    # test to fail. Likewise the test will fail if the assertion error is
    # not present.
    try:
        # Tolerate 0 difference => we want to trigger a numerical diff
        tester.run_method_and_compare_outputs(atol=0, rtol=0, qtol=0)
    except AssertionError:
        pass  # Implicit pass test
    else:
        pytest.fail()


@common.parametrize("test_data", Linear.inputs)
def test_dump_ops_and_dtypes(test_data: input_t1):
    pipeline = TosaPipelineBI[input_t1](Linear(), test_data, [], [])
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.add_stage_after("quantize", pipeline.tester.dump_dtype_distribution)
    pipeline.add_stage_after("quantize", pipeline.tester.dump_operator_distribution)
    pipeline.add_stage_after("export", pipeline.tester.dump_dtype_distribution)
    pipeline.add_stage_after("export", pipeline.tester.dump_operator_distribution)
    pipeline.add_stage_after(
        "to_edge_transform_and_lower", pipeline.tester.dump_dtype_distribution
    )
    pipeline.add_stage_after(
        "to_edge_transform_and_lower", pipeline.tester.dump_operator_distribution
    )
    pipeline.run()
    # Just test that there are no execptions.


@common.parametrize("test_data", Linear.inputs)
def test_dump_ops_and_dtypes_parseable(test_data: input_t1):
    pipeline = TosaPipelineBI[input_t1](Linear(), test_data, [], [])
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.add_stage_after("quantize", pipeline.tester.dump_dtype_distribution, False)
    pipeline.add_stage_after(
        "quantize", pipeline.tester.dump_operator_distribution, False
    )
    pipeline.add_stage_after("export", pipeline.tester.dump_dtype_distribution, False)
    pipeline.add_stage_after(
        "export", pipeline.tester.dump_operator_distribution, False
    )
    pipeline.add_stage_after(
        "to_edge_transform_and_lower", pipeline.tester.dump_dtype_distribution, False
    )
    pipeline.add_stage_after(
        "to_edge_transform_and_lower", pipeline.tester.dump_operator_distribution, False
    )
    pipeline.run()
    # Just test that there are no execptions.


"""Tests the collation of TOSA tests through setting the environment variable TOSA_TESTCASE_BASE_PATH."""


@common.parametrize("test_data", Linear.inputs)
def test_collate_tosa_BI_tests(test_data: input_t1):
    # Set the environment variable to trigger the collation of TOSA tests
    os.environ["TOSA_TESTCASES_BASE_PATH"] = "test_collate_tosa_tests"
    # Clear out the directory
    pipeline = TosaPipelineBI[input_t1](Linear(), test_data, [], [])
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()

    test_collate_dir = (
        "test_collate_tosa_tests/tosa-bi/test_collate_tosa_BI_tests[randn]"
    )
    # test that the output directory is created and contains the expected files
    assert os.path.exists(test_collate_dir)
    tosa_version = conftest.get_option("tosa_version")
    for file in os.listdir(test_collate_dir):
        file_name_prefix = f"TOSA-{tosa_version}+" + (
            "INT" if tosa_version == "1.0" else "BI"
        )
        assert file.endswith((f"{file_name_prefix}.json", f"{file_name_prefix}.tosa"))

    os.environ.pop("TOSA_TESTCASES_BASE_PATH")
    shutil.rmtree("test_collate_tosa_tests", ignore_errors=True)


@common.parametrize("test_data", Linear.inputs)
def test_dump_tosa_ops(caplog, test_data: input_t1):
    pipeline = TosaPipelineBI[input_t1](Linear(), test_data, [], [])
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.dump_operator_distribution("to_edge_transform_and_lower")
    pipeline.run()
    assert "TOSA operators:" in caplog.text


class Add(torch.nn.Module):
    inputs = {
        "ones": (torch.ones(5),),
    }

    def forward(self, x):
        return x + x


@common.parametrize("test_data", Add.inputs)
def test_fail_dump_tosa_ops(caplog, test_data: input_t1):
    pipeline = EthosU55PipelineBI[input_t1](
        Add(), test_data, [], [], use_to_edge_transform_and_lower=True, run_on_fvp=False
    )
    pipeline.dump_operator_distribution("to_edge_transform_and_lower")
    pipeline.run()
    assert "Can not get operator distribution for Vela command stream." in caplog.text
