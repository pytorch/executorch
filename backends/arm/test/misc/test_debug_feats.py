# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import shutil
import tempfile

from pathlib import Path
from typing import Tuple

import pytest

import torch
from executorch.backends.arm.common.arm_compile_spec import ArmCompileSpec
from executorch.backends.arm.test import common
from executorch.backends.arm.test.runner_utils import dbg_tosa_fb_to_json
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
)
from executorch.backends.test.harness.stages import StageType

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


def _tosa_FP_pipeline(module: torch.nn.Module, test_data: input_t1, dump_file=None):
    aten_ops: list[str] = []
    exir_ops: list[str] = []
    pipeline = TosaPipelineFP[input_t1](module, test_data, aten_ops, exir_ops)
    pipeline.dump_artifact("to_edge_transform_and_lower")
    pipeline.dump_artifact("to_edge_transform_and_lower", suffix=dump_file)
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()


def _tosa_INT_pipeline(module: torch.nn.Module, test_data: input_t1, dump_file=None):
    aten_ops: list[str] = []
    exir_ops: list[str] = []
    pipeline = TosaPipelineINT[input_t1](module, test_data, aten_ops, exir_ops)
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
def test_FP_artifact(test_data: input_t1):
    model = Linear()
    tmp_file = common.get_time_formatted_path(
        tempfile.mkdtemp(), test_FP_artifact.__name__
    )
    _tosa_FP_pipeline(model, test_data, dump_file=tmp_file)
    assert os.path.exists(tmp_file), f"File {tmp_file} was not created"
    if _is_tosa_marker_in_file(tmp_file):
        return  # Implicit pass test
    pytest.fail("File does not contain TOSA dump!")


@common.parametrize("test_data", Linear.inputs)
def test_INT_artifact(test_data: input_t1):
    model = Linear()
    tmp_file = common.get_time_formatted_path(
        tempfile.mkdtemp(), test_INT_artifact.__name__
    )
    _tosa_INT_pipeline(model, test_data, dump_file=tmp_file)
    assert os.path.exists(tmp_file), f"File {tmp_file} was not created"
    if _is_tosa_marker_in_file(tmp_file):
        return  # Implicit pass test
    pytest.fail("File does not contain TOSA dump!")


"""Tests trigging the exception printout from the ArmTester's run and compare function."""


@common.parametrize("test_data", Linear.inputs)
def test_numerical_diff_print(test_data: input_t1):
    aten_ops: list[str] = []
    exir_ops: list[str] = []
    pipeline = TosaPipelineINT[input_t1](
        Linear(),
        test_data,
        aten_ops,
        exir_ops,
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
        tester.run_method_and_compare_outputs(
            stage=StageType.INITIAL_MODEL, atol=0, rtol=0, qtol=0
        )
    except AssertionError:
        pass  # Implicit pass test
    else:
        pytest.fail()


@common.parametrize("test_data", Linear.inputs)
def test_dump_ops_and_dtypes(test_data: input_t1):
    aten_ops: list[str] = []
    exir_ops: list[str] = []
    pipeline = TosaPipelineINT[input_t1](Linear(), test_data, aten_ops, exir_ops)
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
    aten_ops: list[str] = []
    exir_ops: list[str] = []
    pipeline = TosaPipelineINT[input_t1](Linear(), test_data, aten_ops, exir_ops)
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
def test_collate_tosa_INT_tests(test_data: input_t1):
    # Set the environment variable to trigger the collation of TOSA tests
    os.environ["TOSA_TESTCASES_BASE_PATH"] = "test_collate_tosa_tests"
    # Clear out the directory
    aten_ops: list[str] = []
    exir_ops: list[str] = []
    pipeline = TosaPipelineINT[input_t1](Linear(), test_data, aten_ops, exir_ops)
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()

    test_collate_dir = (
        "test_collate_tosa_tests/tosa-int/test_collate_tosa_INT_tests[randn]"
    )
    # test that the output directory is created and contains the expected files
    assert os.path.exists(test_collate_dir)
    for file in os.listdir(test_collate_dir):
        file_name_prefix = "TOSA-1.0+INT"
        assert file.endswith((f"{file_name_prefix}.json", f"{file_name_prefix}.tosa"))

    os.environ.pop("TOSA_TESTCASES_BASE_PATH")
    shutil.rmtree("test_collate_tosa_tests", ignore_errors=True)


@common.parametrize("test_data", Linear.inputs)
def test_dump_tosa_debug_json(test_data: input_t1):
    with tempfile.TemporaryDirectory() as tmpdir:
        aten_ops: list[str] = []
        exir_ops: list[str] = []
        pipeline = TosaPipelineINT[input_t1](
            module=Linear(),
            test_data=test_data,
            aten_op=aten_ops,
            exir_op=exir_ops,
            custom_path=tmpdir,
            tosa_debug_mode=ArmCompileSpec.DebugMode.JSON,
        )

        pipeline.pop_stage("run_method_and_compare_outputs")
        pipeline.run()

        json_output_path = Path(tmpdir) / "debug.json"

        # The file should exist
        assert json_output_path.exists()

        # Check the file is valid JSON and can be loaded
        with json_output_path.open("r") as file:
            try:
                data = json.load(file)

                # Check it's not empty
                assert data
            except json.JSONDecodeError:
                pytest.fail("Failed to load debug JSON file")


@common.parametrize("test_data", Linear.inputs)
def test_dump_tosa_debug_tosa(test_data: input_t1):
    output_dir = "test_dump_tosa_debug"

    aten_ops: list[str] = []
    exir_ops: list[str] = []
    pipeline = TosaPipelineFP[input_t1](
        module=Linear(),
        test_data=test_data,
        use_to_edge_transform_and_lower=True,
        aten_op=aten_ops,
        exir_op=exir_ops,
        custom_path=output_dir,
        tosa_debug_mode=ArmCompileSpec.DebugMode.TOSA,
    )

    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()

    output_path = Path(output_dir)
    json_output_path = output_path / "debug.json"

    # A JSON file should not be created when TOSA mode used
    assert not json_output_path.exists()

    # At least one TOSA file should exist
    tosa_files = list(output_path.glob("*.tosa"))
    assert len(tosa_files) > 0

    tosa_file = tosa_files[0]
    with tosa_file.open("rb") as f:
        tosa_json = dbg_tosa_fb_to_json(f.read())

    # Check all non-empty JSON strings are valid
    ops = tosa_json["regions"][0]["blocks"][0]["operators"]
    for op in ops:
        if op["location"]["text"]:
            try:
                json.loads(op["location"]["text"])
            except json.JSONDecodeError:
                pytest.fail("Failed to load debug JSON string")

    shutil.rmtree(output_dir, ignore_errors=True)


@common.parametrize("test_data", Linear.inputs)
def test_dump_tosa_ops(capsys, test_data: input_t1):
    aten_ops: list[str] = []
    exir_ops: list[str] = []
    pipeline = TosaPipelineINT[input_t1](Linear(), test_data, aten_ops, exir_ops)
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.dump_operator_distribution("to_edge_transform_and_lower")
    pipeline.run()
    assert "TOSA operators:" in capsys.readouterr().out


class Add(torch.nn.Module):
    inputs = {
        "ones": (torch.ones(5),),
    }

    def forward(self, x):
        return x + x


@common.parametrize("test_data", Add.inputs)
@common.XfailIfNoCorstone300
def test_fail_dump_tosa_ops(capsys, test_data: input_t1):
    aten_ops: list[str] = []
    exir_ops: list[str] = []
    pipeline = EthosU55PipelineINT[input_t1](
        Add(), test_data, aten_ops, exir_ops, use_to_edge_transform_and_lower=True
    )
    pipeline.dump_operator_distribution("to_edge_transform_and_lower")
    error_msg = "Can not get operator distribution for Vela command stream."
    with pytest.raises(NotImplementedError, match=error_msg):
        pipeline.run()
