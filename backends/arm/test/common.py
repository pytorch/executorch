# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os

import tempfile
from datetime import datetime

from pathlib import Path
from typing import Any

import pytest
from executorch.backends.arm.arm_backend import ArmCompileSpecBuilder
from executorch.backends.arm.test.runner_utils import (
    arm_executor_runner_exists,
    corstone300_installed,
    corstone320_installed,
)
from executorch.backends.arm.tosa_specification import TosaSpecification
from executorch.exir.backend.compile_spec_schema import CompileSpec


def get_time_formatted_path(path: str, log_prefix: str) -> str:
    """
    Returns the log path with the current time appended to it. Used for debugging.

    Args:
        path: The path to the folder where the log file will be stored.
        log_prefix: The name of the test.

    Example output:
        './my_log_folder/test_BI_artifact_28-Nov-14:14:38.log'
    """
    return str(
        Path(path) / f"{log_prefix}_{datetime.now().strftime('%d-%b-%H:%M:%S')}.log"
    )


def maybe_get_tosa_collate_path() -> str | None:
    """
    Checks the environment variable TOSA_TESTCASES_BASE_PATH and returns the
    path to the where to store the current tests if it is set.
    """
    tosa_test_base = os.environ.get("TOSA_TESTCASES_BASE_PATH")
    if tosa_test_base:
        current_test = os.environ.get("PYTEST_CURRENT_TEST")
        #'backends/arm/test/ops/test_mean_dim.py::TestMeanDim::test_meandim_tosa_BI_0_zeros (call)'
        test_class = current_test.split("::")[1]  # type: ignore[union-attr]
        test_name = current_test.split("::")[-1].split(" ")[0]  # type: ignore[union-attr]
        if "BI" in test_name:
            tosa_test_base = os.path.join(tosa_test_base, "tosa-bi")
        elif "MI" in test_name:
            tosa_test_base = os.path.join(tosa_test_base, "tosa-mi")
        else:
            tosa_test_base = os.path.join(tosa_test_base, "other")
        return os.path.join(tosa_test_base, test_class, test_name)

    return None


def get_tosa_compile_spec(
    tosa_spec: str | TosaSpecification, custom_path=None
) -> list[CompileSpec]:
    """
    Default compile spec for TOSA tests.
    """
    return get_tosa_compile_spec_unbuilt(tosa_spec, custom_path).build()


def get_tosa_compile_spec_unbuilt(
    tosa_spec: str | TosaSpecification, custom_path=None
) -> ArmCompileSpecBuilder:
    """Get the ArmCompileSpecBuilder for the default TOSA tests, to modify
    the compile spec before calling .build() to finalize it.
    """
    if not custom_path:
        custom_path = maybe_get_tosa_collate_path()

    if custom_path is not None:
        os.makedirs(custom_path, exist_ok=True)
    compile_spec_builder = (
        ArmCompileSpecBuilder()
        .tosa_compile_spec(tosa_spec)
        .dump_intermediate_artifacts_to(custom_path)
    )

    return compile_spec_builder


def get_u55_compile_spec(
    custom_path=None,
) -> list[CompileSpec]:
    """
    Default compile spec for Ethos-U55 tests.
    """
    return get_u55_compile_spec_unbuilt(
        custom_path=custom_path,
    ).build()


def get_u85_compile_spec(
    custom_path=None,
) -> list[CompileSpec]:
    """
    Default compile spec for Ethos-U85 tests.
    """
    return get_u85_compile_spec_unbuilt(  # type: ignore[attr-defined]
        custom_path=custom_path,
    ).build()


def get_u55_compile_spec_unbuilt(
    custom_path=None,
) -> ArmCompileSpecBuilder:
    """Get the ArmCompileSpecBuilder for the Ethos-U55 tests, to modify
    the compile spec before calling .build() to finalize it.
    """
    artifact_path = custom_path or tempfile.mkdtemp(prefix="arm_u55_")
    if not os.path.exists(artifact_path):
        os.makedirs(artifact_path, exist_ok=True)
    compile_spec = (
        ArmCompileSpecBuilder()
        .ethosu_compile_spec(
            "ethos-u55-128",
            system_config="Ethos_U55_High_End_Embedded",
            memory_mode="Shared_Sram",
            extra_flags="--debug-force-regor --output-format=raw",
        )
        .dump_intermediate_artifacts_to(artifact_path)
    )
    return compile_spec


def get_u85_compile_spec_unbuilt(
    custom_path=None,
) -> list[CompileSpec]:
    """Get the ArmCompileSpecBuilder for the Ethos-U85 tests, to modify
    the compile spec before calling .build() to finalize it.
    """
    artifact_path = custom_path or tempfile.mkdtemp(prefix="arm_u85_")
    compile_spec = (
        ArmCompileSpecBuilder()
        .ethosu_compile_spec(
            "ethos-u85-128",
            system_config="Ethos_U85_SYS_DRAM_Mid",
            memory_mode="Shared_Sram",
            extra_flags="--output-format=raw",
        )
        .dump_intermediate_artifacts_to(artifact_path)
    )
    return compile_spec  # type: ignore[return-value]


SkipIfNoCorstone300 = pytest.mark.skipif(
    not corstone300_installed() or not arm_executor_runner_exists("corstone-300"),
    reason="Did not find Corstone-300 FVP or executor_runner on path",
)
"""Skips a test if Corsone300 FVP is not installed, or if the executor runner is not built"""

SkipIfNoCorstone320 = pytest.mark.skipif(
    not corstone320_installed() or not arm_executor_runner_exists("corstone-320"),
    reason="Did not find Corstone-320 FVP or executor_runner on path",
)
"""Skips a test if Corsone320 FVP is not installed, or if the executor runner is not built."""


def parametrize(
    arg_name: str, test_data: dict[str, Any], xfails: dict[str, str] = None
):
    """
    Custom version of pytest.mark.parametrize with some syntatic sugar and added xfail functionality
        - test_data is expected as a dict of (id, test_data) pairs
        - alllows to specifiy a dict of (id, failure_reason) pairs to mark specific tests as xfail
    """
    if xfails is None:
        xfails = {}

    def decorator_func(func):
        """Test data is transformed from a dict of (id, data) pairs to a list of pytest params to work with the native pytests parametrize function"""
        pytest_testsuite = []
        for id, test_parameters in test_data.items():
            if id in xfails:
                pytest_param = pytest.param(
                    test_parameters, id=id, marks=pytest.mark.xfail(reason=xfails[id])
                )
            else:
                pytest_param = pytest.param(test_parameters, id=id)
            pytest_testsuite.append(pytest_param)

        return pytest.mark.parametrize(arg_name, pytest_testsuite)(func)

    return decorator_func
