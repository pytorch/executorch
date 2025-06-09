# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os

import tempfile
from datetime import datetime

from pathlib import Path
from typing import Any, Optional

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
        # '::test_collate_tosa_BI_tests[randn] (call)'
        test_name = current_test.split("::")[1].split(" ")[0]  # type: ignore[union-attr]
        if "BI" in test_name:
            tosa_test_base = os.path.join(tosa_test_base, "tosa-bi")
        elif "MI" in test_name:
            tosa_test_base = os.path.join(tosa_test_base, "tosa-mi")
        else:
            tosa_test_base = os.path.join(tosa_test_base, "other")
        return os.path.join(tosa_test_base, test_name)

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
    macs: int = 128,
    system_config: str = "Ethos_U55_High_End_Embedded",
    memory_mode: str = "Shared_Sram",
    extra_flags: str = "--debug-force-regor --output-format=raw",
    custom_path: Optional[str] = None,
) -> list[CompileSpec]:
    """
    Compile spec for Ethos-U55.
    """
    return get_u55_compile_spec_unbuilt(
        macs=macs,
        system_config=system_config,
        memory_mode=memory_mode,
        extra_flags=extra_flags,
        custom_path=custom_path,
    ).build()


def get_u85_compile_spec(
    macs: int = 128,
    system_config="Ethos_U85_SYS_DRAM_Mid",
    memory_mode="Shared_Sram",
    extra_flags="--output-format=raw",
    custom_path=None,
) -> list[CompileSpec]:
    """
    Compile spec for Ethos-U85.
    """
    return get_u85_compile_spec_unbuilt(  # type: ignore[attr-defined]
        macs=macs,
        system_config=system_config,
        memory_mode=memory_mode,
        extra_flags=extra_flags,
        custom_path=custom_path,
    ).build()


def get_u55_compile_spec_unbuilt(
    macs: int,
    system_config: str,
    memory_mode: str,
    extra_flags: str,
    custom_path: Optional[str],
) -> ArmCompileSpecBuilder:
    """Get the ArmCompileSpecBuilder for the Ethos-U55 tests, to modify
    the compile spec before calling .build() to finalize it.
    """
    artifact_path = custom_path or tempfile.mkdtemp(prefix="arm_u55_")
    if not os.path.exists(artifact_path):
        os.makedirs(artifact_path, exist_ok=True)

    # https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela/-/blob/main/OPTIONS.md
    assert macs in [32, 64, 128, 256], "Unsupported MACs value"

    compile_spec = (
        ArmCompileSpecBuilder()
        .ethosu_compile_spec(
            f"ethos-u55-{macs}",
            system_config=system_config,
            memory_mode=memory_mode,
            extra_flags=extra_flags,
        )
        .dump_intermediate_artifacts_to(artifact_path)
    )
    return compile_spec


def get_u85_compile_spec_unbuilt(
    macs: int,
    system_config: str,
    memory_mode: str,
    extra_flags: str,
    custom_path: Optional[str],
) -> list[CompileSpec]:
    """Get the ArmCompileSpecBuilder for the Ethos-U85 tests, to modify
    the compile spec before calling .build() to finalize it.
    """
    artifact_path = custom_path or tempfile.mkdtemp(prefix="arm_u85_")
    if not os.path.exists(artifact_path):
        os.makedirs(artifact_path, exist_ok=True)

    assert macs in [128, 256, 512, 1024, 2048], "Unsupported MACs value"

    compile_spec = (
        ArmCompileSpecBuilder()
        .ethosu_compile_spec(
            f"ethos-u85-{macs}",
            system_config=system_config,
            memory_mode=memory_mode,
            extra_flags=extra_flags,
        )
        .dump_intermediate_artifacts_to(artifact_path)
    )
    return compile_spec  # type: ignore[return-value]


XfailIfNoCorstone300 = pytest.mark.xfail(
    condition=not (
        corstone300_installed() and arm_executor_runner_exists("corstone-300")
    ),
    raises=FileNotFoundError,
    reason="Did not find Corstone-300 FVP or executor_runner on path",
)
"""Xfails a test if Corsone300 FVP is not installed, or if the executor runner is not built"""

XfailIfNoCorstone320 = pytest.mark.xfail(
    condition=not (
        corstone320_installed() and arm_executor_runner_exists("corstone-320")
    ),
    raises=FileNotFoundError,
    reason="Did not find Corstone-320 FVP or executor_runner on path",
)
"""Xfails a test if Corsone320 FVP is not installed, or if the executor runner is not built"""

xfail_type = str | tuple[str, type[Exception]]


def parametrize(
    arg_name: str,
    test_data: dict[str, Any],
    xfails: dict[str, xfail_type] | None = None,
    strict: bool = True,
):
    """
    Custom version of pytest.mark.parametrize with some syntatic sugar and added xfail functionality
        - test_data is expected as a dict of (id, test_data) pairs
        - alllows to specifiy a dict of (id, failure_reason) pairs to mark specific tests as xfail.
          Failure_reason can be str, type[Exception], or tuple[str, type[Exception]].
          Strings set the reason for failure, the exception type sets expected error.
    """
    if xfails is None:
        xfails = {}

    def decorator_func(func):
        """Test data is transformed from a dict of (id, data) pairs to a list of pytest params to work with the native pytests parametrize function"""
        pytest_testsuite = []
        for id, test_parameters in test_data.items():
            if id in xfails:
                xfail_info = xfails[id]
                reason = ""
                raises = None
                if isinstance(xfail_info, str):
                    reason = xfail_info
                elif isinstance(xfail_info, tuple):
                    reason, raises = xfail_info
                else:
                    raise RuntimeError(
                        "xfail info needs to be str, or tuple[str, type[Exception]]"
                    )
                # Set up our fail marker
                marker = (
                    pytest.mark.xfail(reason=reason, raises=raises, strict=strict),
                )
            else:
                marker = ()

            pytest_param = pytest.param(test_parameters, id=id, marks=marker)
            pytest_testsuite.append(pytest_param)
        return pytest.mark.parametrize(arg_name, pytest_testsuite)(func)

    return decorator_func
