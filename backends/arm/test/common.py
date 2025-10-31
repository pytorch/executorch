# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os

import tempfile
from datetime import datetime

from pathlib import Path
from typing import Any, Callable, Optional, ParamSpec, TypeVar

import pytest
from executorch.backends.arm.ethosu import EthosUCompileSpec

from executorch.backends.arm.test.runner_utils import (
    arm_executor_runner_exists,
    corstone300_installed,
    corstone320_installed,
    model_converter_installed,
    vkml_emulation_layer_installed,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.vgf import VgfCompileSpec


def get_time_formatted_path(path: str, log_prefix: str) -> str:
    """
    Returns the log path with the current time appended to it. Used for debugging.

    Args:
        path: The path to the folder where the log file will be stored.
        log_prefix: The name of the test.

    Example output:
        './my_log_folder/test_INT_artifact_28-Nov-14:14:38.log'
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
        # '::test_collate_tosa_INT_tests[randn] (call)'
        test_name = current_test.split("::")[1].split(" ")[0]  # type: ignore[union-attr]
        if "INT" in test_name:
            tosa_test_base = os.path.join(tosa_test_base, "tosa-int")
        elif "FP" in test_name:
            tosa_test_base = os.path.join(tosa_test_base, "tosa-fp")
        else:
            tosa_test_base = os.path.join(tosa_test_base, "other")
        return os.path.join(tosa_test_base, test_name)

    return None


def get_tosa_compile_spec(
    tosa_spec: str | TosaSpecification,
    custom_path=None,
    tosa_debug_mode: TosaCompileSpec.DebugMode | None = None,
) -> TosaCompileSpec:
    """Get the compile spec for default TOSA tests."""
    if not custom_path:
        custom_path = maybe_get_tosa_collate_path()
    if custom_path is not None:
        os.makedirs(custom_path, exist_ok=True)

    compile_spec = (
        TosaCompileSpec(tosa_spec)
        .dump_intermediate_artifacts_to(custom_path)
        .dump_debug_info(tosa_debug_mode)
    )
    return compile_spec


def get_u55_compile_spec(
    macs: int = 128,
    system_config: str = "Ethos_U55_High_End_Embedded",
    memory_mode: str = "Shared_Sram",
    extra_flags: str = "--debug-force-regor --output-format=raw --arena-cache-size=2097152",
    custom_path: Optional[str] = None,
    config: Optional[str] = None,
    tosa_debug_mode: EthosUCompileSpec.DebugMode | None = None,
) -> EthosUCompileSpec:
    """Default compile spec for Ethos-U55 tests."""
    artifact_path = custom_path or tempfile.mkdtemp(prefix="arm_u55_")
    if not os.path.exists(artifact_path):
        os.makedirs(artifact_path, exist_ok=True)

    # https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela/-/blob/main/OPTIONS.md
    assert macs in [32, 64, 128, 256], "Unsupported MACs value"

    if extra_flags is not None:
        extra_flags_list = extra_flags.split(" ")
    else:
        extra_flags_list = []
    compile_spec = (
        EthosUCompileSpec(
            f"ethos-u55-{macs}",
            system_config=system_config,
            memory_mode=memory_mode,
            extra_flags=extra_flags_list,
            config_ini=config,
        )
        .dump_intermediate_artifacts_to(artifact_path)
        .dump_debug_info(tosa_debug_mode)
    )
    return compile_spec


def get_u85_compile_spec(
    macs: int = 128,
    system_config="Ethos_U85_SYS_DRAM_Mid",
    memory_mode="Shared_Sram",
    extra_flags="--output-format=raw --arena-cache-size=2097152",
    custom_path: Optional[str] = None,
    config: Optional[str] = None,
    tosa_debug_mode: EthosUCompileSpec.DebugMode | None = None,
) -> EthosUCompileSpec:
    """Default compile spec for Ethos-U85 tests."""

    artifact_path = custom_path or tempfile.mkdtemp(prefix="arm_u85_")
    if not os.path.exists(artifact_path):
        os.makedirs(artifact_path, exist_ok=True)

    assert macs in [128, 256, 512, 1024, 2048], "Unsupported MACs value"

    if extra_flags is not None:
        extra_flags_list = extra_flags.split(" ")
    else:
        extra_flags_list = []

    compile_spec = (
        EthosUCompileSpec(
            f"ethos-u85-{macs}",
            system_config=system_config,
            memory_mode=memory_mode,
            extra_flags=extra_flags_list,
            config_ini=config,
        )
        .dump_intermediate_artifacts_to(artifact_path)
        .dump_debug_info(tosa_debug_mode)
    )
    return compile_spec  # type: ignore[return-value]


def get_vgf_compile_spec(
    tosa_spec: str | TosaSpecification,
    compiler_flags: Optional[str] = "",
    custom_path=None,
    tosa_debug_mode: VgfCompileSpec.DebugMode | None = None,
) -> VgfCompileSpec:
    """Get the ArmCompileSpec for the default VGF tests, to modify
    the compile spec before calling .build() to finalize it.
    """
    if "FP" in repr(tosa_spec):
        artifact_path = custom_path or tempfile.mkdtemp(prefix="arm_vgf_fp_")
    elif "INT" in repr(tosa_spec):
        artifact_path = custom_path or tempfile.mkdtemp(prefix="arm_vgf_int_")
    else:
        raise ValueError(f"Unsupported vgf compile_spec: {repr(tosa_spec)}")

    if not os.path.exists(artifact_path):
        os.makedirs(artifact_path, exist_ok=True)

    if compiler_flags is not None:
        compiler_flags_list = compiler_flags.split(" ")
    else:
        compiler_flags_list = []

    compile_spec = (
        VgfCompileSpec(tosa_spec, compiler_flags_list)
        .dump_intermediate_artifacts_to(artifact_path)
        .dump_debug_info(tosa_debug_mode)
    )

    return compile_spec


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

SkipIfNoModelConverter = pytest.mark.skipif(  # type: ignore[call-arg]
    condition=not (model_converter_installed()),
    raises=FileNotFoundError,
    reason="Did not find model-converter on path",
)
"""Skips a test if model-converter is not installed"""

XfailfNoVKMLEmulationLayer = pytest.mark.xfail(
    condition=not (vkml_emulation_layer_installed()),
    raises=TypeError,
    reason="VKML environment is not set properly or executor_runner path is misused",
)
"""Xfails a test if VKML Emulation Layer is not installed"""

xfail_type = str | tuple[str, type[Exception]]

_P = ParamSpec("_P")
_R = TypeVar("_R")
Decorator = Callable[[Callable[_P, _R]], Callable[_P, _R]]


def parametrize(
    arg_name: str,
    test_data: dict[str, Any],
    xfails: dict[str, xfail_type] | None = None,
    strict: bool = True,
    flakies: dict[str, int] | None = None,
) -> Decorator:
    """
    Custom version of pytest.mark.parametrize with some syntatic sugar and added xfail functionality
        - test_data is expected as a dict of (id, test_data) pairs
        - alllows to specifiy a dict of (id, failure_reason) pairs to mark specific tests as xfail.
          Failure_reason can be str, type[Exception], or tuple[str, type[Exception]].
          Strings set the reason for failure, the exception type sets expected error.
    """
    if xfails is None:
        xfails = {}
    if flakies is None:
        flakies = {}

    def decorator_func(func: Callable[_P, _R]) -> Callable[_P, _R]:
        """Test data is transformed from a dict of (id, data) pairs to a list of pytest params to work with the native pytests parametrize function"""
        pytest_testsuite = []
        for id, test_parameters in test_data.items():
            if id in flakies:
                # Mark this parameter as flaky with given reruns
                marker = (pytest.mark.flaky(reruns=flakies[id]),)
            elif id in xfails:
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
                marker: tuple[pytest.MarkDecorator, ...]  # type: ignore[no-redef]
                marker = (
                    pytest.mark.xfail(reason=reason, raises=raises, strict=strict),
                )
            else:
                marker = ()  # type: ignore[assignment]

            pytest_param = pytest.param(test_parameters, id=id, marks=marker)
            pytest_testsuite.append(pytest_param)
        decorator = pytest.mark.parametrize(arg_name, pytest_testsuite)
        return decorator(func)

    return decorator_func
