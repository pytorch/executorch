# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

import os
import platform
import warnings
from collections.abc import Mapping
from datetime import datetime
from importlib import import_module, metadata
from pathlib import Path
from typing import Any, Callable, Optional, ParamSpec, TypeVar

import pytest
from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.test.runner_utils import (
    arm_executor_runner_exists,
    corstone300_installed,
    corstone300_u65_installed,
    corstone320_installed,
    model_converter_installed,
    vkml_emulation_layer_installed,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.vgf import VgfCompileSpec
from executorch.backends.arm.vgf.model_converter import (
    get_model_converter_version_text,
    parse_model_converter_version,
)
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version


def is_aarch64_host() -> bool:
    return platform.machine().lower() in ("aarch64", "arm64")


def get_time_formatted_path(path: str, log_prefix: str) -> str:
    """Returns the log path with the current time appended to it. Used for
    debugging.

    Args:
        path: The path to the folder where the log file will be stored.
        log_prefix: The name of the test.

    Example output:
        './my_log_folder/test_INT_artifact_28-Nov-14:14:38.log'

    """
    return str(
        Path(path) / f"{log_prefix}_{datetime.now().strftime('%d-%b-%H:%M:%S')}.log"
    )


def maybe_get_tosa_artifact_path() -> str | None:
    """Return the configured artifact directory for the current test."""
    artifact_base_path = getattr(pytest, "_test_options", {}).get("dump_artifacts")
    if artifact_base_path:
        current_test = os.environ.get("PYTEST_CURRENT_TEST")
        if current_test is None:
            raise RuntimeError("Could not determine the current pytest test name")
        test_name = (
            current_test.split(" (")[0]
            .rsplit("::", 1)[-1]
            .replace(",", "_")
            .replace(" ", "")
        )
        return os.path.join(artifact_base_path, test_name)

    return maybe_get_tosa_collate_path()


def maybe_get_tosa_collate_path() -> str | None:
    """Return the current test's TOSA collation directory, when configured."""
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
    custom_path: Optional[str] = None,
    tosa_debug_mode: TosaCompileSpec.DebugMode | None = None,
) -> TosaCompileSpec:
    """Get the compile spec for default TOSA tests."""
    if not custom_path:
        custom_path = maybe_get_tosa_artifact_path()
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
    extra_flags: str = "--arena-cache-size=2097152",
    custom_path: Optional[str] = None,
    config: Optional[str] = None,
    tosa_debug_mode: EthosUCompileSpec.DebugMode | None = None,
) -> EthosUCompileSpec:
    """Default compile spec for Ethos-U55 tests."""
    if not custom_path:
        custom_path = maybe_get_tosa_artifact_path()
    if custom_path is not None:
        os.makedirs(custom_path, exist_ok=True)

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
        .dump_intermediate_artifacts_to(custom_path)
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
    if not custom_path:
        custom_path = maybe_get_tosa_artifact_path()
    if custom_path is not None:
        os.makedirs(custom_path, exist_ok=True)

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
        .dump_intermediate_artifacts_to(custom_path)
        .dump_debug_info(tosa_debug_mode)
    )
    return compile_spec  # type: ignore[return-value]


def get_u65_compile_spec(
    macs: int = 256,
    system_config: str = "Ethos_U65_High_End",
    memory_mode: str = "Dedicated_Sram_384KB",
    extra_flags: str = "--arena-cache-size=393216",
    custom_path: Optional[str] = None,
    config: Optional[str] = None,
    tosa_debug_mode: EthosUCompileSpec.DebugMode | None = None,
) -> EthosUCompileSpec:
    """Default compile spec for Ethos-U65 tests."""
    if not custom_path:
        custom_path = maybe_get_tosa_artifact_path()
    if custom_path is not None:
        os.makedirs(custom_path, exist_ok=True)

    assert macs in [256, 512], "Unsupported MACs value"

    if extra_flags is not None:
        extra_flags_list = extra_flags.split(" ")
    else:
        extra_flags_list = []

    compile_spec = (
        EthosUCompileSpec(
            f"ethos-u65-{macs}",
            system_config=system_config,
            memory_mode=memory_mode,
            extra_flags=extra_flags_list,
            config_ini=config,
        )
        .dump_intermediate_artifacts_to(custom_path)
        .dump_debug_info(tosa_debug_mode)
    )
    return compile_spec


def get_vgf_compile_spec(
    tosa_spec: str | TosaSpecification,
    compiler_flags: Optional[str] = "",
    custom_path: Optional[str] = None,
    tosa_debug_mode: VgfCompileSpec.DebugMode | None = None,
    preserve_io_quantization: bool = False,
) -> VgfCompileSpec:
    """Get the ArmCompileSpec for the default VGF tests, to modify the compile
    spec before calling .build() to finalize it.
    """
    if not custom_path:
        custom_path = maybe_get_tosa_artifact_path()
    if custom_path is not None:
        os.makedirs(custom_path, exist_ok=True)

    profiles = []
    if "FP" in repr(tosa_spec):
        profiles.append("fp")
    if "INT" in repr(tosa_spec):
        profiles.append("int")
    if len(profiles) == 0:
        raise ValueError(f"Unsupported vgf compile_spec: {repr(tosa_spec)}")
    if compiler_flags is not None:
        compiler_flags_list = compiler_flags.split(" ")
    else:
        compiler_flags_list = []

    compile_spec = (
        VgfCompileSpec(tosa_spec, compiler_flags_list)
        .dump_intermediate_artifacts_to(custom_path)
        .dump_debug_info(tosa_debug_mode)
    )

    if preserve_io_quantization:
        compile_spec._set_preserve_io_quantization(True)

    return compile_spec


XfailIfNoCorstone300 = pytest.mark.xfail(
    condition=not (
        corstone300_installed() and arm_executor_runner_exists("corstone-300")
    ),
    raises=FileNotFoundError,
    reason="Did not find Corstone-300 FVP or executor_runner on path",
)
"""Xfails a test if Corsone300 FVP is not installed, or if the executor runner
is not built.
"""


XfailIfNoCorstone300_u65 = pytest.mark.xfail(
    condition=not (
        corstone300_u65_installed() and arm_executor_runner_exists("corstone-300-u65")
    ),
    raises=FileNotFoundError,
    reason="Did not find Corstone-300-u65 FVP or executor_runner on path",
)
"""Xfails a test if Corsone300-u65 FVP is not installed, or if the executor
runner is not built.
"""


XfailIfNoCorstone320 = pytest.mark.xfail(
    condition=not (
        corstone320_installed() and arm_executor_runner_exists("corstone-320")
    ),
    raises=FileNotFoundError,
    reason="Did not find Corstone-320 FVP or executor_runner on path",
)
"""Xfails a test if Corsone320 FVP is not installed, or if the executor runner
is not built.
"""

SkipIfNoModelConverter = pytest.mark.skipif(  # type: ignore[call-arg]
    condition=not (model_converter_installed()),
    raises=FileNotFoundError,
    reason="Did not find model-converter on path",
)
"""Skips a test if model-converter is not installed."""

XfailfNoVKMLEmulationLayer = pytest.mark.xfail(
    condition=not (vkml_emulation_layer_installed()),
    raises=TypeError,
    reason="VKML environment is not set properly or executor_runner path is misused",
)
"""Xfails a test if VKML Emulation Layer is not installed."""


def xfail_if_version(
    version: str | Version | None,
    specifier: str,
    *,
    reason: str,
    strict: bool = True,
    raises: type[Exception] | None = None,
) -> pytest.MarkDecorator:
    """Xfail when a dependency version matches a PEP 440 specifier.

    Use as a test decorator or as a value in `parametrize`'s `xfails` map.
    Missing/unknown versions do not enable the xfail. Unparseable versions
    produce a warning; missing dependencies need a separate skip marker.

    """
    versions = SpecifierSet(specifier)
    try:
        matches = version is not None and version in versions
    except InvalidVersion:
        warnings.warn(
            f"Unparseable dependency version {version!r}; xfail for "
            f"{specifier!r} is disabled ({reason}).",
            RuntimeWarning,
            stacklevel=2,
        )
        matches = False

    return pytest.mark.xfail(
        condition=matches,
        reason=reason,
        strict=strict,
        raises=raises,
    )


def xfail_if_dependency_version(
    dependency: str,
    specifier: str,
    *,
    reason: str,
    module: str | None = None,
    strict: bool = True,
    raises: type[Exception] | None = None,
) -> pytest.MarkDecorator:
    """Xfail for matching versions of an installed Python distribution.

    If distribution metadata is absent, optionally read `module.__version__`.
    For example, Vela uses dependency="ethos-u-vela", module="ethosu.vela".
    Missing dependencies do not enable the xfail; use a separate skip marker.

    """
    try:
        version = metadata.version(dependency)
    except metadata.PackageNotFoundError:
        version = None
        if module is not None:
            try:
                version = getattr(import_module(module), "__version__", None)
            except ModuleNotFoundError as exc:
                if exc.name != module and not module.startswith(f"{exc.name}."):
                    raise

    return xfail_if_version(
        version, specifier, reason=reason, strict=strict, raises=raises
    )


def xfail_if_model_converter_version(
    specifier: str,
    *,
    reason: str,
    strict: bool = True,
    raises: type[Exception] | None = None,
) -> pytest.MarkDecorator:
    """Xfail for matching versions of the converter executable used by VGF."""
    version_text = get_model_converter_version_text()
    version = (
        parse_model_converter_version(version_text)
        if version_text is not None
        else None
    )
    return xfail_if_version(
        version, specifier, reason=reason, strict=strict, raises=raises
    )


xfail_type = str | tuple[str, type[Exception]] | pytest.MarkDecorator

_P = ParamSpec("_P")
_R = TypeVar("_R")
Decorator = Callable[[Callable[_P, _R]], Callable[_P, _R]]


def parametrize(
    arg_name: str,
    test_data: dict[str, Any],
    xfails: Mapping[str, xfail_type] | None = None,
    skips: dict[str, str] | None = None,
    strict: bool = True,
    flakies: dict[str, int] | None = None,
) -> Decorator:
    """Custom version of pytest.mark.parametrize with some syntatic sugar and
    added xfail functionality.

    - test_data is expected as a dict of (id, test_data) pairs
    - alllows to specifiy a dict of (id, failure_reason) pairs to mark specific tests as xfail.
      Failure_reason can be str, tuple[str, type[Exception]], or an xfail marker.
      Strings set the reason for failure, the exception type sets expected error.
      Explicit xfail markers retain their own options, including strictness.

    """
    xfail_cases = xfails or {}
    skip_cases = skips or {}
    flaky_cases = flakies or {}

    def decorator_func(func: Callable[_P, _R]) -> Callable[_P, _R]:
        """Test data is transformed from a dict of (id, data) pairs to a list of
        pytest params to work with the native pytests parametrize function.
        """
        pytest_testsuite = []
        for id, test_parameters in test_data.items():
            if id in flaky_cases:
                # Mark this parameter as flaky with given reruns
                marker = (pytest.mark.flaky(reruns=flaky_cases[id]),)
            elif id in skip_cases:
                # fail markers do not work with 'buck' based ci, so use skip instead
                marker = (pytest.mark.skip(reason=skip_cases[id]),)
            elif id in xfail_cases:
                xfail_info = xfail_cases[id]
                reason = ""
                raises = None
                if isinstance(xfail_info, str):
                    reason = xfail_info
                elif isinstance(xfail_info, tuple):
                    reason, raises = xfail_info
                elif isinstance(xfail_info, pytest.MarkDecorator):
                    if xfail_info.name != "xfail":
                        raise ValueError("xfails only accepts xfail markers")
                else:
                    raise RuntimeError(
                        "xfail info needs to be str, tuple[str, type[Exception]], "
                        "or an xfail marker"
                    )
                # Set up our fail marker
                marker: tuple[pytest.MarkDecorator, ...]  # type: ignore[no-redef]
                xfail_marker = (
                    xfail_info
                    if isinstance(xfail_info, pytest.MarkDecorator)
                    else pytest.mark.xfail(reason=reason, raises=raises, strict=strict)
                )
                marker = (xfail_marker,)
            else:
                marker = ()  # type: ignore[assignment]

            pytest_param = pytest.param(test_parameters, id=id, marks=marker)
            pytest_testsuite.append(pytest_param)
        decorator = pytest.mark.parametrize(arg_name, pytest_testsuite)
        return decorator(func)

    return decorator_func
