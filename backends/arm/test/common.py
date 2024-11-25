# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from enum import auto, Enum
from pathlib import Path
from typing import Any

import pytest

import torch

from executorch.backends.arm.arm_backend import ArmCompileSpecBuilder
from executorch.exir.backend.compile_spec_schema import CompileSpec


class arm_test_options(Enum):
    quantize_io = auto()
    corstone300 = auto()
    dump_path = auto()
    date_format = auto()
    fast_fvp = auto()


_test_options: dict[arm_test_options, Any] = {}

# ==== Pytest hooks ====


def pytest_addoption(parser):
    parser.addoption("--arm_quantize_io", action="store_true")
    parser.addoption("--arm_run_corstone300", action="store_true")
    parser.addoption("--default_dump_path", default=None)
    parser.addoption("--date_format", default="%d-%b-%H:%M:%S")
    parser.addoption("--fast_fvp", action="store_true")


def pytest_configure(config):
    if config.option.arm_quantize_io:
        load_libquantized_ops_aot_lib()
        _test_options[arm_test_options.quantize_io] = True
    if config.option.arm_run_corstone300:
        corstone300_exists = shutil.which("FVP_Corstone_SSE-300_Ethos-U55")
        if not corstone300_exists:
            raise RuntimeError(
                "Tests are run with --arm_run_corstone300 but corstone300 FVP is not installed."
            )
        _test_options[arm_test_options.corstone300] = True
    if config.option.default_dump_path:
        dump_path = Path(config.option.default_dump_path).expanduser()
        if dump_path.exists() and os.path.isdir(dump_path):
            _test_options[arm_test_options.dump_path] = dump_path
        else:
            raise RuntimeError(
                f"Supplied argument 'default_dump_path={dump_path}' that does not exist or is not a directory."
            )
    _test_options[arm_test_options.date_format] = config.option.date_format
    _test_options[arm_test_options.fast_fvp] = config.option.fast_fvp
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)


def pytest_collection_modifyitems(config, items):
    if not config.option.arm_quantize_io:
        skip_if_aot_lib_not_loaded = pytest.mark.skip(
            "u55 tests can only run with quantize_io=True."
        )

        for item in items:
            if "u55" in item.name:
                item.add_marker(skip_if_aot_lib_not_loaded)


def pytest_sessionstart(session):
    pass


def pytest_sessionfinish(session, exitstatus):
    if get_option(arm_test_options.dump_path):
        _clean_dir(
            get_option(arm_test_options.dump_path),
            f"ArmTester_{get_option(arm_test_options.date_format)}.log",
        )


# ==== End of Pytest hooks =====

# ==== Custom Pytest decorators =====


def expectedFailureOnFVP(test_item):
    if is_option_enabled("corstone300"):
        test_item.__unittest_expecting_failure__ = True
    return test_item


# ==== End of Custom Pytest decorators =====


def load_libquantized_ops_aot_lib():
    so_ext = {
        "Darwin": "dylib",
        "Linux": "so",
        "Windows": "dll",
    }.get(platform.system(), None)

    find_lib_cmd = [
        "find",
        "cmake-out-aot-lib",
        "-name",
        f"libquantized_ops_aot_lib.{so_ext}",
    ]
    res = subprocess.run(find_lib_cmd, capture_output=True)
    if res.returncode == 0:
        library_path = res.stdout.decode().strip()
        torch.ops.load_library(library_path)


def is_option_enabled(
    option: str | arm_test_options, fail_if_not_enabled: bool = False
) -> bool:
    """
    Returns whether an option is successfully enabled, i.e. if the flag was
    given to pytest and the necessary requirements are available.
    Implemented options are:
        - corstone300.
        - quantize_io.

    The optional parameter 'fail_if_not_enabled' makes the function raise
      a RuntimeError instead of returning False.
    """
    if isinstance(option, str):
        option = arm_test_options[option.lower()]

    if option in _test_options and _test_options[option]:
        return True
    else:
        if fail_if_not_enabled:
            raise RuntimeError(f"Required option '{option}' for test is not enabled")
        else:
            return False


def get_option(option: arm_test_options) -> Any | None:
    if option in _test_options:
        return _test_options[option]
    return None


def maybe_get_tosa_collate_path() -> str | None:
    """
    Checks the environment variable TOSA_TESTCASES_BASE_PATH and returns the
    path to the where to store the current tests if it is set.
    """
    tosa_test_base = os.environ.get("TOSA_TESTCASES_BASE_PATH")
    if tosa_test_base:
        current_test = os.environ.get("PYTEST_CURRENT_TEST")
        #'backends/arm/test/ops/test_mean_dim.py::TestMeanDim::test_meandim_tosa_BI_0_zeros (call)'
        test_class = current_test.split("::")[1]
        test_name = current_test.split("::")[-1].split(" ")[0]
        if "BI" in test_name:
            tosa_test_base = os.path.join(tosa_test_base, "tosa-bi")
        elif "MI" in test_name:
            tosa_test_base = os.path.join(tosa_test_base, "tosa-mi")
        else:
            tosa_test_base = os.path.join(tosa_test_base, "other")

        return os.path.join(tosa_test_base, test_class, test_name)

    return None


def get_tosa_compile_spec(
    tosa_version: str, permute_memory_to_nhwc=True, custom_path=None
) -> list[CompileSpec]:
    """
    Default compile spec for TOSA tests.
    """
    return get_tosa_compile_spec_unbuilt(
        tosa_version, permute_memory_to_nhwc, custom_path
    ).build()


def get_tosa_compile_spec_unbuilt(
    tosa_version: str, permute_memory_to_nhwc=False, custom_path=None
) -> ArmCompileSpecBuilder:
    """Get the ArmCompileSpecBuilder for the default TOSA tests, to modify
    the compile spec before calling .build() to finalize it.
    """
    if not custom_path:
        intermediate_path = maybe_get_tosa_collate_path() or tempfile.mkdtemp(
            prefix="arm_tosa_"
        )
    else:
        intermediate_path = custom_path

    if not os.path.exists(intermediate_path):
        os.makedirs(intermediate_path, exist_ok=True)
    compile_spec_builder = (
        ArmCompileSpecBuilder()
        .tosa_compile_spec(tosa_version)
        .set_permute_memory_format(permute_memory_to_nhwc)
        .dump_intermediate_artifacts_to(intermediate_path)
    )

    return compile_spec_builder


def get_u55_compile_spec(
    permute_memory_to_nhwc=True, quantize_io=False, custom_path=None
) -> list[CompileSpec]:
    """
    Default compile spec for Ethos-U55 tests.
    """
    return get_u55_compile_spec_unbuilt(
        permute_memory_to_nhwc, quantize_io=quantize_io, custom_path=custom_path
    ).build()


def get_u85_compile_spec(
    permute_memory_to_nhwc=True, quantize_io=False, custom_path=None
) -> list[CompileSpec]:
    """
    Default compile spec for Ethos-U85 tests.
    """
    return get_u85_compile_spec_unbuilt(
        permute_memory_to_nhwc, quantize_io=quantize_io, custom_path=custom_path
    ).build()


def get_u55_compile_spec_unbuilt(
    permute_memory_to_nhwc=True, quantize_io=False, custom_path=None
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
        .set_quantize_io(is_option_enabled("quantize_io") or quantize_io)
        .set_permute_memory_format(permute_memory_to_nhwc)
        .dump_intermediate_artifacts_to(artifact_path)
    )
    return compile_spec


def get_u85_compile_spec_unbuilt(
    permute_memory_to_nhwc=True, quantize_io=False, custom_path=None
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
        .set_quantize_io(is_option_enabled("quantize_io") or quantize_io)
        .set_permute_memory_format(permute_memory_to_nhwc)
        .dump_intermediate_artifacts_to(artifact_path)
    )
    return compile_spec


def current_time_formated() -> str:
    """Return current time as a formated string"""
    return datetime.now().strftime(get_option(arm_test_options.date_format))


def _clean_dir(dir: Path, filter: str, num_save=10):
    sorted_files: list[tuple[datetime, Path]] = []
    for file in dir.iterdir():
        try:
            creation_time = datetime.strptime(file.name, filter)
            insert_index = -1
            for i, to_compare in enumerate(sorted_files):
                compare_time = to_compare[0]
                if creation_time < compare_time:
                    insert_index = i
                    break
            if insert_index == -1 and len(sorted_files) < num_save:
                sorted_files.append((creation_time, file))
            else:
                sorted_files.insert(insert_index, (creation_time, file))
        except ValueError:
            continue

    if len(sorted_files) > num_save:
        for remove in sorted_files[0 : len(sorted_files) - num_save]:
            file = remove[1]
            file.unlink()
