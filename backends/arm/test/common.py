# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import subprocess
import tempfile

import pytest

import torch

from executorch.backends.arm.arm_backend import ArmCompileSpecBuilder

_enabled_options: list[str] = []

# ==== Pytest hooks ====


def pytest_addoption(parser):
    parser.addoption("--arm_quantize_io", action="store_true")
    parser.addoption("--arm_run_corstone300", action="store_true")


def pytest_configure(config):
    if config.option.arm_quantize_io:
        load_libquantized_ops_aot_lib()
        _enabled_options.append("quantize_io")
    if config.option.arm_run_corstone300:
        corstone300_exists = shutil.which("FVP_Corstone_SSE-300_Ethos-U55")
        if not corstone300_exists:
            raise RuntimeError(
                "Tests are run with --arm_run_corstone300 but corstone300 FVP is not installed."
            )
        _enabled_options.append("corstone300")


def pytest_collection_modifyitems(config, items):
    if not config.option.arm_quantize_io:
        skip_if_aot_lib_not_loaded = pytest.mark.skip(
            "u55 tests can only run with quantize_io=True."
        )

        for item in items:
            if "u55" in item.name:
                item.add_marker(skip_if_aot_lib_not_loaded)


# ==== End of Pytest hooks =====


def load_libquantized_ops_aot_lib():
    find_lib_cmd = [
        "find",
        "cmake-out-aot-lib",
        "-name",
        "libquantized_ops_aot_lib.so",
    ]
    res = subprocess.run(find_lib_cmd, capture_output=True)
    if res.returncode == 0:
        library_path = res.stdout.decode().strip()
        torch.ops.load_library(library_path)


def is_option_enabled(option: str, fail_if_not_enabled: bool = False) -> bool:
    """
    Returns whether an option is successfully enabled, i.e. if the flag was
    given to pytest and the necessary requirements are available.
    Implemented options are:
        - corstone300.
        - quantize_io.

    The optional parameter 'fail_if_not_enabled' makes the function raise
      a RuntimeError instead of returning False.
    """
    if option.lower() in _enabled_options:
        return True
    else:
        if fail_if_not_enabled:
            raise RuntimeError(f"Required option '{option}' for test is not enabled")
        else:
            return False


def get_tosa_compile_spec(permute_memory_to_nhwc=True, custom_path=None):
    """
    Default compile spec for TOSA tests.
    """
    return get_tosa_compile_spec_unbuilt(permute_memory_to_nhwc, custom_path).build()


def get_tosa_compile_spec_unbuilt(
    permute_memory_to_nhwc=False, custom_path=None
) -> ArmCompileSpecBuilder:
    """Get the ArmCompileSpecBuilder for the default TOSA tests, to modify
    the compile spec before calling .build() to finalize it.
    """
    intermediate_path = custom_path or tempfile.mkdtemp(prefix="arm_tosa_")
    if not os.path.exists(intermediate_path):
        os.makedirs(intermediate_path, exist_ok=True)
    compile_spec_builder = (
        ArmCompileSpecBuilder()
        .tosa_compile_spec()
        .set_permute_memory_format(permute_memory_to_nhwc)
        .dump_intermediate_artifacts_to(intermediate_path)
    )

    return compile_spec_builder


def get_u55_compile_spec(
    permute_memory_to_nhwc=False, quantize_io=False, custom_path=None
):
    """
    Default compile spec for Ethos-U55 tests.
    """
    return get_u55_compile_spec_unbuilt(
        permute_memory_to_nhwc, quantize_io=quantize_io, custom_path=custom_path
    ).build()


def get_u55_compile_spec_unbuilt(
    permute_memory_to_nhwc=False, quantize_io=False, custom_path=None
) -> ArmCompileSpecBuilder:
    """Get the ArmCompileSpecBuilder for the default TOSA tests, to modify
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
            extra_flags=None,
        )
        .set_quantize_io(is_option_enabled("quantize_io") or quantize_io)
        .set_permute_memory_format(permute_memory_to_nhwc)
        .dump_intermediate_artifacts_to(artifact_path)
    )
    return compile_spec
