# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os

import tempfile
from datetime import datetime
from pathlib import Path

from executorch.backends.arm.arm_backend import ArmCompileSpecBuilder

from executorch.backends.arm.test.conftest import is_option_enabled
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
        custom_path = maybe_get_tosa_collate_path()

    if custom_path is not None:
        os.makedirs(custom_path, exist_ok=True)
    compile_spec_builder = (
        ArmCompileSpecBuilder()
        .tosa_compile_spec(tosa_version)
        .set_permute_memory_format(permute_memory_to_nhwc)
        .dump_intermediate_artifacts_to(custom_path)
    )

    return compile_spec_builder


def get_u55_compile_spec(
    permute_memory_to_nhwc=True,
    quantize_io=False,
    custom_path=None,
    reorder_inputs=None,
) -> list[CompileSpec]:
    """
    Default compile spec for Ethos-U55 tests.
    """
    return get_u55_compile_spec_unbuilt(
        permute_memory_to_nhwc,
        quantize_io=quantize_io,
        custom_path=custom_path,
        reorder_inputs=reorder_inputs,
    ).build()


def get_u85_compile_spec(
    permute_memory_to_nhwc=True,
    quantize_io=False,
    custom_path=None,
    reorder_inputs=None,
) -> list[CompileSpec]:
    """
    Default compile spec for Ethos-U85 tests.
    """
    return get_u85_compile_spec_unbuilt(
        permute_memory_to_nhwc,
        quantize_io=quantize_io,
        custom_path=custom_path,
        reorder_inputs=reorder_inputs,
    ).build()


def get_u55_compile_spec_unbuilt(
    permute_memory_to_nhwc=True,
    quantize_io=False,
    custom_path=None,
    reorder_inputs=None,
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
        .set_input_order(reorder_inputs)
    )
    return compile_spec


def get_u85_compile_spec_unbuilt(
    permute_memory_to_nhwc=True,
    quantize_io=False,
    custom_path=None,
    reorder_inputs=None,
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
        .set_input_order(reorder_inputs)
    )
    return compile_spec


def get_target_board(compile_spec: list[CompileSpec]) -> str | None:
    for spec in compile_spec:
        if spec.key == "compile_flags":
            flags = spec.value.decode()
            if "u55" in flags:
                return "corstone-300"
            elif "u85" in flags:
                return "corstone-320"
    return None
