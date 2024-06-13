# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile

from executorch.backends.arm.arm_backend import ArmCompileSpecBuilder


def get_tosa_compile_spec(permute_memory_to_nhwc=False, custom_path=None):
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
        .dump_intermediate_tosa(intermediate_path)
    )

    return compile_spec_builder


def get_u55_compile_spec(permute_memory_to_nhwc=False):
    """
    Default compile spec for Ethos-U55 tests.
    """
    return get_u55_compile_spec_unbuilt(permute_memory_to_nhwc).build()


def get_u55_compile_spec_unbuilt(permute_memory_to_nhwc=False) -> ArmCompileSpecBuilder:
    """Get the ArmCompileSpecBuilder for the default TOSA tests, to modify
    the compile spec before calling .build() to finalize it.
    """
    compile_spec = (
        ArmCompileSpecBuilder()
        .ethosu_compile_spec(
            "ethos-u55-128",
            system_config="Ethos_U55_High_End_Embedded",
            memory_mode="Shared_Sram",
            extra_flags=None,
        )
        .set_permute_memory_format(permute_memory_to_nhwc)
    )
    return compile_spec
