# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import shutil
import tempfile

from executorch.backends.arm.arm_backend import ArmCompileSpecBuilder

# TODO: fixme! These globs are a temporary workaround. Reasoning:
# Running the jobs in _unittest.yml will not work since that environment doesn't
# have the vela tool, nor the tosa_reference_model tool. Hence, we need a way to
# run what we can in that env temporarily. Long term, vela and tosa_reference_model
# should be installed in the CI env.
TOSA_REF_MODEL_INSTALLED = shutil.which("tosa_reference_model")
VELA_INSTALLED = shutil.which("vela")


def get_tosa_compile_spec(permute_memory_to_nhwc=False, custom_path=None):
    """
    Default compile spec for TOSA tests.
    """
    intermediate_path = custom_path or tempfile.mkdtemp(prefix="arm_tosa_")
    compile_spec = (
        ArmCompileSpecBuilder()
        .tosa_compile_spec()
        .set_permute_memory_format(permute_memory_to_nhwc)
        .dump_intermediate_tosa(intermediate_path)
        .build()
    )
    return compile_spec


def get_u55_compile_spec(permute_memory_to_nhwc=False):
    """
    Default compile spec for Ethos-U55 tests.
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
        .build()
    )
    return compile_spec
