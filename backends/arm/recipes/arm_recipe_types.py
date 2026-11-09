# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from executorch.export import RecipeType


ARM_BACKEND: str = "arm"


class ArmRecipeType(RecipeType):
    """Arm-specific recipe types.

    Covers the delegated targets of ``backends/arm/scripts/aot_arm_compiler.py``.
    Its non-delegated Cortex-M/CMSIS-NN path is a separate backend and is not
    reachable from these recipes.

    Ethos-U recipes accept the following kwargs:
        macs (int): MAC count for the family, validated against the accelerator
            configurations the installed Vela accepts -- today 32/64/128/256 for
            U55, 256/512 for U65 and 128/256/512/1024/2048 for U85. Defaults to
            128 for U55 and 256 for U65 and U85.
        system_config (str): Vela system config name. Defaults from
            ``EthosUCompileSpec`` apply when omitted.
        memory_mode (str): Vela memory mode. Defaults from
            ``EthosUCompileSpec`` apply when omitted.
        extra_flags (list[str]): Vela compiler flags, appended to the
            ``--verbose-operators --verbose-cycle-estimate`` the CLI always
            passes rather than replacing them.
        config_ini (str): Path to a Vela .ini configuration file. Defaults to
            ``"Arm/vela.ini"``.

    """

    ETHOS_U55_INT8 = "arm_ethos_u55_int8"
    ETHOS_U65_INT8 = "arm_ethos_u65_int8"
    ETHOS_U85_INT8 = "arm_ethos_u85_int8"

    TOSA_FP = "arm_tosa_fp"
    TOSA_INT8 = "arm_tosa_int8"
    TOSA_A16W8 = "arm_tosa_a16w8"

    VGF_FP = "arm_vgf_fp"
    VGF_INT8 = "arm_vgf_int8"

    @classmethod
    def get_backend_name(cls) -> str:
        return ARM_BACKEND
