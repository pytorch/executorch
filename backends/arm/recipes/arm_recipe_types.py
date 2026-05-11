# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.export import RecipeType


ARM_BACKEND = "arm"


class ArmRecipeType(RecipeType):
    """Arm-specific recipe types.

    Coverage matches ``backends/arm/scripts/aot_arm_compiler.py`` today
    (Cortex-M is not yet supported via recipes).

    Ethos-U recipes accept the following kwargs:
        macs (int): MAC count for the family.
            U55: 32 / 64 / 128 / 256 (default 128).
            U65: 256 / 512 (default 256).
            U85: 128 / 256 / 512 / 1024 / 2048 (default 256).
        system_config (str): Vela system config name. Defaults from
            ``EthosUCompileSpec`` apply when omitted.
        memory_mode (str): Vela memory mode. Defaults from
            ``EthosUCompileSpec`` apply when omitted.
        extra_flags (list[str]): Additional Vela compiler flags.
        config_ini (str): Path to a Vela .ini configuration file.

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
