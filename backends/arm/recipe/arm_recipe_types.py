# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from executorch.export import RecipeType  # type: ignore[import-untyped]


class ArmRecipeType(RecipeType):
    """Arm-specific recipe types"""

    TOSA_FP = "arm_tosa_fp"
    """ Kwargs:
            - custom_path: str=None,
            - tosa_debug_mode: TosaCompileSpec.DebugMode | None = None,
    """
    TOSA_INT8_STATIC_PER_TENSOR = "arm_tosa_int8_static_per_tensor"
    """ Kwargs:
            - custom_path: str=None,
            - tosa_debug_mode: TosaCompileSpec.DebugMode | None = None,
    """
    TOSA_INT8_STATIC_PER_CHANNEL = "arm_tosa_int8_static_per_channel"
    """ Kwargs:
            - custom_path: str=None,
            - tosa_debug_mode: TosaCompileSpec.DebugMode | None = None,
    """
    ETHOSU_U55_INT8_STATIC_PER_CHANNEL = "arm_ethosu_u55_int_static_per_channel"
    """ Kwargs:
            - macs: int = 128,
            - system_config: str = "Ethos_U55_High_End_Embedded",
            - memory_mode: str = "Shared_Sram",
            - extra_flags: str = "--debug-force-regor --output-format=raw",
            - custom_path: Optional[str] = None,
            - config: Optional[str] = None,
            - tosa_debug_mode: EthosUCompileSpec.DebugMode | None = None,
    """
    ETHOSU_U55_INT8_STATIC_PER_TENSOR = "arm_ethosu_u55_int_static_per_channel"
    """ Kwargs:
            - macs: int = 128,
            - system_config: str = "Ethos_U55_High_End_Embedded",
            - memory_mode: str = "Shared_Sram",
            - extra_flags: str = "--debug-force-regor --output-format=raw",
            - custom_path: Optional[str] = None,
            - config: Optional[str] = None,
            - tosa_debug_mode: EthosUCompileSpec.DebugMode | None = None,
    """
    ETHOSU_U85_INT8_STATIC_PER_TENSOR = "arm_ethosu_u85_int_static_per_tensor"
    """ Kwargs:
            - macs: int = 128,
            - system_config="Ethos_U85_SYS_DRAM_Mid",
            - memory_mode="Shared_Sram",
            - extra_flags="--output-format=raw",
            - custom_path: Optional[str] = None,
            - config: Optional[str] = None,
            - tosa_debug_mode: EthosUCompileSpec.DebugMode | None = None,
    """
    ETHOSU_U85_INT8_STATIC_PER_CHANNEL = "arm_ethosu_u85_int_static_per_channel"
    """ Kwargs:
            - macs: int = 128,
            - system_config="Ethos_U85_SYS_DRAM_Mid",
            - memory_mode="Shared_Sram",
            - extra_flags="--output-format=raw",
            - custom_path: Optional[str] = None,
            - config: Optional[str] = None,
            - tosa_debug_mode: EthosUCompileSpec.DebugMode | None = None,
    """

    VGF_FP = "arm_vgf_fp"
    """ Kwargs:
            - compiler_flags: Optional[str] = "",
            - custom_path=None,
            - tosa_debug_mode: VgfCompileSpec.DebugMode | None = None,
    """
    VGF_INT8_STATIC_PER_TENSOR = "arm_vgf_int8_static_per_tensor"
    """ Kwargs:
            - compiler_flags: Optional[str] = "",
            - custom_path=None,
            - tosa_debug_mode: VgfCompileSpec.DebugMode | None = None,
    """
    VGF_INT8_STATIC_PER_CHANNEL = "arm_vgf_int8_static_per_channel"
    """ Kwargs:
            - compiler_flags: Optional[str] = "",
            - custom_path=None,
            - tosa_debug_mode: VgfCompileSpec.DebugMode | None = None,
    """
    CUSTOM = "Provide your own ArmRecipeType to the kwarg 'recipe'."

    @classmethod
    def get_backend_name(cls) -> str:
        return "arm"
