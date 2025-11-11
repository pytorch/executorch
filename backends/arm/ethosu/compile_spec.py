# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.arm.common.arm_compile_spec import ArmCompileSpec

from executorch.backends.arm.tosa import (  # type: ignore[import-not-found]
    TosaSpecification,
)

from executorch.exir.backend.compile_spec_schema import (  # type: ignore[import-not-found]
    CompileSpec,
)


class EthosUCompileSpec(ArmCompileSpec):
    """
    Compile spec for Ethos-U NPU.

    Args:
        target: Ethos-U accelerator configuration, e.g. ethos-u55-128.
        system_config: System configuration to select from the Vela configuration file.
        memory_mode: Memory mode to select from the Vela configuration file.
        extra_flags: Extra flags for the Vela compiler.
        config_ini: Vela configuration file(s) in Python ConfigParser .ini file format.
    """

    _TARGET_KEY = "target"

    def __init__(
        self,
        target: str,
        system_config: str | None = None,
        memory_mode: str | None = None,
        extra_flags: list[str] | None = None,
        config_ini: str | None = "Arm/vela.ini",
    ):
        self.target = target

        # Set vela compiler flags
        if config_ini is None:
            config_ini = "Arm/vela.ini"
        compiler_flags = [] if extra_flags is None else extra_flags
        compiler_flags.extend(
            [
                f"--accelerator-config={target}",
                f"--config={config_ini}",
                "--output-format=raw",
                "--debug-force-regor",
            ]
        )
        # default system config and memory mode
        if "ethos-u55" in self.target:
            if system_config is None:
                system_config = "Ethos_U55_High_End_Embedded"
            if memory_mode is None:
                memory_mode = "Shared_Sram"
        elif "ethos-u85" in self.target:
            if system_config is None:
                system_config = "Ethos_U85_SYS_DRAM_Mid"
            if memory_mode is None:
                memory_mode = "Sram_Only"
        else:
            raise RuntimeError(f"Unknown ethos target: {self.target}")

        compiler_flags.append(f"--system-config={system_config}")
        compiler_flags.append(f"--memory-mode={memory_mode}")

        # Set TOSA version.
        base_tosa_version = "TOSA-1.0+INT+int16"
        if "u55" in self.target:
            # Add the Ethos-U55 extension marker
            base_tosa_version += "+u55"
        tosa_spec = TosaSpecification.create_from_string(base_tosa_version)

        self._set_compile_specs(tosa_spec, compiler_flags)
        self.validate()

    def to_list(self):
        compile_specs = super().to_list()
        compile_specs.append(CompileSpec(self._TARGET_KEY, self.target.encode()))
        return compile_specs

    @classmethod
    def from_list_hook(cls, compile_spec, specs: dict[str, str]):
        compile_spec.target = specs.get(cls._TARGET_KEY, None)

    def validate(self):
        """Throws an error if the compile spec is not valid."""
        if len(self.compiler_flags) == 0:
            raise ValueError(
                "compile_flags are required in the CompileSpec list for EthosUBackend"
            )
        if "u55" in self.target and not self.tosa_spec.is_U55_subset:
            raise ValueError(
                f"Target was {self.target} but tosa spec was not u55 subset."
            )

    @classmethod
    def get_output_format(cls) -> str:
        return "vela"
