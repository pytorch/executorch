# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.arm.common.arm_compile_spec import ArmCompileSpec
from executorch.backends.arm.common.pipeline_config import (  # noqa: unused
    ArmPassPipelineConfig,
)
from executorch.backends.arm.tosa import (  # type: ignore[import-not-found]
    TosaSpecification,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec


class EthosUCompileSpec(ArmCompileSpec):
    """Compile specification for Ethos-U NPU targets."""

    _TARGET_KEY = "target"

    def __init__(
        self,
        target: str,
        system_config: str | None = None,
        memory_mode: str | None = None,
        extra_flags: list[str] | None = None,
        config_ini: str | None = "Arm/vela.ini",
    ):
        """Normalise Ethos-U compile configuration and compiler flags.

        Args:
            target (str): Ethos-U accelerator configuration (for example,
                ``"ethos-u55-128"``).
            system_config (str | None): System configuration name from the Vela
                config file. Defaults based on ``target`` when omitted.
            memory_mode (str | None): Memory mode selection from the Vela config
                file. Defaults based on ``target`` when omitted.
            extra_flags (list[str] | None): Additional command-line flags for
                Vela.
            config_ini (str | None): Path to a Vela .ini configuration file.
                Defaults to ``"Arm/vela.ini"``.

        """
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
        target_lower = self.target.lower()
        if "ethos-u55" in target_lower:
            if system_config is None:
                system_config = "Ethos_U55_High_End_Embedded"
            if memory_mode is None:
                memory_mode = "Shared_Sram"
        elif "ethos-u85" in target_lower:
            if system_config is None:
                system_config = "Ethos_U85_SYS_DRAM_Mid"
            if memory_mode is None:
                memory_mode = "Sram_Only"
        else:
            raise RuntimeError(f"Unknown ethos target: {target}")

        compiler_flags.append(f"--system-config={system_config}")
        compiler_flags.append(f"--memory-mode={memory_mode}")

        # Set TOSA version.
        base_tosa_version = "TOSA-1.0+INT+int16+int4"
        if "u55" in target_lower:
            # Add the Ethos-U55 extension marker
            base_tosa_version += "+u55"
        if "u85" in self.target:
            base_tosa_version += "+cf"
        tosa_spec = TosaSpecification.create_from_string(base_tosa_version)

        self._set_compile_specs(tosa_spec, compiler_flags)
        self.validate()

    def to_list(self):
        """Return compile specs including the encoded Ethos-U target."""
        compile_specs = super().to_list()
        compile_specs.append(CompileSpec(self._TARGET_KEY, self.target.encode()))
        return compile_specs

    @classmethod
    def from_list_hook(cls, compile_spec, specs: dict[str, str]):
        """Restore target-specific metadata from serialized compile specs."""
        compile_spec.target = specs.get(cls._TARGET_KEY, None)

    def validate(self):
        """Validate the configuration against supported Ethos-U settings."""
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
        """Return the artifact format emitted by this compile spec."""
        return "vela"

    def _create_default_pipeline_config(self) -> ArmPassPipelineConfig:
        # Any u55 subset passes are treated as tosa specification configs
        # As such, they should be added to the base class default.
        return super()._create_default_pipeline_config()
