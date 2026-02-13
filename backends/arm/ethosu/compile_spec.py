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

    _TARGET_KEY = "target"

    @staticmethod
    def _default_system_config_and_memory_mode(
        target_lower: str,
        system_config: str | None,
        memory_mode: str | None,
    ) -> tuple[str, str]:
        if "ethos-u55" in target_lower:
            resolved_system_config = (
                "Ethos_U55_High_End_Embedded"
                if system_config is None
                else system_config
            )
            resolved_memory_mode = "Shared_Sram" if memory_mode is None else memory_mode
            return resolved_system_config, resolved_memory_mode
        if "ethos-u65" in target_lower:
            resolved_system_config = (
                "Ethos_U65_SYS_DRAM_Mid" if system_config is None else system_config
            )
            resolved_memory_mode = "Sram_Only" if memory_mode is None else memory_mode
            return resolved_system_config, resolved_memory_mode
        if "ethos-u85" in target_lower:
            resolved_system_config = (
                "Ethos_U85_SYS_DRAM_Mid" if system_config is None else system_config
            )
            resolved_memory_mode = "Sram_Only" if memory_mode is None else memory_mode
            return resolved_system_config, resolved_memory_mode
        raise RuntimeError(f"Unknown ethos target: {target_lower}")

    @staticmethod
    def _build_compiler_flags(
        *,
        target: str,
        config_ini: str,
        extra_flags: list[str] | None,
        system_config: str,
        memory_mode: str,
    ) -> list[str]:
        compiler_flags = [] if extra_flags is None else list(extra_flags)
        compiler_flags.extend(
            [
                f"--accelerator-config={target}",
                f"--config={config_ini}",
                "--output-format=raw",
                "--debug-force-regor",
                f"--system-config={system_config}",
                f"--memory-mode={memory_mode}",
            ]
        )
        return compiler_flags

    @staticmethod
    def _tosa_spec_for_target(target_lower: str) -> TosaSpecification:
        base_tosa_version = "TOSA-1.0+INT+int16+int4"
        if "u55" in target_lower:
            base_tosa_version += "+u55"
        if "u65" in target_lower:
            base_tosa_version += "+u55"
        if "u85" in target_lower:
            base_tosa_version += "+cf"
        return TosaSpecification.create_from_string(base_tosa_version)

    def __init__(
        self,
        target: str,
        system_config: str | None = None,
        memory_mode: str | None = None,
        extra_flags: list[str] | None = None,
        config_ini: str | None = "Arm/vela.ini",
    ):
        self.target = target
        target_lower = self.target.lower()
        resolved_config_ini = "Arm/vela.ini" if config_ini is None else config_ini
        resolved_system_config, resolved_memory_mode = (
            self._default_system_config_and_memory_mode(
                target_lower=target_lower,
                system_config=system_config,
                memory_mode=memory_mode,
            )
        )
        compiler_flags = self._build_compiler_flags(
            target=self.target,
            config_ini=resolved_config_ini,
            extra_flags=extra_flags,
            system_config=resolved_system_config,
            memory_mode=resolved_memory_mode,
        )
        tosa_spec = self._tosa_spec_for_target(target_lower)
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
        if "u55" in self.target.lower() and not self.tosa_spec.is_U55_subset:
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
