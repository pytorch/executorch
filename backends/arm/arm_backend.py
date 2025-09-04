# Copyright 2023-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

#
# Main implementation of AoT flow to partition and preprocess for Arm target
# backends. Converts via TOSA as an intermediate form supported by AoT and
# JIT compiler flows.
#
from enum import Enum
from typing import List, Optional

from executorch.backends.arm.tosa_specification import (  # type: ignore[import-not-found]
    TosaSpecification,
)

from executorch.exir.backend.compile_spec_schema import (  # type: ignore[import-not-found]
    CompileSpec,
)


class ArmCompileSpecBuilder:
    class DebugMode(Enum):
        JSON = 1

    def __init__(self):
        self.compile_spec: List[CompileSpec] = []
        self.compiler_flags = []
        self.output_format = None
        self.path_for_intermediates = None
        self.tosa_spec = None
        self.tosa_debug_mode = None

    def vgf_compile_spec(
        self,
        tosa_spec: TosaSpecification = None,  # type: ignore[assignment]
        compiler_flags: Optional[str] = "",
    ) -> "ArmCompileSpecBuilder":
        """
        Generate compile spec for VGF compatible targets

        Args:
            compiler_flags: Extra compiler flags for converter_backend
        """
        self.output_format = "vgf"
        self.compiler_flags = [
            compiler_flags,
        ]

        if tosa_spec is None:
            tosa_spec = TosaSpecification.create_from_string("TOSA-1.0+FP")

        tosa_version = tosa_spec.version  # type: ignore[attr-defined]
        tosa_profiles = tosa_spec.profiles  # type: ignore[attr-defined]

        if tosa_version.major != 1:
            raise ValueError(
                "Arm backend only supports converter-backend for TOSA version 1. "
                f"Invalid TOSA version: {tosa_version}"
            )

        if "FP" not in tosa_profiles and "INT" not in tosa_profiles:
            raise ValueError(
                "Arm backend only supports converter-backend for FP or INT. "
                f"Invalid TOSA profile: {tosa_profiles}"
            )

        if len(tosa_profiles) != 1:
            raise ValueError(
                "For now Arm backend only supports converter-backend for either FP or INT. "
                f"Invalid TOSA profile: {tosa_profiles}"
            )

        self.tosa_spec = tosa_spec

        return self

    def ethosu_compile_spec(
        self,
        target: str,
        system_config: Optional[str] = None,
        memory_mode: Optional[str] = None,
        extra_flags: Optional[str] = None,
        config_ini: Optional[str] = "Arm/vela.ini",
    ) -> "ArmCompileSpecBuilder":
        """
        Generate compile spec for Ethos-U NPU

        Args:
            target: Ethos-U accelerator configuration, e.g. ethos-u55-128
            system_config: System configuration to select from the Vel
                configuration file
            memory_mode: Memory mode to select from the Vela configuration file
            extra_flags: Extra flags for the Vela compiler
            config_ini: Vela configuration file(s) in Python ConfigParser .ini
                file format
        """
        assert (
            self.output_format is None
        ), f"Output format already set to f{self.output_format}"
        self.output_format = "vela"
        self.compiler_flags = [
            f"--accelerator-config={target}",
            f"--config={config_ini}",
        ]

        # default system config and memory mode
        if "ethos-u55" in target:
            if system_config is None:
                system_config = "Ethos_U55_High_End_Embedded"
            if memory_mode is None:
                memory_mode = "Shared_Sram"
        elif "ethos-u85" in target:
            if system_config is None:
                system_config = "Ethos_U85_SYS_DRAM_Mid"
            if memory_mode is None:
                memory_mode = "Sram_Only"
        else:
            raise RuntimeError(f"Unknown ethos target: {target}")

        if system_config is not None:
            self.compiler_flags.append(f"--system-config={system_config}")
        if memory_mode is not None:
            self.compiler_flags.append(f"--memory-mode={memory_mode}")
        if extra_flags is not None:
            self.compiler_flags.append(extra_flags)

        # We require raw output and regor, so add these flags if absent. This
        # overrides any other output setting.
        self.compiler_flags.append("--output-format=raw")
        self.compiler_flags.append("--debug-force-regor")

        base_tosa_version = "TOSA-1.0+INT+int16"
        if "u55" in target:
            # Add the Ethos-U55 extension marker
            base_tosa_version += "+u55"
        self.tosa_spec = TosaSpecification.create_from_string(base_tosa_version)

        return self

    def tosa_compile_spec(
        self, tosa_spec: str | TosaSpecification
    ) -> "ArmCompileSpecBuilder":
        """
        Generate compile spec for TOSA flatbuffer output
        """
        assert (
            self.output_format is None
        ), f"Output format already set: {self.output_format}"
        self.output_format = "tosa"
        if isinstance(tosa_spec, TosaSpecification):
            self.tosa_spec = tosa_spec
        elif isinstance(tosa_spec, str):
            self.tosa_spec = TosaSpecification.create_from_string(tosa_spec)
        else:
            raise RuntimeError(f"Invalid type for {tosa_spec}!")
        return self

    def dump_intermediate_artifacts_to(
        self, output_path: str
    ) -> "ArmCompileSpecBuilder":
        """
        Sets a path for dumping intermediate results during such as tosa and pte.
        """
        self.path_for_intermediates = output_path
        return self

    def dump_debug_info(self, debug_mode: DebugMode) -> "ArmCompileSpecBuilder":
        """
        Dump debugging information into the intermediates path
        """
        self.tosa_debug_mode = debug_mode.name
        return self

    def build(self) -> List[CompileSpec]:
        """
        Generate a list of compile spec objects from the builder
        """
        assert self.tosa_spec

        # Always supply a TOSA version
        self.compile_spec = [CompileSpec("tosa_spec", str(self.tosa_spec).encode())]

        # Add compile flags, these are backend specific, refer to the backend
        # documentation.
        self.compile_spec += [
            CompileSpec("compile_flags", " ".join(self.compiler_flags).encode()),
        ]

        # encode output format
        self.compile_spec.append(
            CompileSpec("output_format", self.output_format.encode())
        )

        if self.path_for_intermediates is not None:
            self.compile_spec.append(
                CompileSpec("debug_artifact_path", self.path_for_intermediates.encode())
            )

        if self.tosa_debug_mode is not None:
            if not self.path_for_intermediates:
                raise ValueError(
                    "dump_debug_info() must be used in conjunction with dump_intermediate_artifacts_to()"
                )

            self.compile_spec.append(
                CompileSpec("dump_debug_info", self.tosa_debug_mode.encode())
            )

        return self.compile_spec


def is_tosa(compile_spec: List[CompileSpec]) -> bool:
    has_tosa_output = False
    has_tosa_spec = False
    for spec in compile_spec:
        if spec.key == "output_format":
            has_tosa_output = spec.value.decode() == "tosa"
        if spec.key == "tosa_spec":
            has_tosa_spec = True

    return has_tosa_output and has_tosa_spec


def is_ethosu(compile_spec: List[CompileSpec]) -> bool:
    for spec in compile_spec:
        if spec.key == "output_format":
            return spec.value.decode() == "vela"
    return False


def is_vgf(compile_spec: List[CompileSpec]) -> bool:
    for spec in compile_spec:
        if spec.key == "output_format":
            return spec.value.decode() == "vgf"
    return False


def get_intermediate_path(compile_spec: List[CompileSpec]) -> Optional[str]:
    for spec in compile_spec:
        if spec.key == "debug_artifact_path":
            return spec.value.decode()
    return None
