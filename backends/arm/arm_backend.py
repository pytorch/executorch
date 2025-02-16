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

import logging

from typing import List, Optional

from executorch.backends.arm.tosa_specification import TosaSpecification

from executorch.exir.backend.compile_spec_schema import CompileSpec


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class ArmCompileSpecBuilder:
    def __init__(self):
        self.compile_spec: List[CompileSpec] = []
        self.compiler_flags = []
        self.output_format = None
        self.path_for_intermediates = None
        self.tosa_spec = None
        self.input_order = None

    def ethosu_compile_spec(
        self,
        config: str,
        system_config: str,
        memory_mode: str,
        extra_flags: Optional[str] = None,
        config_ini: Optional[str] = "Arm/vela.ini",
    ) -> "ArmCompileSpecBuilder":
        """
        Generate compile spec for Ethos-U NPU

        Args:
            config: Ethos-U accelerator configuration, e.g. ethos-u55-128
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
            f"--accelerator-config={config}",
            f"--config={config_ini}",
        ]
        if system_config is not None:
            self.compiler_flags.append(f"--system-config={system_config}")
        if memory_mode is not None:
            self.compiler_flags.append(f"--memory-mode={memory_mode}")
        if extra_flags is not None:
            self.compiler_flags.append(extra_flags)

        base_tosa_version = "TOSA-0.80+BI"
        if "u55" in config:
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

    def build(self) -> List[CompileSpec]:
        """
        Generate a list of compile spec objects from the builder
        """
        assert self.tosa_spec

        # Always supply a TOSA version
        self.compile_spec = [CompileSpec("tosa_spec", str(self.tosa_spec).encode())]

        if self.output_format == "vela":
            self.compile_spec += [
                CompileSpec("output_format", "vela".encode()),
                CompileSpec("compile_flags", " ".join(self.compiler_flags).encode()),
            ]
        elif self.output_format == "tosa":
            self.compile_spec.append(CompileSpec("output_format", "tosa".encode()))

        if self.path_for_intermediates is not None:
            self.compile_spec.append(
                CompileSpec("debug_artifact_path", self.path_for_intermediates.encode())
            )

        if self.input_order:
            self.compile_spec.append(
                CompileSpec(
                    "input_order", " ".join(map(str, self.input_order)).encode()
                )
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


def get_tosa_spec(compile_spec: List[CompileSpec]) -> TosaSpecification:
    for spec in compile_spec:
        if spec.key == "tosa_spec":
            return TosaSpecification.create_from_string(spec.value.decode())
    raise ValueError("Could not find TOSA version in CompileSpec")


def get_intermediate_path(compile_spec: List[CompileSpec]) -> Optional[str]:
    for spec in compile_spec:
        if spec.key == "debug_artifact_path":
            return spec.value.decode()
    return None
