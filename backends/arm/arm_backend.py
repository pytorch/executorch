# Copyright 2023-2024 Arm Limited and/or its affiliates.
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
from typing import Callable, List, Optional

from executorch.backends.arm.arm_ethosu_backend import ArmEthosUBackend
from executorch.backends.arm.arm_tosa_backend import ArmTOSABackend

from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import DelegationSpec


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class ArmCompileSpecBuilder:
    def __init__(self):
        self.compile_spec: List[CompileSpec] = []
        self.compiler_flags = []
        self.output_format = None
        self.path_for_intermediates = None
        # TODO MLETORCH-265 Remove permute_nhwc flag
        self.permute_nhwc = False
        self.quantize_io = False

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

        return self

    def tosa_compile_spec(self) -> "ArmCompileSpecBuilder":
        """
        Generate compile spec for TOSA flatbuffer output
        """
        assert (
            self.output_format is None
        ), f"Output format already set: {self.output_format}"
        self.output_format = "tosa"
        return self

    def dump_intermediate_artifacts_to(
        self, output_path: str
    ) -> "ArmCompileSpecBuilder":
        """
        Sets a path for dumping intermediate results during such as tosa and pte.
        """
        self.path_for_intermediates = output_path
        return self

    def set_permute_memory_format(
        self, set_nhwc_permutation: bool = True
    ) -> "ArmCompileSpecBuilder":
        """
        Permute to channel last in compiler and runtime. Compilation and
        runtime will convert rank 4 inputs to channel last for each sub-graph.
        """
        self.permute_nhwc = set_nhwc_permutation
        return self

    def set_quantize_io(self, quantize_io: bool = False) -> "ArmCompileSpecBuilder":
        """
        Quantization of inputs and dequantization of outputs for cases where
        whole graph is quantized and method signature is not of quantized type.
        """
        self.quantize_io = quantize_io
        return self

    def build(self) -> List[CompileSpec]:
        """
        Generate a list of compile spec objects from the builder
        """
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

        if self.permute_nhwc:
            self.compile_spec.append(
                CompileSpec("permute_memory_format", "nhwc".encode())
            )

        if self.quantize_io:
            self.compile_spec.append(CompileSpec("quantize_io", "True".encode()))

        return self.compile_spec


def is_permute_memory(compile_spec: List[CompileSpec]) -> bool:
    for spec in compile_spec:
        if spec.key == "permute_memory_format":
            return spec.value.decode() == "nhwc"
    return False


def is_tosa(compile_spec: List[CompileSpec]) -> bool:
    for spec in compile_spec:
        if spec.key == "output_format":
            return spec.value.decode() == "tosa"
    return False


def is_ethosu(compile_spec: List[CompileSpec]) -> bool:
    for spec in compile_spec:
        if spec.key == "output_format":
            return spec.value.decode() == "vela"
    return False


def is_arm_compile_spec(compile_spec: list[CompileSpec]) -> bool:
    for spec in compile_spec:
        if spec.key == "output_format":
            return spec.value.decode() in ["tosa", "vela"]
    return False


def get_intermediate_path(compile_spec: List[CompileSpec]) -> Optional[str]:
    for spec in compile_spec:
        if spec.key == "debug_artifact_path":
            return spec.value.decode()
    return None


class ArmBackendSelector:
    backend_filtering: list[Callable[[List[CompileSpec]], bool], str] = [
        (is_tosa, ArmTOSABackend.__name__),
        (is_ethosu, ArmEthosUBackend.__name__),
    ]

    @staticmethod
    def get_delegation_spec(compile_spec: List[CompileSpec]) -> DelegationSpec:
        """
        Returns a corresponding DelegationSpec from a list of CompileSpec.
        Figures out what the compile_spec list is targeting, e.g. tosa or vela.
        """

        backend_id = None
        for filter_fn, backend_name in ArmBackendSelector.backend_filtering:
            if filter_fn(compile_spec):
                backend_id = backend_name
                break
        if backend_id is None:
            raise RuntimeError("Wrong compile_spec. Not targetting Arm hardware")

        return DelegationSpec(backend_id, compile_spec)
