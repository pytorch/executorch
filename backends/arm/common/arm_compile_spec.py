# Copyright 2023-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#
# Main implementation of AoT flow to partition and preprocess for Arm target
# backends. Converts via TOSA as an intermediate form supported by AoT and
# JIT compiler flows.
#

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from executorch.backends.arm.tosa import TosaSpecification

from executorch.exir.backend.compile_spec_schema import CompileSpec


@dataclass(init=False)
class ArmCompileSpec(ABC):
    class DebugMode(Enum):
        JSON = 1
        TOSA = 2

    tosa_spec: TosaSpecification
    compiler_flags: list[str] = field(default_factory=list)
    path_for_intermediates: str | None = None
    tosa_debug_mode: DebugMode | None = None

    _TOSA_SPEC_KEY = "tosa_spec"
    _COMPILE_FLAGS_KEY = "compile_flags"
    _OUTPUT_FORMAT_KEY = "output_format"
    _DEBUG_ARTIFACT_KEY = "debug_artifact_path"
    _DEBUG_MODE_KEY = "dump_debug_info"

    def _set_compile_specs(
        self,
        tosa_spec: TosaSpecification,
        compiler_flags: list[str],
        path_for_intermediates: str | None = None,
        tosa_debug_mode: DebugMode | None = None,
    ):
        """Set all values of dataclass directly."""
        self.tosa_spec = tosa_spec
        self.compiler_flags = compiler_flags
        self.path_for_intermediates = path_for_intermediates
        self.tosa_debug_mode = tosa_debug_mode

    @classmethod
    def from_list(cls, compile_specs: list[CompileSpec]):  # noqa: C901
        tosa_spec: TosaSpecification | None = None
        output_format: str | None = None
        compiler_flags: list[str] | None = None
        path_for_intermediates: str | None = None
        tosa_debug_mode: ArmCompileSpec.DebugMode | None = None
        unknown_specs: dict[str, str] = {}
        for spec in compile_specs:
            key = spec.key
            val = spec.value.decode()
            if key == ArmCompileSpec._TOSA_SPEC_KEY:
                if tosa_spec is not None:
                    raise ValueError("More than one tosa_spec entry in compile spec.")
                tosa_spec = TosaSpecification.create_from_string(val)
            elif key == ArmCompileSpec._COMPILE_FLAGS_KEY:
                if compiler_flags is not None:
                    raise ValueError(
                        "More than one compiler flags entry in compile spec."
                    )
                compiler_flags = val.split(" ")
            elif key == ArmCompileSpec._OUTPUT_FORMAT_KEY:
                if output_format is not None:
                    raise ValueError(
                        "More than one output format entry in compile spec."
                    )
                output_format = val
            elif key == ArmCompileSpec._DEBUG_ARTIFACT_KEY:
                if path_for_intermediates is not None:
                    raise ValueError(
                        "More than one debug artifact path entry in compile spec."
                    )
                path_for_intermediates = val
            elif key == ArmCompileSpec._DEBUG_MODE_KEY:
                if tosa_debug_mode is not None:
                    raise ValueError(
                        "More than one tosa_debug_mode entry in compile spec."
                    )
                tosa_debug_mode = ArmCompileSpec.DebugMode[val]
            else:
                unknown_specs[key] = val

        if tosa_spec is None:
            raise ValueError("No tosa_spec in compile spec.")
        if output_format is None:
            raise ValueError("No output_format in compile spec.")
        if output_format != cls.get_output_format():
            raise ValueError(
                f"Incorrect output format '{output_format}' for {cls.__name__}, expected '{cls.get_output_format()}'"
            )
        if compiler_flags is None:
            compiler_flags = []

        # Create new object from class, but bypass __init__ and use _set_compile_specs instead.
        compile_spec = cls.__new__(cls)
        compile_spec._set_compile_specs(
            tosa_spec=tosa_spec,
            compiler_flags=compiler_flags,
            path_for_intermediates=path_for_intermediates,
            tosa_debug_mode=tosa_debug_mode,
        )
        cls.from_list_hook(compile_spec, unknown_specs)
        compile_spec.validate()
        return compile_spec

    @classmethod
    def from_list_hook(cls, compile_spec, specs: dict[str, str]):  # noqa: B027
        """Allows subclasses to hook into parsing compile spec lists."""
        pass

    @abstractmethod
    def validate(self):
        """Throws an error if the compile spec is not valid."""

    def to_list(self):
        """Get the ArmCompileSpec in list form."""
        if not self.tosa_spec:
            raise ValueError("tosa_spec must be set before calling to_list()")

        # Always supply a TOSA version
        compile_spec = [
            CompileSpec(ArmCompileSpec._TOSA_SPEC_KEY, str(self.tosa_spec).encode())
        ]

        # Add compile flags, these are backend specific, refer to the backend
        # documentation.
        if len(self.compiler_flags) > 0:
            compile_spec += [
                CompileSpec(
                    ArmCompileSpec._COMPILE_FLAGS_KEY,
                    " ".join(self.compiler_flags).encode(),
                ),
            ]

        # Add output format to identify kind of compile spec.
        compile_spec.append(
            CompileSpec(
                ArmCompileSpec._OUTPUT_FORMAT_KEY, self.get_output_format().encode()
            )
        )

        if self.path_for_intermediates is not None:
            compile_spec.append(
                CompileSpec(
                    ArmCompileSpec._DEBUG_ARTIFACT_KEY,
                    self.path_for_intermediates.encode(),
                )
            )

        if self.tosa_debug_mode is not None:
            if not self.path_for_intermediates:
                raise ValueError(
                    "dump_debug_info() must be used in conjunction with dump_intermediate_artifacts_to()"
                )

            compile_spec.append(
                CompileSpec(
                    ArmCompileSpec._DEBUG_MODE_KEY, self.tosa_debug_mode.name.encode()
                )
            )

        return compile_spec

    def get_intermediate_path(self) -> str | None:
        """
        Gets the path used for dumping intermediate results such as tosa and pte.

        Returns:
            Path where intermediate results are saved.
        """
        return self.path_for_intermediates

    def dump_intermediate_artifacts_to(self, output_path: str | None):
        """
        Sets a path for dumping intermediate results during such as tosa and pte.

        Args:
            output_path: Path to dump intermediate results to.
        """
        self.path_for_intermediates = output_path
        return self

    def dump_debug_info(self, debug_mode: DebugMode | None):
        """
        Dump debugging information into the intermediates path.

        Args:
            debug_mode: The debug mode to use for dumping debug information.
        """
        self.tosa_debug_mode = debug_mode
        return self

    @classmethod
    @abstractmethod
    def get_output_format(cls) -> str:
        """Returns a constant string that is the output format of the class."""
