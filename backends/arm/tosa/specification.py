# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

#
# Main implementation of AoT flow to partition and preprocess for Arm target
# backends. Converts via TOSA as an intermediate form supported by AoT and
# JIT compiler flows.
#

import contextvars
import re
from typing import List

from packaging.version import Version


class TosaSpecification:
    """
    This class implements a representation of TOSA specification
    (https://www.mlplatform.org/tosa/tosa_spec.html) with a version, a profile
    (with extension) and a level (8k).
    For 1.00 releases the profile is INT or FP, and the extensions are for
        INT: int16, int4, var, cf
        FP: bf16, fp8e4m3, fp8e5m2, fft, var, cf

    The TOSA specification is encoded in the string represenatation
        TOSA-major.minor.patch+profile[+level][+extensions]

    Profiles are uppercase letters and extensions and level is lowercase.
    """

    version: Version
    is_U55_subset: bool

    def support_integer(self) -> bool:
        """
        Returns true if any integer operations are supported for the specification.
        """
        raise NotImplementedError

    def support_float(self) -> bool:
        """
        Returns true if any float operations are supported for the specification.
        """
        raise NotImplementedError

    def __init__(self, version: Version, extras: List[str]):
        self.version = version

        self.is_U55_subset = "u55" in extras
        if self.is_U55_subset:
            extras.remove("u55")

    @staticmethod
    def create_from_string(repr: str) -> "TosaSpecification":
        """
        Creates a TOSA specification class from a string representation:
        TOSA-1.00.0+INT+FP+int4+cf
        """

        pattern = r"^(TOSA)-([\d.]+)\+(.+)$"
        match = re.match(pattern, repr)
        if match:
            name = match.group(1)
            version = Version(match.group(2))
            extras = match.group(3).split("+")
            if name != "TOSA":
                raise ValueError(f"Malformed TOSA specification representation: {repr}")
            match version:
                case _ if version.major == 1 and version.minor == 0:
                    return Tosa_1_00(version, extras)
                case _:
                    raise ValueError(f"Wrong TOSA version: {version} from {repr}")

        raise ValueError(f"Failed to parse TOSA specification representation: {repr}")


class Tosa_1_00(TosaSpecification):
    profiles: List[str]
    level_8k: bool
    extensions: List[str]

    available_profiles = ["INT", "FP"]
    valid_extensions = {
        "INT": ["int16", "int4", "var", "cf", "u55"],
        "FP": ["bf16", "fp8e4m3", "fp8e5m2", "fft", "var", "cf"],
    }

    def __init__(self, version: Version, extras: List[str]):
        super().__init__(version, extras)

        # Check that we have at least one profile in the extensions list
        if [e in Tosa_1_00.available_profiles for e in extras].count(True) == 0:
            raise ValueError(
                f"No profile ({Tosa_1_00.available_profiles}) found in: {extras}."
            )

        # and not more than number of available profiles
        if [e in Tosa_1_00.available_profiles for e in extras].count(True) > len(
            Tosa_1_00.available_profiles
        ):
            raise ValueError(
                f"Too many profiles ({Tosa_1_00.available_profiles}) found in: {extras}."
            )

        # The list contains one profile at least, so pick them
        self.profiles = [e for e in extras if e in Tosa_1_00.available_profiles]
        for p in self.profiles:
            extras.remove(p)

        self.level_8k = "8k" in extras
        if self.level_8k:
            extras.remove("8k")

        combined_extensions = []
        for p in self.profiles:
            combined_extensions += Tosa_1_00.valid_extensions[p]

        if not all(e in combined_extensions for e in extras):
            raise ValueError(
                f"Bad extensions for TOSA-{version}{self._get_profiles_string()}: {extras}"
            )

        # all the rest of the extras are handled extensions
        self.extensions = extras

    def _get_profiles_string(self) -> str:
        return "".join(["+" + p for p in self.profiles])

    def _get_extensions_string(self) -> str:
        return "".join(["+" + e for e in self.extensions])

    def __repr__(self):
        extensions = self._get_extensions_string()
        if self.level_8k:
            extensions += "+8k"
        if self.is_U55_subset:
            extensions += "+u55"
        return f"TOSA-{self.version}{self._get_profiles_string()}{extensions}"

    def __hash__(self) -> int:
        return hash(str(self.version) + self._get_profiles_string())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Tosa_1_00):
            return (self.version == other.version) and (
                self._get_profiles_string() == other._get_profiles_string()
            )
        return False

    def support_integer(self):
        return "INT" in self.profiles

    def support_float(self):
        return "FP" in self.profiles

    def support_extension(self, extension: str) -> bool:
        for p in self.profiles:
            if extension in self.valid_extensions[p] and extension in self.extensions:
                return True

        return False


class TosaLoweringContext:
    """
    A context manager to handle the TOSA specific aspects of the lowering process.
    For now it only handles the TOSA specification context, but it can be extended
    to include other policies or configurations.
    """

    # Define a context variable for the spec
    tosa_spec_var: contextvars.ContextVar = contextvars.ContextVar("tosa_spec")

    def __init__(self, spec: TosaSpecification):
        self.spec = spec

    def __enter__(self):
        # Set the spec in the context variable and store the token for later reset
        self.token = TosaLoweringContext.tosa_spec_var.set(self.spec)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Reset the context variable to its previous state
        TosaLoweringContext.tosa_spec_var.reset(self.token)


# A helper function to retrieve the current spec anywhere in your code
def get_context_spec() -> TosaSpecification:
    try:
        return TosaLoweringContext.tosa_spec_var.get()
    except LookupError:
        raise RuntimeError("Function must be executed within a TosaLoweringContext")
