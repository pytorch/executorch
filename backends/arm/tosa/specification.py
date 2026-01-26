# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Provide TOSA specification parsing and context utilities.

Use these helpers to parse and validate TOSA profile/extension strings and to
manage a lowering-time context for the active specification.

"""

import contextvars
import re
from typing import Dict, Generic, List, Set, TypeVar

from packaging.version import Version

T = TypeVar("T")


class TosaSpecMapping(Generic[T]):
    def __init__(self):
        self._mapping: Dict[TosaSpecification, List[T]] = {}

    def add(self, spec: "TosaSpecification", value: T) -> None:
        """
        Adds a value to the mapping for the given TOSA specification.
        The specification is normalized to its canonical form, which means that
        only the version and profiles are considered, without extensions.
        This allows for grouping of values under the same TOSA specification
        regardless of the extensions they may have.
        """

        if spec.is_U55_subset or spec.extensions:
            raise ValueError(
                f"TosaSpecMapping does not support extensions, got: {spec}"
            )

        if isinstance(spec, Tosa_1_00) and len(spec.profiles) > 1:
            raise ValueError(
                f"TosaSpecMapping does not support multiple profiles, got: {spec}"
            )

        norm_spec = spec._canonical_key()
        if norm_spec not in self._mapping:
            self._mapping[norm_spec] = []
        self._mapping[norm_spec].append(value)

    @staticmethod
    def _get_base_specs(spec: "TosaSpecification") -> List["TosaSpecification"]:
        # Handles combined TOSA-1.0+FP+INT, etc.
        if isinstance(spec, Tosa_1_00):
            profiles: Set[str] = set(spec.profiles)
            if profiles == {"FP", "INT"}:
                version = spec.version
                return [
                    TosaSpecification.create_from_string(f"TOSA-{version}+FP"),
                    TosaSpecification.create_from_string(f"TOSA-{version}+INT"),
                ]
        return [spec]

    def get(self, spec: "TosaSpecification") -> List[T]:
        """
        Returns a list of values associated with the given TOSA specification.
        The specification is normalized to its canonical form, which means that
        only the version and profiles are considered, without extensions.
        """

        base_specs = self._get_base_specs(spec)
        result: List[T] = []
        for base in base_specs:
            norm_base = base._canonical_key()
            result.extend(self._mapping.get(norm_base, []))
        if len(result) == 0:
            raise KeyError(f"No values found for TOSA specification: {spec}")

        return result  # Do not deduplicate with set(), as values may be unhashable


class TosaSpecification:
    """Represent a TOSA specification.

    A specification includes a semantic version, one or more profiles, and
    optional extensions and levels (for example ``8k``).
    The encoded form follows ``TOSA-<major>.<minor>.<patch>+<PROFILE>[+<LEVEL>][+<EXT>...]``.
    Profiles use uppercase (for example ``INT``, ``FP``); levels and extensions
    use lowercase.

    Attributes:
        version (Version): Parsed TOSA semantic version.
        is_U55_subset (bool): True if the ``u55`` subset is requested.

    """

    version: Version
    is_U55_subset: bool
    extensions: List[str]

    def support_integer(self) -> bool:
        """Return True if integer operations are supported."""
        raise NotImplementedError

    def support_float(self) -> bool:
        """Return True if floating-point operations are supported."""
        raise NotImplementedError

    def support_extension(self, extension: str) -> bool:
        """Return True if an extension is supported and enabled.

        Args:
            extension (str): Extension name (for example ``int4``, ``bf16``).

        Returns:
            bool: True if the extension is valid for the active profiles and selected.

        """
        raise NotImplementedError

    def __init__(self, version: Version, extras: List[str]):
        """Initialize the base specification.

        Args:
            version (Version): Parsed TOSA semantic version.
            extras (List[str]): Remaining tokens such as profiles, levels, and extensions.

        """
        self.version = version
        self.extensions = []

        self.is_U55_subset = "u55" in extras
        if self.is_U55_subset:
            extras.remove("u55")

    @staticmethod
    def create_from_string(repr: str) -> "TosaSpecification":
        """Create a specification from a standard string format.

        Example: ``TOSA-1.00.0+INT+FP+int4+cf``.

        Args:
            repr (str): Standard representation string.

        Returns:
            TosaSpecification: Parsed specification instance.

        Raises:
            ValueError: If the representation is malformed or version is unsupported.

        """
        pattern = r"^(TOSA)-([\d.]+)\+(.+)$"
        match = re.match(pattern, repr)
        if match:
            name = match.group(1)
            version = Version(match.group(2))
            extras = match.group(3).split("+")
            if name != "TOSA":
                raise ValueError(f"Malformed TOSA specification representation: {repr}")
            match version.major, version.minor:
                case [1, 0]:
                    return Tosa_1_00(version, extras)
                case [1, 1]:
                    return Tosa_1_1(version, extras)
                case _:
                    raise ValueError(f"Wrong TOSA version: {version} from {repr}")

        raise ValueError(f"Failed to parse TOSA specification representation: {repr}")

    def _canonical_key(self) -> "TosaSpecification":
        """
        Returns a new TosaSpecification instance with only version and profiles (no extensions).
        """
        raise NotImplementedError


class Tosa_1_00(TosaSpecification):
    """Provide TOSA 1.00 profile and extension semantics.

    This variant validates profiles (``INT``, ``FP``), the optional ``8k`` level,
    and allowed extensions based on the selected profiles.

    Attributes:
        profiles (List[str]): Selected profiles, e.g., ``["INT"]`` or ``["INT", "FP"]``.
        level_8k (bool): True if the ``8k`` level is enabled.
        extensions (List[str]): Enabled extensions valid for the chosen profiles.

    """

    profiles: List[str]
    level_8k: bool
    extensions: List[str]

    available_profiles = ["INT", "FP"]
    valid_extensions = {
        "INT": ["int16", "int4", "var", "cf", "u55"],
        "FP": ["bf16", "fp8e4m3", "fp8e5m2", "fft", "var", "cf"],
    }

    def __init__(self, version: Version, extras: List[str]):
        """Initialize the 1.00 specification and validate extras.

        Args:
            version (Version): Semantic version (major=1, minor=0).
            extras (List[str]): Tokens including profiles, level, and extensions.

        Raises:
            ValueError: If no/too many profiles are provided or extensions are invalid.

        """
        super().__init__(version, extras)

        cls = self.__class__

        # Check that we have at least one profile in the extensions list
        if [e in cls.available_profiles for e in extras].count(True) == 0:
            raise ValueError(
                f"No profile ({cls.available_profiles}) found in: {extras}."
            )

        # and not more than number of available profiles
        if [e in cls.available_profiles for e in extras].count(True) > len(
            cls.available_profiles
        ):
            raise ValueError(
                f"Too many profiles ({cls.available_profiles}) found in: {extras}."
            )

        # The list contains one profile at least, so pick them
        self.profiles = [e for e in extras if e in cls.available_profiles]
        for p in self.profiles:
            extras.remove(p)

        self.level_8k = "8k" in extras
        if self.level_8k:
            extras.remove("8k")

        combined_extensions = []
        for p in self.profiles:
            combined_extensions += cls.valid_extensions[p]

        if not all(e in combined_extensions for e in extras):
            raise ValueError(
                f"Bad extensions for TOSA-{version}{self._get_profiles_string()}: {extras}"
            )

        # all the rest of the extras are handled extensions
        self.extensions = extras

    def _get_profiles_string(self) -> str:
        """Return the ``+``-joined profile segment (e.g., ``+INT+FP``)."""
        return "".join(["+" + p for p in self.profiles])

    def _get_extensions_string(self) -> str:
        """Return the ``+``-joined extensions segment (e.g., ``+int4+cf``)."""
        return "".join(["+" + e for e in self.extensions])

    def __repr__(self):
        """Return the standard specification string format.

        Returns:
            str: Standard form like ``TOSA-1.00.0+INT+8k+int4``.

        """
        extensions = self._get_extensions_string()
        if self.level_8k:
            extensions += "+8k"
        if self.is_U55_subset:
            extensions += "+u55"
        return f"TOSA-{self.version}{self._get_profiles_string()}{extensions}"

    def __hash__(self) -> int:
        """Return a stable hash for use in sets and dict keys.

        Returns:
            int: Hash value derived from version and profiles.

        """
        return hash(str(self.version) + self._get_profiles_string())

    def __eq__(self, other: object) -> bool:
        """Return True if another instance represents the same spec.

        Args:
            other (object): Object to compare.

        Returns:
            bool: True if versions and profiles match.

        """
        if isinstance(other, Tosa_1_00):
            return (self.version == other.version) and (
                self._get_profiles_string() == other._get_profiles_string()
            )
        return False

    def support_integer(self):
        """Return True if the ``INT`` profile is present."""
        return "INT" in self.profiles

    def support_float(self):
        """Return True if the ``FP`` profile is present."""
        return "FP" in self.profiles

    def support_extension(self, extension: str) -> bool:
        """Return True if an extension is supported and enabled.

        Args:
            extension (str): Extension name (for example ``int4``, ``bf16``).

        Returns:
            bool: True if the extension is valid for the active profiles and selected.

        """
        cls = self.__class__
        for p in self.profiles:
            if extension in cls.valid_extensions[p] and extension in self.extensions:
                return True
        return False

    def _canonical_key(self) -> "Tosa_1_00":
        """
        Returns a new Tosa_1_00 instance with only major.minor version and profiles (no extensions).
        Patch version is set to zero for normalization.
        """
        from packaging.version import Version

        norm_version = Version(f"{self.version.major}.{self.version.minor}.0")
        return Tosa_1_00(norm_version, self.profiles.copy())


class Tosa_1_1(Tosa_1_00):

    valid_extensions = {
        "INT": ["shape", "int64", "int16", "int4", "var", "cf", "u55"],
        "FP": [
            "shape",
            "int64",
            "bf16",
            "fp8e4m3",
            "fp8e5m2",
            "fft",
            "var",
            "cf",
            "random",
            "mxfp",
            "blockscale_ue5m3",
        ],
    }

    pass


class TosaLoweringContext:
    """Manage the TOSA specification context for lowering.

    For now, only the active ``TosaSpecification`` is tracked, but this can be
    extended to carry additional lowering policies or configuration.

    Attributes:
        tosa_spec_var (contextvars.ContextVar): Context variable storing the active spec.
        spec (TosaSpecification): Specification passed to the context manager.

    """

    # Define a context variable for the spec
    tosa_spec_var: contextvars.ContextVar = contextvars.ContextVar("tosa_spec")

    def __init__(self, spec: TosaSpecification):
        """Initialize the lowering context with a specification.

        Args:
            spec (TosaSpecification): Active specification to put into context.

        """
        self.spec = spec

    def __enter__(self):
        """Set the context variable and return self.

        Returns:
            TosaLoweringContext: This context manager instance.

        """
        # Set the spec in the context variable and store the token for later reset
        self.token = TosaLoweringContext.tosa_spec_var.set(self.spec)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Reset the context variable to its previous state.

        Args:
            exc_type (type | None): Exception type, if any.
            exc_value (BaseException | None): Exception instance, if any.
            traceback (TracebackType | None): Traceback, if any.

        """
        # Reset the context variable to its previous state
        TosaLoweringContext.tosa_spec_var.reset(self.token)


def get_context_spec() -> TosaSpecification:
    """Get the current ``TosaSpecification`` from the lowering context.

    Returns:
        TosaSpecification: Active specification retrieved from the context var.

    Raises:
        RuntimeError: If called outside a ``TosaLoweringContext``.

    """
    try:
        return TosaLoweringContext.tosa_spec_var.get()
    except LookupError:
        raise RuntimeError("Function must be executed within a TosaLoweringContext")


def tosa_spec_in_set(spec: TosaSpecification, specs: Set[TosaSpecification]) -> bool:
    """Check if a specification matches any in a set, considering base specs.

    Args:
        spec (TosaSpecification): Specification to check.
        specs (Set[TosaSpecification]): Set of specifications to match against.

    Returns:
        bool: True if a match is found, False otherwise.

    """
    base_specs = TosaSpecMapping._get_base_specs(spec)
    for base in base_specs:
        if base in specs:
            return True
    return False
