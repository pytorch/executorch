# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-25 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import platform
import re
import subprocess
import sys
from typing import List, Optional

# Supported CUDA versions - modify this to add/remove supported versions
# Format: tuple of (major, minor) version numbers
SUPPORTED_CUDA_VERSIONS = (
    (12, 6),
    (12, 8),
    (12, 9),
    (13, 0),
)


def is_cmake_option_on(
    cmake_configuration_args: List[str], var_name: str, default: bool
) -> bool:
    """
    Get a boolean CMake variable, from a list of CMake configuration arguments.
    The var_name should not include the "-D" prefix.

    Args:
        cmake_configuration_args: List of CMake configuration arguments.
        var_name: Name of the CMake variable.
        default: Default boolean value if the variable is not set.

    Returns:
        Boolean value of the CMake variable.
    """
    cmake_define = _extract_cmake_define(cmake_configuration_args, var_name)

    return _normalize_cmake_bool(cmake_define, default)


def is_cuda_available() -> bool:
    """
    Check if CUDA is available on the system by attempting to get the CUDA version.

    Returns:
        True if CUDA is available and supported, False otherwise.
    """
    try:
        _get_cuda_version()
        return True
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def _get_cuda_version():
    """
    Get the CUDA version installed on the system using nvcc command.
    Returns a tuple (major, minor).

    Raises:
        RuntimeError: if nvcc is not found or version cannot be parsed
    """
    try:
        # Get CUDA version from nvcc (CUDA compiler)
        nvcc_result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, check=True
        )
        # Parse nvcc output for CUDA version
        # Output contains line like "Cuda compilation tools, release 12.6, V12.6.68"
        match = re.search(r"release (\d+)\.(\d+)", nvcc_result.stdout)
        if match:
            major, minor = int(match.group(1)), int(match.group(2))

            # Check if the detected version is supported
            if (major, minor) not in SUPPORTED_CUDA_VERSIONS:
                available_versions = ", ".join(
                    [f"{maj}.{min}" for maj, min in SUPPORTED_CUDA_VERSIONS]
                )
                raise RuntimeError(
                    f"Detected CUDA version {major}.{minor} is not supported. "
                    f"Supported versions: {available_versions}."
                )

            return (major, minor)
        else:
            raise RuntimeError(
                "Failed to parse CUDA version from nvcc output. "
                "Ensure CUDA is properly installed."
            )
    except FileNotFoundError:
        raise RuntimeError(
            "nvcc (CUDA compiler) is not found in PATH. Install the CUDA toolkit."
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"nvcc command failed with error: {e}. "
            "Ensure CUDA is properly installed."
        )


def _extract_cmake_define(args: List[str], name: str) -> Optional[str]:
    prefix = f"-D{name}="
    for arg in args:
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def _normalize_cmake_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().upper()
    if normalized in {"ON", "1", "TRUE", "YES"}:
        return True
    if normalized in {"OFF", "0", "FALSE", "NO"}:
        return False
    return default


def _cuda_version_to_pytorch_suffix(major, minor):
    """
    Generate PyTorch CUDA wheel suffix from CUDA version numbers.

    Args:
        major: CUDA major version (e.g., 12)
        minor: CUDA minor version (e.g., 6)

    Returns:
        PyTorch wheel suffix string (e.g., "cu126")
    """
    return f"cu{major}{minor}"


def _get_pytorch_cuda_url(cuda_version, torch_nightly_url_base):
    """
    Get the appropriate PyTorch CUDA URL for the given CUDA version.

    Args:
        cuda_version: tuple of (major, minor) version numbers
        torch_nightly_url_base: Base URL for PyTorch nightly packages

    Returns:
        URL string for PyTorch CUDA packages
    """
    major, minor = cuda_version
    # Generate CUDA suffix (version validation is already done in _get_cuda_version)
    cuda_suffix = _cuda_version_to_pytorch_suffix(major, minor)

    return f"{torch_nightly_url_base}/{cuda_suffix}"


@functools.lru_cache(maxsize=1)
def determine_torch_url(torch_nightly_url_base):
    """
    Determine the appropriate PyTorch installation URL based on CUDA availability.
    Uses @functools.lru_cache to avoid redundant CUDA detection and print statements.

    Args:
        torch_nightly_url_base: Base URL for PyTorch nightly packages

    Returns:
        URL string for PyTorch packages
    """
    if platform.system().lower() == "windows":
        print(
            "Windows detected, using CPU-only PyTorch until CUDA support is available"
        )
        return f"{torch_nightly_url_base}/cpu"

    print("Attempting to detect CUDA via nvcc...")

    try:
        cuda_version = _get_cuda_version()
    except Exception as err:
        print(f"CUDA detection failed ({err}), using CPU-only PyTorch")
        return f"{torch_nightly_url_base}/cpu"

    major, minor = cuda_version
    print(f"Detected CUDA version: {major}.{minor}")

    # Get appropriate PyTorch CUDA URL
    torch_url = _get_pytorch_cuda_url(cuda_version, torch_nightly_url_base)
    print(f"Using PyTorch URL: {torch_url}")

    return torch_url


# Prebuilt binaries for Intel-based macOS are no longer available on PyPI; users must compile from source.
# PyTorch stopped building macOS x86_64 binaries since version 2.3.0 (January 2024).
def is_intel_mac_os():
    # Returns True if running on Intel macOS.
    return platform.system().lower() == "darwin" and platform.machine().lower() in (
        "x86",
        "x86_64",
        "i386",
    )


def python_is_compatible():
    # Scrape the version range from pyproject.toml, which should be in the current directory.
    version_specifier = None
    with open("pyproject.toml", "r") as file:
        for line in file:
            if line.startswith("requires-python"):
                match = re.search(r'"([^"]*)"', line)
                if match:
                    version_specifier = match.group(1)
                    break

    if not version_specifier:
        print(
            "WARNING: Skipping python version check: version range not found",
            file=sys.stderr,
        )
        return False

    # Install the packaging module if necessary.
    try:
        import packaging
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "packaging"], check=True
        )
    # Compare the current python version to the range in version_specifier. Exits
    # with status 1 if the version is not compatible, or with status 0 if the
    # version is compatible or the logic itself fails.
    try:
        import packaging.specifiers
        import packaging.version

        python_version = packaging.version.parse(platform.python_version())
        version_range = packaging.specifiers.SpecifierSet(version_specifier)
        if python_version not in version_range:
            print(
                f'ERROR: ExecuTorch does not support python version {python_version}: must satisfy "{version_specifier}"',
                file=sys.stderr,
            )
            return False
    except Exception as e:
        print(f"WARNING: Skipping python version check: {e}", file=sys.stderr)
    return True
