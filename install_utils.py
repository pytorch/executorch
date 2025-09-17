# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-25 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import platform
import re
import subprocess


def _is_cuda_enabled():
    """Check if CUDA delegate is enabled via CMAKE_ARGS environment variable."""
    cmake_args = os.environ.get("CMAKE_ARGS", "")
    return "-DEXECUTORCH_BUILD_CUDA=ON" in cmake_args


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


def _get_cuda_version(supported_cuda_versions):
    """
    Get the CUDA version installed on the system using nvcc command.
    Returns a tuple (major, minor).

    Args:
        supported_cuda_versions: List of supported CUDA versions as tuples

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
            if (major, minor) not in supported_cuda_versions:
                available_versions = ", ".join(
                    [f"{maj}.{min}" for maj, min in supported_cuda_versions]
                )
                raise RuntimeError(
                    f"Detected CUDA version {major}.{minor} is not supported. "
                    f"Only the following CUDA versions are supported: {available_versions}. "
                    f"Please install a supported CUDA version or try on CPU-only delegates."
                )

            return (major, minor)
        else:
            raise RuntimeError(
                "CUDA delegate is enabled but could not parse CUDA version from nvcc output. "
                "Please ensure CUDA is properly installed or try on CPU-only delegates."
            )
    except FileNotFoundError:
        raise RuntimeError(
            "CUDA delegate is enabled but nvcc (CUDA compiler) is not found in PATH. "
            "Please install CUDA toolkit or try on CPU-only delegates."
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"CUDA delegate is enabled but nvcc command failed with error: {e}. "
            "Please ensure CUDA is properly installed or try on CPU-only delegates."
        )


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


# Global variable for caching torch URL
_torch_url_cache = ""


def determine_torch_url(torch_nightly_url_base, supported_cuda_versions):
    """
    Determine the appropriate PyTorch installation URL based on CUDA availability and CMAKE_ARGS.
    Uses caching to avoid redundant CUDA detection and print statements.

    Args:
        torch_nightly_url_base: Base URL for PyTorch nightly packages
        supported_cuda_versions: List of supported CUDA versions as tuples

    Returns:
        URL string for PyTorch packages
    """
    global _torch_url_cache

    # Return cached URL if already determined
    if _torch_url_cache:
        return _torch_url_cache

    # Check if CUDA delegate is enabled
    if not _is_cuda_enabled():
        print("CUDA delegate not enabled, using CPU-only PyTorch")
        _torch_url_cache = f"{torch_nightly_url_base}/cpu"
        return _torch_url_cache

    print("CUDA delegate enabled, detecting CUDA version...")

    # Get CUDA version
    cuda_version = _get_cuda_version(supported_cuda_versions)

    major, minor = cuda_version
    print(f"Detected CUDA version: {major}.{minor}")

    # Get appropriate PyTorch CUDA URL
    torch_url = _get_pytorch_cuda_url(cuda_version, torch_nightly_url_base)
    print(f"Using PyTorch URL: {torch_url}")

    # Cache the result
    _torch_url_cache = torch_url
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
