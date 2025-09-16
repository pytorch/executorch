# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-25 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import platform
import re
import subprocess
import sys


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


# The pip repository that hosts nightly torch packages.
# This will be dynamically set based on CUDA availability and CUDA backend enabled/disabled.
TORCH_NIGHTLY_URL_BASE = "https://download.pytorch.org/whl/nightly"

# Supported CUDA versions - modify this to add/remove supported versions
# Format: tuple of (major, minor) version numbers
SUPPORTED_CUDA_VERSIONS = [
    (12, 6),
    (12, 8),
    (12, 9),
]

# Since ExecuTorch often uses main-branch features of pytorch, only the nightly
# pip versions will have the required features.
#
# NOTE: If a newly-fetched version of the executorch repo changes the value of
# NIGHTLY_VERSION, you should re-run this script to install the necessary
# package versions.
#
# NOTE: If you're changing, make the corresponding change in .ci/docker/ci_commit_pins/pytorch.txt
# by picking the hash from the same date in https://hud.pytorch.org/hud/pytorch/pytorch/nightly/
#
# NOTE: If you're changing, make the corresponding supported CUDA versions in
# SUPPORTED_CUDA_VERSIONS above if needed.
NIGHTLY_VERSION = "dev20250915"


def _check_cuda_enabled():
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


def _get_pytorch_cuda_url(cuda_version):
    """
    Get the appropriate PyTorch CUDA URL for the given CUDA version.

    Args:
        cuda_version: tuple of (major, minor) version numbers

    Returns:
        URL string for PyTorch CUDA packages
    """
    major, minor = cuda_version
    # Generate CUDA suffix (version validation is already done in _get_cuda_version)
    cuda_suffix = _cuda_version_to_pytorch_suffix(major, minor)

    return f"{TORCH_NIGHTLY_URL_BASE}/{cuda_suffix}"


# url for the PyTorch ExecuTorch depending on, which will be set by _determine_torch_url().
# please do not directly rely on it, but use _determine_torch_url() instead.
_torch_url = None


def _determine_torch_url():
    """
    Determine the appropriate PyTorch installation URL based on CUDA availability and CMAKE_ARGS.
    Uses caching to avoid redundant CUDA detection and print statements.

    Returns:
        URL string for PyTorch packages
    """
    global _torch_url

    # Return cached URL if already determined
    if _torch_url is not None:
        return _torch_url

    # Check if CUDA delegate is enabled
    if not _check_cuda_enabled():
        print("CUDA delegate not enabled, using CPU-only PyTorch")
        _torch_url = f"{TORCH_NIGHTLY_URL_BASE}/cpu"
        return _torch_url

    print("CUDA delegate enabled, detecting CUDA version...")

    # Get CUDA version
    cuda_version = _get_cuda_version()

    major, minor = cuda_version
    print(f"Detected CUDA version: {major}.{minor}")

    # Get appropriate PyTorch CUDA URL
    torch_url = _get_pytorch_cuda_url(cuda_version)
    print(f"Using PyTorch URL: {torch_url}")

    # Cache the result
    _torch_url = torch_url
    return torch_url


def install_requirements(use_pytorch_nightly):
    # Skip pip install on Intel macOS if using nightly.
    if use_pytorch_nightly and is_intel_mac_os():
        print(
            "ERROR: Prebuilt PyTorch wheels are no longer available for Intel-based macOS.\n"
            "Please build from source by following https://docs.pytorch.org/executorch/main/using-executorch-building-from-source.html",
            file=sys.stderr,
        )
        sys.exit(1)

    # Determine the appropriate PyTorch URL based on CUDA delegate status
    torch_url = _determine_torch_url()

    # pip packages needed by exir.
    TORCH_PACKAGE = [
        # Setting use_pytorch_nightly to false to test the pinned PyTorch commit. Note
        # that we don't need to set any version number there because they have already
        # been installed on CI before this step, so pip won't reinstall them
        f"torch==2.10.0.{NIGHTLY_VERSION}" if use_pytorch_nightly else "torch",
    ]

    # Install the requirements for core ExecuTorch package.
    # `--extra-index-url` tells pip to look for package
    # versions on the provided URL if they aren't available on the default URL.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            "requirements-dev.txt",
            *TORCH_PACKAGE,
            "--extra-index-url",
            torch_url,
        ],
        check=True,
    )

    LOCAL_REQUIREMENTS = [
        "third-party/ao",  # We need the latest kernels for fast iteration, so not relying on pypi.
    ] + (
        [
            "extension/llm/tokenizers",  # TODO(larryliu0820): Setup a pypi package for this.
        ]
        if sys.platform != "win32"
        else []
    )  # TODO(gjcomer): Re-enable when buildable on Windows.

    # Install packages directly from local copy instead of pypi.
    # This is usually not recommended.
    new_env = os.environ.copy()
    if ("EXECUTORCH_BUILD_KERNELS_TORCHAO" not in new_env) or (
        new_env["EXECUTORCH_BUILD_KERNELS_TORCHAO"] == "0"
    ):
        new_env["USE_CPP"] = "0"
    else:
        assert new_env["EXECUTORCH_BUILD_KERNELS_TORCHAO"] == "1"
        new_env["USE_CPP"] = "1"
        new_env["CMAKE_POLICY_VERSION_MINIMUM"] = "3.5"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            # Without --no-build-isolation, setup.py can't find the torch module.
            "--no-build-isolation",
            *LOCAL_REQUIREMENTS,
        ],
        env=new_env,
        check=True,
    )


def install_optional_example_requirements(use_pytorch_nightly):
    # Determine the appropriate PyTorch URL based on CUDA delegate status
    torch_url = _determine_torch_url()

    print("Installing torch domain libraries")
    DOMAIN_LIBRARIES = [
        (
            f"torchvision==0.25.0.{NIGHTLY_VERSION}"
            if use_pytorch_nightly
            else "torchvision"
        ),
        f"torchaudio==2.8.0.{NIGHTLY_VERSION}" if use_pytorch_nightly else "torchaudio",
    ]
    # Then install domain libraries
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            *DOMAIN_LIBRARIES,
            "--extra-index-url",
            torch_url,
        ],
        check=True,
    )

    print("Installing packages in requirements-examples.txt")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            "requirements-examples.txt",
            "--extra-index-url",
            torch_url,
            "--upgrade-strategy",
            "only-if-needed",
        ],
        check=True,
    )


# Prebuilt binaries for Intel-based macOS are no longer available on PyPI; users must compile from source.
# PyTorch stopped building macOS x86_64 binaries since version 2.3.0 (January 2024).
def is_intel_mac_os():
    # Returns True if running on Intel macOS.
    return platform.system().lower() == "darwin" and platform.machine().lower() in (
        "x86",
        "x86_64",
        "i386",
    )


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-pt-pinned-commit",
        action="store_true",
        help="build from the pinned PyTorch commit instead of nightly",
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Also installs required packages for running example scripts.",
    )
    args = parser.parse_args(args)
    use_pytorch_nightly = not bool(args.use_pt_pinned_commit)
    install_requirements(use_pytorch_nightly)
    if args.example:
        install_optional_example_requirements(use_pytorch_nightly)


if __name__ == "__main__":
    # Before doing anything, cd to the directory containing this script.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if not python_is_compatible():
        sys.exit(1)
    main(sys.argv[1:])
