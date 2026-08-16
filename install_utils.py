# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-25 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from typing import List, Optional

# Supported CUDA versions - modify this to add/remove supported versions
# Format: tuple of (major, minor) version numbers
SUPPORTED_CUDA_VERSIONS = (
    (12, 6),
    (13, 0),
    (13, 2),
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


def is_vulkan_available() -> bool:
    """
    Check if the Vulkan shader compiler (glslc) is available on the system.

    glslc is the only build-time dependency for the Vulkan backend; the Vulkan
    loader itself is dlopen()ed at runtime via volk. Restricted to Linux and
    Windows, the desktop GPU platforms the backend supports (macOS would require
    MoltenVK).

    glslc is accepted from PATH or under $VULKAN_SDK/{bin,Bin}. The Windows
    Vulkan SDK sets VULKAN_SDK but does not add its bin directory to PATH, so a
    PATH-only probe would miss it there.

    Returns:
        True if glslc is available on a supported platform, False otherwise.
    """
    if sys.platform not in ("linux", "win32"):
        return False
    candidates = ["glslc"]
    vulkan_sdk = os.environ.get("VULKAN_SDK")
    if vulkan_sdk:
        glslc_name = "glslc.exe" if sys.platform == "win32" else "glslc"
        candidates += [
            os.path.join(vulkan_sdk, "bin", glslc_name),
            os.path.join(vulkan_sdk, "Bin", glslc_name),
        ]
    for glslc in candidates:
        try:
            # Only the exit status matters, so skip text=True; keep the except
            # tight to avoid masking things like UnicodeDecodeError.
            subprocess.run([glslc, "--version"], capture_output=True, check=True)
            return True
        except (OSError, subprocess.SubprocessError):
            # glslc missing or not runnable -> unavailable; try the next candidate.
            continue
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
        # Same selection rule as _detected_cuda_major, so the two cannot disagree about which
        # toolkit this build uses.
        nvcc_result = subprocess.run(
            _selected_nvcc(), capture_output=True, text=True, check=True
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


def _selected_nvcc() -> List[str]:
    """The nvcc command line to ask for a version, matching what the build will use.

    Reading the bare command described whichever toolkit was on PATH, while the build also honours
    these variables. Packaging then declared the runtime for one toolkit while compiling against
    another, or declared nothing at all when the selected compiler was not on PATH.
    """
    # CMake reads -DCMAKE_CUDA_COMPILER from the command line and CUDACXX from the environment. It does
    # NOT read an environment variable named CMAKE_CUDA_COMPILER, measured with cmake 3.31.8, so asking
    # the environment for that name first described a compiler the build would never use.
    explicit = _extract_cmake_define(_cmake_args_from_env(), "CMAKE_CUDA_COMPILER")
    if not explicit:
        explicit = os.environ.get("CUDACXX")
    if explicit:
        # Only the program, because CMake accepts trailing options here and splits them off:
        # measured, CUDACXX="/path/nvcc -allow-unsupported-compiler" still compiles with
        # /path/nvcc. Passing the whole value as one program name found nothing, so a build
        # that compiles fine reported no toolkit at all. That option is a common workaround
        # on a newer host compiler, so the value is not unusual.
        #
        # Splitting also matches CMake on an unquoted path containing a space, which CMake
        # reports as NOTFOUND rather than treating as one name, so this reads nothing exactly
        # where the build would also refuse to compile.
        program = shlex.split(explicit) or [explicit]
        return [program[0], "--version"]
    # Follow CMake's COMPILER search, since that is what decides which nvcc compiles the sources:
    # CUDACXX above, then PATH, then CUDA_PATH. PATH has to come before CUDA_PATH. Measured with
    # only PATH pointing at 13.0 on a box whose conventional symlink is 12.8, CMake compiles with
    # 13.0, so consulting the symlink first reported 12.8 and produced metadata for a train the
    # binaries were not built with.
    #
    # The list stops at CUDA_PATH because CMake's compiler search does:
    # CMakeDetermineCUDACompiler.cmake sets its search paths to CUDA_PATH/bin and nothing else.
    # Adding the conventional /usr/local/cuda symlink reported a toolkit on a box where CMake finds
    # no compiler at all, which drops the CUDA sources silently, and the guard meant to catch that
    # reads this same value.
    #
    # CUDAToolkit_ROOT is deliberately absent: it steers find_package(CUDAToolkit) but CMake's
    # compiler search ignores it, so reading it here names a compiler that will not be used.
    on_path = shutil.which("nvcc")
    if on_path:
        return [on_path, "--version"]
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidate = os.path.join(cuda_path, "bin", "nvcc")
        if os.path.exists(candidate):
            return [candidate, "--version"]
    return ["nvcc", "--version"]


def _cmake_args_from_env() -> List[str]:
    """CMAKE_ARGS split into arguments, tolerating an unbalanced quote.

    shlex is the right parser for a value naming a shell argument list, but it raises on an unbalanced
    quote, and a path containing an apostrophe is enough to trigger it.
    """
    raw = os.environ.get("CMAKE_ARGS", "")
    try:
        return shlex.split(raw)
    except ValueError:
        return raw.split()


@functools.lru_cache(maxsize=1)
def _detected_cuda_major() -> Optional[int]:
    """The CUDA major version of the installed toolkit, or None if none is installed.

    Kept separate from `_get_cuda_version` because the mismatch guard in the wheel build
    needs the major regardless of whether the exact (major, minor) is listed in
    SUPPORTED_CUDA_VERSIONS. Reading through the validator caused the guard to see an
    empty detection for any unlisted minor (say 12.8), so a cu130 row built on a CUDA 12
    toolkit produced a wheel with no error.
    """
    try:
        result = subprocess.run(
            _selected_nvcc(), capture_output=True, text=True, check=True
        )
    except (subprocess.CalledProcessError, OSError):
        return None
    match = re.search(r"release (\d+)\.\d+", result.stdout)
    return int(match.group(1)) if match else None


def _extract_cmake_define(args: List[str], name: str) -> Optional[str]:
    """The value CMake would use for -D<name>, which is the last one given.

    Repeating a definition is how a caller overrides an earlier one, and CMake keeps the last, so returning
    the first would let packaging read one value while the build used another.

    All three spellings CMake accepts are matched, because it treats them identically: -D<name>=<value>,
    -D<name>:<type>=<value>, and -D followed by <name>=<value> as a separate argument. Matching only the
    first meant a caller who switched an option off in either of the other two forms was read as leaving it
    on, so a CPU row could ship a wheel carrying CUDA.
    """
    # A bare -D takes its definition from the next argument, so both spellings collapse to one form.
    definitions = []
    remaining = iter(args)
    for arg in remaining:
        if arg == "-D":
            definitions.append(next(remaining, ""))
        elif arg.startswith("-D"):
            definitions.append(arg[2:])

    # The name may carry a CMake type, as in EXECUTORCH_BUILD_CUDA:BOOL.
    pattern = re.compile(rf"{re.escape(name)}(?::\w+)?=(.*)", re.DOTALL)
    value = None
    for definition in definitions:
        match = pattern.fullmatch(definition)
        if match:
            value = match.group(1)
    return value


def _normalize_cmake_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().upper()
    # Deliberately stricter than CMake. CMake decides false by exclusion, so anything that is not one
    # of its false constants is true, including values like "2.0" and "enabled". Here an unrecognised
    # spelling reads as off, because this decides whether a component's libraries are packaged and
    # shipping a component whose libraries were never built is worse than shipping one fewer.
    if normalized in {"ON", "TRUE", "YES", "Y"}:
        return True
    try:
        return int(normalized) != 0
    except ValueError:
        return False


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
