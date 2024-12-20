# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import glob
import os
import platform
import re
import shutil
import subprocess
import sys

# Before doing anything, cd to the directory containing this script.
os.chdir(os.path.dirname(os.path.abspath(__file__)))


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


if not python_is_compatible():
    sys.exit(1)

# Parse options.
EXECUTORCH_BUILD_PYBIND = "OFF"
CMAKE_ARGS = os.getenv("CMAKE_ARGS", "")
CMAKE_BUILD_ARGS = os.getenv("CMAKE_BUILD_ARGS", "")
USE_PYTORCH_NIGHTLY = True

for arg in sys.argv[1:]:
    if arg == "--pybind":
        EXECUTORCH_BUILD_PYBIND = "ON"
    elif arg in ["coreml", "mps", "xnnpack"]:
        if EXECUTORCH_BUILD_PYBIND == "ON":
            arg_upper = arg.upper()
            CMAKE_ARGS += f" -DEXECUTORCH_BUILD_{arg_upper}=ON"
        else:
            print(f"Error: {arg} must follow --pybind")
            sys.exit(1)
    elif arg == "--clean":
        print("Cleaning build artifacts...")
        print("Cleaning pip-out/...")
        shutil.rmtree("pip-out/", ignore_errors=True)
        dirs = glob.glob("cmake-out*/") + glob.glob("cmake-android-out/")
        for d in dirs:
            print(f"Cleaning {d}...")
            shutil.rmtree(d, ignore_errors=True)
        print("Done cleaning build artifacts.")
        sys.exit(0)
    elif arg == "--use-pt-pinned-commit":
        # This option is used in CI to make sure that PyTorch build from the pinned commit
        # is used instead of nightly. CI jobs wouldn't be able to catch regression from the
        # latest PT commit otherwise
        USE_PYTORCH_NIGHTLY = False
    else:
        print(f"Error: Unknown option {arg}")
        sys.exit(1)

# Use ClangCL on Windows.
# ClangCL is an alias to Clang that configures it to work in an MSVC-compatible
# mode. Using it on Windows to avoid compiler compatibility issues for MSVC.
if os.name == "nt":
    CMAKE_ARGS += " -T ClangCL"

# Since ExecuTorch often uses main-branch features of pytorch, only the nightly
# pip versions will have the required features.
#
# NOTE: If a newly-fetched version of the executorch repo changes the value of
# NIGHTLY_VERSION, you should re-run this script to install the necessary
# package versions.
NIGHTLY_VERSION = "dev20241218"

# The pip repository that hosts nightly torch packages.
TORCH_NIGHTLY_URL = "https://download.pytorch.org/whl/nightly/cpu"

# pip packages needed by exir.
EXIR_REQUIREMENTS = [
    # Setting USE_PYTORCH_NIGHTLY to false to test the pinned PyTorch commit. Note
    # that we don't need to set any version number there because they have already
    # been installed on CI before this step, so pip won't reinstall them
    f"torch==2.6.0.{NIGHTLY_VERSION}" if USE_PYTORCH_NIGHTLY else "torch",
    (
        f"torchvision==0.22.0.{NIGHTLY_VERSION}"
        if USE_PYTORCH_NIGHTLY
        else "torchvision"
    ),  # For testing.
    "typing-extensions",
]

# pip packages needed to run examples.
# TODO: Make each example publish its own requirements.txt
EXAMPLES_REQUIREMENTS = [
    "timm==1.0.7",
    f"torchaudio==2.6.0.{NIGHTLY_VERSION}" if USE_PYTORCH_NIGHTLY else "torchaudio",
    "torchsr==1.0.4",
    "transformers==4.46.1",
]

# pip packages needed for development.
DEVEL_REQUIREMENTS = [
    "cmake",  # For building binary targets.
    "pip>=23",  # For building the pip package.
    "pyyaml",  # Imported by the kernel codegen tools.
    "setuptools>=63",  # For building the pip package.
    "tomli",  # Imported by extract_sources.py when using python < 3.11.
    "wheel",  # For building the pip package archive.
    "zstd",  # Imported by resolve_buck.py.
]

# Assemble the list of requirements to actually install.
# TODO: Add options for reducing the number of requirements.
REQUIREMENTS_TO_INSTALL = EXIR_REQUIREMENTS + DEVEL_REQUIREMENTS + EXAMPLES_REQUIREMENTS

# Install the requirements. `--extra-index-url` tells pip to look for package
# versions on the provided URL if they aren't available on the default URL.
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        *REQUIREMENTS_TO_INSTALL,
        "--extra-index-url",
        TORCH_NIGHTLY_URL,
    ],
    check=True,
)

#
# Install executorch pip package. This also makes `flatc` available on the path.
# The --extra-index-url may be necessary if pyproject.toml has a dependency on a
# pre-release or nightly version of a torch package.
#

# Set environment variables
os.environ["EXECUTORCH_BUILD_PYBIND"] = EXECUTORCH_BUILD_PYBIND
os.environ["CMAKE_ARGS"] = CMAKE_ARGS
os.environ["CMAKE_BUILD_ARGS"] = CMAKE_BUILD_ARGS

# Run the pip install command
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        ".",
        "--no-build-isolation",
        "-v",
        "--extra-index-url",
        TORCH_NIGHTLY_URL,
    ],
    check=True,
)
