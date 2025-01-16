# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-25 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import glob
import itertools
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


def clean():
    print("Cleaning build artifacts...")
    print("Cleaning pip-out/...")
    shutil.rmtree("pip-out/", ignore_errors=True)
    dirs = glob.glob("cmake-out*/") + glob.glob("cmake-android-out/")
    for d in dirs:
        print(f"Cleaning {d}...")
        shutil.rmtree(d, ignore_errors=True)
    print("Done cleaning build artifacts.")


VALID_PYBINDS = ["coreml", "mps", "xnnpack"]


# The pip repository that hosts nightly torch packages.
TORCH_NIGHTLY_URL = "https://download.pytorch.org/whl/nightly/cpu"


# Since ExecuTorch often uses main-branch features of pytorch, only the nightly
# pip versions will have the required features.
#
# NOTE: If a newly-fetched version of the executorch repo changes the value of
# NIGHTLY_VERSION, you should re-run this script to install the necessary
# package versions.
NIGHTLY_VERSION = "dev20250104"


def install_requirements(use_pytorch_nightly):
    # pip packages needed by exir.
    EXIR_REQUIREMENTS = [
        # Setting use_pytorch_nightly to false to test the pinned PyTorch commit. Note
        # that we don't need to set any version number there because they have already
        # been installed on CI before this step, so pip won't reinstall them
        f"torch==2.6.0.{NIGHTLY_VERSION}" if use_pytorch_nightly else "torch",
        (
            f"torchvision==0.22.0.{NIGHTLY_VERSION}"
            if use_pytorch_nightly
            else "torchvision"
        ),  # For testing.
        "typing-extensions",
    ]

    # pip packages needed to run examples.
    # TODO: Make each example publish its own requirements.txt
    EXAMPLES_REQUIREMENTS = [
        "timm==1.0.7",
        f"torchaudio==2.6.0.{NIGHTLY_VERSION}" if use_pytorch_nightly else "torchaudio",
        "torchsr==1.0.4",
        "transformers==4.47.1",
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
        "ai-edge-model-explorer>=0.1.16",  # For visualizing ExportedPrograms
    ]

    # Assemble the list of requirements to actually install.
    # TODO: Add options for reducing the number of requirements.
    REQUIREMENTS_TO_INSTALL = (
        EXIR_REQUIREMENTS + DEVEL_REQUIREMENTS + EXAMPLES_REQUIREMENTS
    )

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

    LOCAL_REQUIREMENTS = [
        "third-party/ao",  # We need the latest kernels for fast iteration, so not relying on pypi.
    ]

    # Install packages directly from local copy instead of pypi.
    # This is usually not recommended.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            *LOCAL_REQUIREMENTS,
        ],
        check=True,
    )


def main(args):
    if not python_is_compatible():
        sys.exit(1)

    # Parse options.

    EXECUTORCH_BUILD_PYBIND = ""
    CMAKE_ARGS = os.getenv("CMAKE_ARGS", "")
    CMAKE_BUILD_ARGS = os.getenv("CMAKE_BUILD_ARGS", "")
    use_pytorch_nightly = True

    parser = argparse.ArgumentParser(prog="install_requirements")
    parser.add_argument(
        "--pybind",
        action="append",
        nargs="+",
        help="one or more of coreml/mps/xnnpack, or off",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="clean build artifacts and pip-out instead of installing",
    )
    parser.add_argument(
        "--use-pt-pinned-commit",
        action="store_true",
        help="build from the pinned PyTorch commit instead of nightly",
    )
    args = parser.parse_args(args)
    if args.pybind:
        # Flatten list of lists.
        args.pybind = list(itertools.chain(*args.pybind))
        if "off" in args.pybind:
            if len(args.pybind) != 1:
                raise Exception(
                    f"Cannot combine `off` with other pybinds: {args.pybind}"
                )
            EXECUTORCH_BUILD_PYBIND = "OFF"
        else:
            for pybind_arg in args.pybind:
                if pybind_arg not in VALID_PYBINDS:
                    raise Exception(
                        f"Unrecognized pybind argument {pybind_arg}; valid options are: {", ".join(VALID_PYBINDS)}"
                    )
                EXECUTORCH_BUILD_PYBIND = "ON"
                CMAKE_ARGS += f" -DEXECUTORCH_BUILD_{pybind_arg.upper()}=ON"

    if args.clean:
        clean()
        return

    if args.use_pt_pinned_commit:
        # This option is used in CI to make sure that PyTorch build from the pinned commit
        # is used instead of nightly. CI jobs wouldn't be able to catch regression from the
        # latest PT commit otherwise
        use_pytorch_nightly = False

    install_requirements(use_pytorch_nightly)

    # If --pybind is not set explicitly for backends (e.g., --pybind xnnpack)
    # or is not turned off explicitly (--pybind off)
    # then install XNNPACK by default.
    if EXECUTORCH_BUILD_PYBIND == "":
        EXECUTORCH_BUILD_PYBIND = "ON"
        CMAKE_ARGS += " -DEXECUTORCH_BUILD_XNNPACK=ON"

    # Use ClangCL on Windows.
    # ClangCL is an alias to Clang that configures it to work in an MSVC-compatible
    # mode. Using it on Windows to avoid compiler compatibility issues for MSVC.
    if os.name == "nt":
        CMAKE_ARGS += " -T ClangCL"

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


if __name__ == "__main__":
    main(sys.argv[1:])
