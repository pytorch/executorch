# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-25 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import subprocess
import sys

from install_utils import determine_torch_url, is_intel_mac_os, python_is_compatible

# The pip repository that hosts nightly torch packages.
# This will be dynamically set based on CUDA availability and CUDA backend enabled/disabled.
TORCH_NIGHTLY_URL_BASE = "https://download.pytorch.org/whl/nightly"

# Supported CUDA versions - modify this to add/remove supported versions
# Format: tuple of (major, minor) version numbers
SUPPORTED_CUDA_VERSIONS = (
    (12, 6),
    (12, 8),
    (12, 9),
)

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
    torch_url = determine_torch_url(TORCH_NIGHTLY_URL_BASE, SUPPORTED_CUDA_VERSIONS)

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
    torch_url = determine_torch_url(TORCH_NIGHTLY_URL_BASE, SUPPORTED_CUDA_VERSIONS)

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
