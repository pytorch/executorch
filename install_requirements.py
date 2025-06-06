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
TORCH_NIGHTLY_URL = "https://download.pytorch.org/whl/nightly/cpu"


# Since ExecuTorch often uses main-branch features of pytorch, only the nightly
# pip versions will have the required features.
#
# NOTE: If a newly-fetched version of the executorch repo changes the value of
# NIGHTLY_VERSION, you should re-run this script to install the necessary
# package versions.
#
# NOTE: If you're changing, make the corresponding change in .ci/docker/ci_commit_pins/pytorch.txt
# by picking the hash from the same date in https://hud.pytorch.org/hud/pytorch/pytorch/nightly/
NIGHTLY_VERSION = "dev20250601"


def install_requirements(use_pytorch_nightly):
    # pip packages needed by exir.
    EXIR_REQUIREMENTS = [
        # Setting use_pytorch_nightly to false to test the pinned PyTorch commit. Note
        # that we don't need to set any version number there because they have already
        # been installed on CI before this step, so pip won't reinstall them
        f"torch==2.8.0.{NIGHTLY_VERSION}" if use_pytorch_nightly else "torch",
        (
            f"torchvision==0.23.0.{NIGHTLY_VERSION}"
            if use_pytorch_nightly
            else "torchvision"
        ),  # For testing.
    ]

    EXAMPLES_REQUIREMENTS = [
        f"torchaudio==2.8.0.{NIGHTLY_VERSION}" if use_pytorch_nightly else "torchaudio",
    ]

    # Assemble the list of requirements to actually install.
    # TODO: Add options for reducing the number of requirements.
    REQUIREMENTS_TO_INSTALL = EXIR_REQUIREMENTS + EXAMPLES_REQUIREMENTS

    # Install the requirements. `--extra-index-url` tells pip to look for package
    # versions on the provided URL if they aren't available on the default URL.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            "requirements-examples.txt",
            "-r",
            "requirements-dev.txt",
            *REQUIREMENTS_TO_INSTALL,
            "--extra-index-url",
            TORCH_NIGHTLY_URL,
        ],
        check=True,
    )

    LOCAL_REQUIREMENTS = [
        "third-party/ao",  # We need the latest kernels for fast iteration, so not relying on pypi.
        "extension/llm/tokenizers",  # TODO(larryliu0820): Setup a pypi package for this.
    ]

    # Install packages directly from local copy instead of pypi.
    # This is usually not recommended.
    new_env = os.environ.copy()
    new_env["USE_CPP"] = "1"  # install torchao kernels
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


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-pt-pinned-commit",
        action="store_true",
        help="build from the pinned PyTorch commit instead of nightly",
    )
    args = parser.parse_args(args)
    install_requirements(use_pytorch_nightly=not bool(args.use_pt_pinned_commit))


if __name__ == "__main__":
    # Before doing anything, cd to the directory containing this script.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if not python_is_compatible():
        sys.exit(1)
    main(sys.argv[1:])
