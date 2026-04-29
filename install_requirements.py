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

from torch_pin import (
    CHANNEL,
    torch_index_url_base,
    torch_spec,
    torchaudio_spec,
    torchvision_spec,
)

# Only RC wheels at /whl/test/ get re-uploaded under the same version, so
# pip's local cache can serve stale content. Nightly and release wheels are
# immutable per their identifier.
_NO_CACHE_DIR_FLAG = ["--no-cache-dir"] if CHANNEL == "test" else []

# Since ExecuTorch often uses main-branch features of pytorch, only the nightly
# pip versions will have the required features.
#
# NOTE: If a newly-fetched version of the executorch repo changes the value of
# NIGHTLY_VERSION, you should re-run this script to install the necessary
# package versions.
#
# NOTE: If you change torch_pin.py, the pre-commit hook runs
# .github/scripts/update_pytorch_pin.py to refresh
# .ci/docker/ci_commit_pins/pytorch.txt and the c10 grafted headers.
# If you bypass the hook, run that script manually.
#
# NOTE: If you're changing, make the corresponding supported CUDA versions in
# SUPPORTED_CUDA_VERSIONS in install_utils.py if needed.


def install_requirements(install_pinned_version):
    # No prebuilt wheels are available for Intel macOS, regardless of channel.
    if install_pinned_version and is_intel_mac_os():
        print(
            "ERROR: Prebuilt PyTorch wheels are no longer available for Intel-based macOS.\n"
            "Please build from source by following https://docs.pytorch.org/executorch/main/using-executorch-building-from-source.html",
            file=sys.stderr,
        )
        sys.exit(1)

    # Determine the appropriate PyTorch URL based on CUDA delegate status
    torch_url = determine_torch_url(torch_index_url_base())

    # pip packages needed by exir.
    TORCH_PACKAGE = [
        # Default: install the specific pinned version from the channel selected
        # in torch_pin.py. With --use-pt-pinned-commit, pass plain "torch" and
        # let pip resolve its default (CI's source-build is already installed).
        (torch_spec() if install_pinned_version else "torch"),
    ]

    # Install the requirements for core ExecuTorch package.
    # `--extra-index-url` tells pip to look for package versions on the
    # provided URL if they aren't available on the default URL.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            *_NO_CACHE_DIR_FLAG,
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


def install_optional_example_requirements(install_pinned_version):
    # Determine the appropriate PyTorch URL based on CUDA delegate status
    torch_url = determine_torch_url(torch_index_url_base())

    print("Installing torch domain libraries")
    DOMAIN_LIBRARIES = [
        (torchvision_spec() if install_pinned_version else "torchvision"),
        (torchaudio_spec() if install_pinned_version else "torchaudio"),
    ]
    # Then install domain libraries
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            *_NO_CACHE_DIR_FLAG,
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
        help="install plain `torch` (whatever pip resolves by default; CI "
        "uses this when torch is already built from source against the "
        "pinned ref in pytorch.txt). Without this flag, install the specific "
        "pinned version from the channel selected in torch_pin.py "
        "(nightly / test / release).",
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Also installs required packages for running example scripts.",
    )
    args = parser.parse_args(args)
    install_pinned_version = not bool(args.use_pt_pinned_commit)
    install_requirements(install_pinned_version)
    if args.example:
        install_optional_example_requirements(install_pinned_version)


if __name__ == "__main__":
    # Before doing anything, cd to the directory containing this script.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if not python_is_compatible():
        sys.exit(1)
    main(sys.argv[1:])
