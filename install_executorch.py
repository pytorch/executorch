# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-25 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import glob
import logging
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager

from install_requirements import (
    install_optional_example_requirements,
    install_requirements,
    python_is_compatible,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [ExecuTorch] %(levelname)s: %(message)s"
)
logger = logging.getLogger()


@contextmanager
def pushd(new_dir):
    """Change the current directory to new_dir and yield. When exiting the context, change back to the original directory."""
    original_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(original_dir)


def clean():
    print("Cleaning build artifacts...")
    print("Cleaning pip-out/...")
    shutil.rmtree("pip-out/", ignore_errors=True)
    dirs = glob.glob("cmake-out*/") + glob.glob("cmake-android-out/")
    for d in dirs:
        print(f"Cleaning {d}...")
        shutil.rmtree(d, ignore_errors=True)
    print("Cleaning buck-out/...")
    shutil.rmtree("buck-out/", ignore_errors=True)

    # Removes all buck cached state and metadata
    print("Cleaning buck cached state and metadata ...")
    shutil.rmtree(os.path.expanduser("~/.buck/buckd"), ignore_errors=True)

    # tokenizers build cleanup
    tokenizer_dirs = [
        "extension/llm/tokenizers/build",
        "extension/llm/tokenizers/pytorch_tokenizers.egg-info",
    ]

    for d in tokenizer_dirs:
        print(f"Cleaning {d}...")
        shutil.rmtree(d, ignore_errors=True)

    # Clean ccache if available
    try:
        result = subprocess.run(["ccache", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("Cleaning ccache...")
            subprocess.run(["ccache", "--clear"], check=True)
            print("ccache cleared successfully.")
        else:
            print("ccache not found, skipping ccache cleanup.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ccache not found, skipping ccache cleanup.")

    print("Done cleaning build artifacts.")


################################################################################
# Git submodules
################################################################################
# The following submodules are required to be able to build ExecuTorch. If any of
# these folders are missing or missing CMakeLists.txt, we will run
# `git submodule update` to try to fix it. If the command fails, we will raise an
# error.
# An alternative to this would be to run `git submodule status` and run
# `git submodule update` if there's any local changes. However this is a bit
# too restrictive for users who modifies and tests the dependencies locally.

# keep sorted
REQUIRED_SUBMODULES = {
    "ao": "LICENSE",  # No CMakeLists.txt, choose a sort of stable file to check.
    "cpuinfo": "CMakeLists.txt",
    "eigen": "CMakeLists.txt",
    "flatbuffers": "CMakeLists.txt",
    "FP16": "CMakeLists.txt",
    "FXdiv": "CMakeLists.txt",
    "gflags": "CMakeLists.txt",
    "prelude": "BUCK",
    "pthreadpool": "CMakeLists.txt",
    "pybind11": "CMakeLists.txt",
    "shim": "BUCK",
    "tokenizers": "CMakeLists.txt",
    "XNNPACK": "CMakeLists.txt",
}


def get_required_submodule_paths():
    gitmodules_path = os.path.join(os.getcwd(), ".gitmodules")

    if not os.path.isfile(gitmodules_path):
        logger.error(".gitmodules file not found.")
        exit(1)

    with open(gitmodules_path, "r") as file:
        lines = file.readlines()

    # Extract paths of required submodules
    required_paths = {}
    for line in lines:
        if line.strip().startswith("path ="):
            path = line.split("=")[1].strip()
            for submodule, file_name in REQUIRED_SUBMODULES.items():
                if submodule in path:
                    required_paths[path] = file_name
    return required_paths


def check_and_update_submodules():
    def check_folder(folder: str, file: str) -> bool:
        return os.path.isdir(folder) and os.path.isfile(os.path.join(folder, file))

    # Check if the directories exist for each required submodule
    missing_submodules = {}
    for path, file in get_required_submodule_paths().items():
        if not check_folder(path, file):
            missing_submodules[path] = file

    # If any required submodule directories are missing, update them
    if missing_submodules:
        logger.warning("Some required submodules are missing. Updating submodules...")
        try:
            subprocess.check_call(["git", "submodule", "sync", "--recursive"])
            subprocess.check_call(
                ["git", "submodule", "update", "--init", "--recursive"]
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Error updating submodules: {e}")
            exit(1)

        # After updating submodules, check again
        for path, file in missing_submodules.items():
            if not check_folder(path, file):
                logger.error(f"{file} not found in {path}.")
                logger.error(
                    "Submodule update failed. Please run `git submodule update --init --recursive` manually."
                )
                exit(1)
    logger.info("All required submodules are present.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install executorch in your Python environment."
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
    parser.add_argument(
        "--editable",
        "-e",
        action="store_true",
        help="build an editable pip wheel, changes to python code will be "
        "picked up without rebuilding the wheel. Extension libraries will be "
        "installed inside the source tree.",
    )
    parser.add_argument(
        "--minimal",
        "-m",
        action="store_true",
        help="Only installs necessary dependencies for core executorch and skips "
        " packages necessary for running example scripts.",
    )
    return parser.parse_args()


def main(args):
    if not python_is_compatible():
        sys.exit(1)

    args = _parse_args()

    if args.clean:
        clean()
        return

    check_and_update_submodules()
    # This option is used in CI to make sure that PyTorch build from the pinned commit
    # is used instead of nightly. CI jobs wouldn't be able to catch regression from the
    # latest PT commit otherwise
    use_pytorch_nightly = not args.use_pt_pinned_commit

    # Step 1: Install core dependencies first
    install_requirements(use_pytorch_nightly)

    # Step 2: Install core package
    cmd = (
        [
            sys.executable,
            "-m",
            "pip",
            "install",
        ]
        + (["--editable"] if args.editable else [])
        + [
            ".",
            "--no-build-isolation",
            "-v",
        ]
    )
    subprocess.run(cmd, check=True)

    # Step 3: Extra (optional) packages that is only useful for running examples.
    if not args.minimal:
        install_optional_example_requirements(use_pytorch_nightly)


if __name__ == "__main__":
    # Before doing anything, cd to the directory containing this script.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main(sys.argv[1:])
