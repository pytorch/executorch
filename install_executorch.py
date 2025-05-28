# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-25 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import glob
import itertools
import logging
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from typing import List, Tuple

from install_requirements import (
    install_requirements,
    python_is_compatible,
    TORCH_NIGHTLY_URL,
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
    print("Done cleaning build artifacts.")


# Please keep this insync with `ShouldBuild.pybindings` in setup.py.
VALID_PYBINDS = ["coreml", "mps", "xnnpack", "training", "openvino"]


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


def build_args_parser() -> argparse.ArgumentParser:
    # Parse options.
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--editable",
        "-e",
        action="store_true",
        help="build an editable pip wheel, changes to python code will be "
        "picked up without rebuilding the wheel. Extension libraries will be "
        "installed inside the source tree.",
    )
    return parser


# Returns (wants_off, wanted_pybindings)
def _list_pybind_defines(args) -> Tuple[bool, List[str]]:
    if args.pybind is None:
        return False, []

    # Flatten list of lists.
    args.pybind = list(itertools.chain(*args.pybind))
    if "off" in args.pybind:
        if len(args.pybind) != 1:
            raise Exception(f"Cannot combine `off` with other pybinds: {args.pybind}")
        return True, []

    cmake_args = []
    for pybind_arg in args.pybind:
        if pybind_arg not in VALID_PYBINDS:
            raise Exception(
                f"Unrecognized pybind argument {pybind_arg}; valid options are: {', '.join(VALID_PYBINDS)}"
            )
        if pybind_arg == "training":
            cmake_args.append("-DEXECUTORCH_BUILD_EXTENSION_TRAINING=ON")
        else:
            cmake_args.append(f"-DEXECUTORCH_BUILD_{pybind_arg.upper()}=ON")
            if pybind_arg == "xnnpack":
                cmake_args.append("-DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON")

    return False, cmake_args


def main(args):
    if not python_is_compatible():
        sys.exit(1)

    parser = build_args_parser()
    args = parser.parse_args()

    cmake_args = [os.getenv("CMAKE_ARGS", "")]
    use_pytorch_nightly = True

    wants_pybindings_off, pybind_defines = _list_pybind_defines(args)
    if wants_pybindings_off:
        cmake_args.append("-DEXECUTORCH_BUILD_PYBIND=OFF")
    else:
        cmake_args += pybind_defines

    if args.clean:
        clean()
        return

    if args.use_pt_pinned_commit:
        # This option is used in CI to make sure that PyTorch build from the pinned commit
        # is used instead of nightly. CI jobs wouldn't be able to catch regression from the
        # latest PT commit otherwise
        use_pytorch_nightly = False

    # Use ClangCL on Windows.
    # ClangCL is an alias to Clang that configures it to work in an MSVC-compatible
    # mode. Using it on Windows to avoid compiler compatibility issues for MSVC.
    if os.name == "nt":
        cmake_args.append("-T ClangCL")

    #
    # Install executorch pip package. This also makes `flatc` available on the path.
    # The --extra-index-url may be necessary if pyproject.toml has a dependency on a
    # pre-release or nightly version of a torch package.
    #

    # Set environment variables
    os.environ["CMAKE_ARGS"] = " ".join(cmake_args)

    # Check if the required submodules are present and update them if not
    check_and_update_submodules()

    install_requirements(use_pytorch_nightly)

    # Run the pip install command
    subprocess.run(
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
            "--extra-index-url",
            TORCH_NIGHTLY_URL,
        ],
        check=True,
    )


if __name__ == "__main__":
    # Before doing anything, cd to the directory containing this script.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if not python_is_compatible():
        sys.exit(1)

    main(sys.argv[1:])
