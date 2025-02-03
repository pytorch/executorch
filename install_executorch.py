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
import shutil
import subprocess
import sys

from install_requirements import (
    install_requirements,
    python_is_compatible,
    TORCH_NIGHTLY_URL,
)


def clean():
    print("Cleaning build artifacts...")
    print("Cleaning pip-out/...")
    shutil.rmtree("pip-out/", ignore_errors=True)
    dirs = glob.glob("cmake-out*/") + glob.glob("cmake-android-out/")
    for d in dirs:
        print(f"Cleaning {d}...")
        shutil.rmtree(d, ignore_errors=True)
    print("Done cleaning build artifacts.")


VALID_PYBINDS = ["coreml", "mps", "xnnpack", "training"]


def main(args):
    if not python_is_compatible():
        sys.exit(1)

    # Parse options.

    EXECUTORCH_BUILD_PYBIND = ""
    CMAKE_ARGS = os.getenv("CMAKE_ARGS", "")
    use_pytorch_nightly = True

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
                        f"Unrecognized pybind argument {pybind_arg}; valid options are: {', '.join(VALID_PYBINDS)}"
                    )
                if pybind_arg == "training":
                    CMAKE_ARGS += " -DEXECUTORCH_BUILD_EXTENSION_TRAINING=ON"
                    os.environ["EXECUTORCH_BUILD_TRAINING"] = "ON"
                else:
                    CMAKE_ARGS += f" -DEXECUTORCH_BUILD_{pybind_arg.upper()}=ON"
                EXECUTORCH_BUILD_PYBIND = "ON"

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
    # Before doing anything, cd to the directory containing this script.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if not python_is_compatible():
        sys.exit(1)

    main(sys.argv[1:])
