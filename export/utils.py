# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import logging
import platform
import sys

import torch


def is_fbcode() -> bool:
    return not hasattr(torch.version, "git_version")


# Check if lowering for CoreML is supported on the current platform
def is_supported_platform_for_coreml_lowering() -> bool:
    system = platform.system()
    machine = platform.machine().lower()

    # coremltools has no wheel for 3.14 yet, so setup.py does not declare it
    # there. Callers use this as an import guard, so reporting the platform as
    # supported would turn a handled "not supported" into a ModuleNotFoundError.
    # Keep this in step with the coremltools marker in setup.py.
    if sys.version_info >= (3, 14):
        logging.info(
            f"Unsupported Python for CoreML: {sys.version_info.major}."
            f"{sys.version_info.minor}"
        )
        return False

    # Check for Linux x86_64
    if system == "Linux" and machine == "x86_64":
        return True

    # Check for macOS aarch64
    if system == "Darwin" and machine in ("arm64", "aarch64"):
        return True

    logging.info(f"Unsupported platform: {system} {machine}")

    return False


# Check if lowering for QNN is supported on the current platform
def is_supported_platform_for_qnn_lowering() -> bool:
    system = platform.system()
    machine = platform.machine().lower()

    # Check for Linux x86_64
    if platform.system().lower() == "linux" and platform.machine().lower() in (
        "x86_64",
        "amd64",
        "i386",
        "i686",
    ):
        return True

    logging.error(f"Unsupported platform for QNN lowering: {system} {machine}")
    return False
