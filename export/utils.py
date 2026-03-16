# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import logging
import platform

import torch


def is_fbcode() -> bool:
    return not hasattr(torch.version, "git_version")


# Check if lowering for CoreML is supported on the current platform
def is_supported_platform_for_coreml_lowering() -> bool:
    system = platform.system()
    machine = platform.machine().lower()

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
