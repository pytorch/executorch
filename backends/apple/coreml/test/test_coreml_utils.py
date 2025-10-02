# Copyright © 2025 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.

import platform
import sys

import torch


def is_fbcode():
    return not hasattr(torch.version, "git_version")


IS_VALID_TEST_RUNTIME: bool = (
    (sys.platform == "darwin")
    and not is_fbcode()
    and tuple(map(int, platform.mac_ver()[0].split("."))) >= (15, 0)
)
