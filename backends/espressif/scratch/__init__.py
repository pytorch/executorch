# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
from types import ModuleType

from executorch.backends.espressif.target import EspKernelMode, EspTarget


def get_scratch_queries(
    target: EspTarget, kernel_mode: EspKernelMode
) -> ModuleType | None:
    """Load the vendor getters selected by ESP-NN 1.5.0's device configuration."""
    target = EspTarget(target)
    kernel_mode = EspKernelMode(kernel_mode)
    if kernel_mode == EspKernelMode.ANSI:
        profile = "ansi"
    else:
        profile = {
            EspTarget.ESP32S3: "esp32s3",
            EspTarget.ESP32P4: "riscv_pie",
            EspTarget.ESP32S31: "riscv_pie",
        }.get(target, "generic")
    try:
        return importlib.import_module(
            f"executorch.backends.espressif._esp_nn_scratch_{profile}"
        )
    except ImportError:
        return None
