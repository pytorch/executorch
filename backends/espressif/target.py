# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum


class EspTarget(str, Enum):
    ESP32 = "esp32"
    ESP32S2 = "esp32s2"
    ESP32S3 = "esp32s3"
    ESP32P4 = "esp32p4"
    ESP32S31 = "esp32s31"
    ESP32C2 = "esp32c2"
    ESP32C3 = "esp32c3"
    ESP32C5 = "esp32c5"
    ESP32C6 = "esp32c6"
    ESP32C61 = "esp32c61"
    ESP32H2 = "esp32h2"
    ESP32H21 = "esp32h21"
    ESP32H4 = "esp32h4"


class EspKernelMode(str, Enum):
    OPTIMIZED = "optimized"
    ANSI = "ansi"
