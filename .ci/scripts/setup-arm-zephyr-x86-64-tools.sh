#!/bin/bash
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# NB: This function could be used to install Arm dependencies
# Setup arm example environment (including TOSA tools)
git config --global user.email "github_executorch@arm.com"
git config --global user.name "Github Executorch"
bash examples/arm/setup.sh --i-agree-to-the-contained-eula --user-toolchain-url "https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v0.17.2/toolchain_linux-x86_64_arm-zephyr-eabi.tar.xz" --user-toolchain-dir "arm-zephyr-eabi"
