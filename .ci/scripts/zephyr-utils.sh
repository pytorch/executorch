#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

download_arm_zephyr_sdk () {
    wget https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v0.17.2/zephyr-sdk-0.17.2_linux-x86_64.tar.xz
    tar -xf zephyr-sdk-0.17.2_linux-x86_64.tar.xz
    rm -f zephyr-sdk-0.17.2_linux-x86_64.tar.xz
}

setup_zephyr_et_module () {
    git clone --branch executorch-module-integration https://github.com/BujSet/zephyr.git
    west init -l zephyr
    west config manifest.project-filter -- +executorch
    west -v update
}
