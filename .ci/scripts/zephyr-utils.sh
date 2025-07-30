#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

download_arm_zephyr_sdk () {
    wget https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v0.16.0/zephyr-sdk-0.16.0_linux-x86_64.tar.xz
    tar -xf zephyr-sdk-0.16.0_linux-x86_64.tar.xz
    rm -f zephyr-sdk-0.16.0_linux-x86_64.tar.xz
}

setup_zephyr_et_module () {
    git clone --branch executorch-module-integration https://github.com/BujSet/zephyr.git
    west init -l zephyr
    west config manifest.project-filter -- +executorch
    west -v update
}

setup_optimum() {
    git clone https://github.com/huggingface/optimum-executorch.git
    cd optimum-executorch
    python -m pip install --upgrade pip
    python -m pip install '.[dev]'
    python install_dev.py --skip_override_torch
    python -m pip install torchao==0.11.0
    python -m pip install transformers==4.52.4
    python -m pip install torchcodec==0.4.0
}
