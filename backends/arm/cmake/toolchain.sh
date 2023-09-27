#!/bin/bash
# Copyright 2023 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -e

# Cross compiler for Arm baremetal (e.g. Corestone-300 FVP or silcon)
curl -o gcc.tar.xz https://armkeil.blob.core.windows.net/developer/Files/downloads/gnu/12.3.rel1/binrel/arm-gnu-toolchain-12.3.rel1-aarch64-arm-none-eabi.tar.xz
tar xf gcc.tar.xz
export PATH=${PATH}:`(cd arm-gnu-toolchain-12.3.rel1-aarch64-arm-none-eabi/bin/; pwd)`
