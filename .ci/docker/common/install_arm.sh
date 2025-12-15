#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

install_arm_prerequiresites() {
    apt-get update -y
    apt-get install -y --no-install-recommends \
            mesa-vulkan-drivers libvulkan1
    rm -rf /var/lib/apt/lists/*
}

install_arm_prerequiresites
