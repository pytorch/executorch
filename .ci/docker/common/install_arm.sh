#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

install_arm_prerequiresites() {
    apt-get update -y
    apt-get install -y --no-install-recommends \
            mesa-vulkan-drivers libvulkan1 util-linux
    rm -rf /var/lib/apt/lists/*
}

cache_arm_fvps() {
    local script_dir
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    if [[ "${script_dir}" != /opt/executorch/common && \
          -f "${script_dir}/arm_fvp_cache.sh" && \
          -f "${script_dir}/../ci_commit_pins/arm-fvp.sh" ]]; then
        install -D -m 644 "${script_dir}/arm_fvp_cache.sh" \
            /opt/executorch/common/arm_fvp_cache.sh
        install -D -m 644 "${script_dir}/../ci_commit_pins/arm-fvp.sh" \
            /opt/executorch/ci_commit_pins/arm-fvp.sh
    fi
    source /opt/executorch/common/arm_fvp_cache.sh
    local platform
    for platform in "${ARM_FVP_PLATFORMS[@]}"; do
        ARM_FVP_CACHE_DIR=/opt/arm-fvp-cache ARM_FVP_CACHE_ONLY=0 \
            arm_fvp_cache_archive "$(uname -m)" "${platform}"
    done
    chmod -R a+rX /opt/arm-fvp-cache
}

install_arm_prerequiresites
cache_arm_fvps
