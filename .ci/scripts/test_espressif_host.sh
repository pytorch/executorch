#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Builds the ESP-NN delegate for the host, plain and with ASan/UBSan, and checks
# fully delegated programs against the bit-exact Python reference on both. Needs
# no ESP-IDF toolchain or emulator.

set -euo pipefail

et_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
build_dir="${ESPRESSIF_HOST_BUILD_DIR:-${et_root}/cmake-out-espnn-host}"
cd "${et_root}"

for variant in plain sanitized; do
    sanitize=OFF
    if [[ "${variant}" == sanitized ]]; then
        sanitize=ON
    fi
    cmake -S backends/espressif/test/host -B "${build_dir}/${variant}" -G Ninja \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DPYTHON_EXECUTABLE="$(command -v python)" \
        -DESPNN_HOST_SANITIZE="${sanitize}"
    cmake --build "${build_dir}/${variant}"
    ESPNN_HOST_RUNNER="${build_dir}/${variant}/espnn_host_runner" \
        python -m pytest -v --tb=short backends/espressif/test/test_host_runtime.py
done
