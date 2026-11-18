#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Builds the ESP-NN delegate for the host and checks fully delegated programs
# against the bit-exact Python reference: plain, and with ASan/UBSan over
# esp-nn's portable kernels, over its optimized C kernels, and with the
# delegate's non-default options. The sanitized builds make the kernels use
# scratch, in a separate buffer or in the arena. Needs no ESP-IDF toolchain or
# emulator.

set -euo pipefail

et_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
build_dir="${ESPRESSIF_HOST_BUILD_DIR:-${et_root}/cmake-out-espnn-host}"
cd "${et_root}"

run_variant() {
    local variant=$1
    shift
    cmake -S backends/espressif/test/host -B "${build_dir}/${variant}" -G Ninja \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DPYTHON_EXECUTABLE="$(command -v python)" "$@"
    cmake --build "${build_dir}/${variant}"
    ESPNN_HOST_RUNNER="${build_dir}/${variant}/espnn_host_runner" \
        ESPNN_HOST_INTERPRETER="${build_dir}/${variant}/espnn_host_interpreter" \
        python -m pytest -v --tb=short backends/espressif/test/test_host_runtime.py \
        backends/espressif/test/test_host_interpreter.py
}

run_variant plain
scratch=(-DESPNN_HOST_SANITIZE=ON -DESPNN_HOST_SCRATCH=ON)
run_variant sanitized "${scratch[@]}"
run_variant sanitized-optimized "${scratch[@]}" -DESPNN_HOST_OPTIMIZED=ON
run_variant sanitized-options "${scratch[@]}" -DESPNN_HOST_OPTIMIZED=ON \
    -DESPNN_HOST_PSRAM=ON -DESPNN_DELEGATE_CONSTANTS_TO_PSRAM=ON \
    -DESPNN_DELEGATE_PROFILE=ON -DESPNN_DELEGATE_SCRATCH_IN_ARENA=ON
