#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

et_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
build_dir="${ESPRESSIF_BUILD_DIR:-${et_root}/cmake-out-esp-emulator}"
export ESPRESSIF_TOOLS_DIR="${ESPRESSIF_TOOLS_DIR:-${RUNNER_TEMP:-/tmp}/executorch-espressif}"
mkdir -p "${build_dir}"
build_dir=$(cd "${build_dir}" && pwd)
cd "${et_root}"

collect_artifacts() {
    if [[ -n "${RUNNER_ARTIFACT_DIR:-}" ]]; then
        mkdir -p "${RUNNER_ARTIFACT_DIR}/esp32s3"
        for file in "${build_dir}"/*.log "${build_dir}"/*.bpte \
            "${build_dir}/firmware/merged-binary.bin" \
            "${build_dir}/firmware/executorch_esp_runner.elf"; do
            if [[ -f "${file}" ]]; then
                cp "${file}" "${RUNNER_ARTIFACT_DIR}/esp32s3/"
            fi
        done
    fi
}
trap collect_artifacts EXIT

bash examples/espressif/setup.sh --minimal 2>&1 | tee "${build_dir}/setup.log"
# shellcheck source=/dev/null
source "${ESPRESSIF_TOOLS_DIR}/setup_path.sh"

python examples/espressif/export_smoke_model.py \
    --output "${build_dir}/smoke.bpte" 2>&1 | tee "${build_dir}/export.log"
python examples/espressif/export_espnn_test_programs.py \
    --output-dir "${build_dir}/espnn-tests" 2>&1 | tee "${build_dir}/espnn-export.log"

# One runtime serves the smoke test and the ESP-NN programs. quantized_ops_lib
# registers kernels outside the selected list, so the registry is sized
# explicitly instead of from the list. The quantized kernels' parallel_for
# runs inline without pthreadpool, which needs POSIX calls ESP-IDF lacks.
espnn_ops=$(tr -d '\n' < "${build_dir}/espnn-tests/operators.txt" \
    | tr ',' '\n' | grep -v '^quantized_decomposed::' | paste -sd, -)

cmake --preset esp-baremetal -G Ninja -B "${build_dir}/runtime" \
    -DCMAKE_TOOLCHAIN_FILE="${IDF_PATH}/tools/cmake/toolchain-esp32s3.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_DEVTOOLS=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_SELECT_OPS_LIST="aten::add.out,aten::mul.out,${espnn_ops}" \
    -DMAX_KERNEL_NUM=256 \
    -DEXECUTORCH_BUILD_PTHREADPOOL=OFF -DEXECUTORCH_BUILD_CPUINFO=OFF \
    2>&1 | tee "${build_dir}/runtime-configure.log"
cmake --build "${build_dir}/runtime" \
    2>&1 | tee "${build_dir}/runtime-build.log"
cmake --install "${build_dir}/runtime" \
    2>&1 | tee "${build_dir}/runtime-install.log"

idf.py -C examples/espressif/project -B "${build_dir}/firmware" \
    -DIDF_TARGET=esp32s3 \
    -DSDKCONFIG="${build_dir}/sdkconfig" \
    -DET_BUILD_DIR_PATH="${build_dir}/runtime" \
    -DET_PTE_FILE_PATH="${build_dir}/smoke.bpte" \
    -DET_BUNDLE_IO=ON -DET_NUM_INFERENCES=3 \
    -DET_ATOL=0.000001 -DET_RTOL=0.000001 reconfigure \
    2>&1 | tee "${build_dir}/firmware-configure.log"
cmake --build "${build_dir}/firmware" \
    2>&1 | tee "${build_dir}/firmware-build.log"
idf.py -C examples/espressif/project -B "${build_dir}/firmware" merge-bin \
    2>&1 | tee "${build_dir}/firmware-merge.log"

timeout 130s esp-emu --chip esp32s3 \
    --firmware "${build_dir}/firmware/merged-binary.bin" \
    --elf "${build_dir}/firmware/executorch_esp_runner.elf" \
    --net user --psram-size 8M --timeout 120s --exit-on 'Program complete.' \
    2>&1 | tee "${build_dir}/emulator.log"
python examples/espressif/test_emulator.py "${build_dir}/emulator.log"

# ESP-NN backend: run every exported program through the ExecuTorch runtime
# and the ESP-NN delegate, and check it against the bundled host reference.
# The emulator does not implement the ESP32-S3 SIMD instructions, so ESP-NN is
# built with its ANSI C kernels.
printf 'CONFIG_NN_ANSI_C=y\n' > "${build_dir}/espnn-emulator.defaults"
# Runs a command with its output in a log, and shows the log's tail on failure.
logged() {
    local log="${build_dir}/$1"
    shift
    if ! "$@" > "${log}" 2>&1; then
        tail -n 60 "${log}"
        echo "FAIL: see ${log}" >&2
        return 1
    fi
}
while read -r -u 3 name atol rtol; do
    logged "espnn-${name}-configure.log" \
        idf.py -C examples/espressif/project -B "${build_dir}/espnn-firmware" \
        -DIDF_TARGET=esp32s3 \
        -DSDKCONFIG="${build_dir}/espnn-sdkconfig" \
        "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;${build_dir}/espnn-emulator.defaults" \
        -DET_ESPNN_BACKEND=ON \
        -DET_BUILD_DIR_PATH="${build_dir}/runtime" \
        -DET_PTE_FILE_PATH="${build_dir}/espnn-tests/${name}.bpte" \
        -DET_BUNDLE_IO=ON -DET_NUM_INFERENCES=1 \
        -DET_ATOL="${atol}" -DET_RTOL="${rtol}" reconfigure
    logged "espnn-${name}-build.log" cmake --build "${build_dir}/espnn-firmware"
    logged "espnn-${name}-merge.log" \
        idf.py -C examples/espressif/project -B "${build_dir}/espnn-firmware" merge-bin
    logged "espnn-${name}-emulator.log" \
        timeout 90s esp-emu --chip esp32s3 \
        --firmware "${build_dir}/espnn-firmware/merged-binary.bin" \
        --elf "${build_dir}/espnn-firmware/executorch_esp_runner.elf" \
        --net user --psram-size 8M --timeout 60s --exit-on 'Program complete.'
    echo -n "${name}: "
    python examples/espressif/test_emulator.py "${build_dir}/espnn-${name}-emulator.log"
done 3< <(python -c '
import json, sys
for entry in json.load(open(sys.argv[1])):
    print(entry["name"], entry["atol"], entry["rtol"])
' "${build_dir}/espnn-tests/manifest.json")
