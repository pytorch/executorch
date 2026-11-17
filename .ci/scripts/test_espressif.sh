#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

et_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
build_dir="${ESPRESSIF_BUILD_DIR:-${et_root}/cmake-out-esp-emulator}"
espnn_kernels="${ESPRESSIF_NN_KERNELS:-optimized}"
case "${espnn_kernels}" in
    optimized) espnn_config=CONFIG_NN_OPTIMIZED ;;
    ansi) espnn_config=CONFIG_NN_ANSI_C ;;
    *) echo "ESPRESSIF_NN_KERNELS must be optimized or ansi" >&2; exit 1 ;;
esac
export ESPRESSIF_TOOLS_DIR="${ESPRESSIF_TOOLS_DIR:-${RUNNER_TEMP:-/tmp}/executorch-espressif}"
mkdir -p "${build_dir}"
build_dir=$(cd "${build_dir}" && pwd)
cd "${et_root}"

collect_artifacts() {
    local file operator_dir artifact_dir
    if [[ -n "${RUNNER_ARTIFACT_DIR:-}" ]]; then
        mkdir -p "${RUNNER_ARTIFACT_DIR}/esp32s3"
        for file in "${build_dir}"/*.log "${build_dir}"/*.bpte \
            "${build_dir}"/espnn-*-sdkconfig \
            "${build_dir}/espnn-tests"/*.bpte \
            "${build_dir}/espnn-tests/manifest.json" \
            "${build_dir}/espnn-tests/operators.txt" \
            "${et_root}/examples/espressif/project/dependencies.lock" \
            "${build_dir}/firmware/merged-binary.bin" \
            "${build_dir}/firmware/executorch_esp_runner.elf"; do
            if [[ -f "${file}" ]]; then
                cp "${file}" "${RUNNER_ARTIFACT_DIR}/esp32s3/"
            fi
        done
        for operator_dir in "${build_dir}"/operator-library-*; do
            [[ -d "${operator_dir}" ]] || continue
            artifact_dir="${RUNNER_ARTIFACT_DIR}/esp32s3/${operator_dir##*/}"
            mkdir -p "${artifact_dir}"
            for file in "${operator_dir}"/*.log "${operator_dir}"/*.bpte \
                "${operator_dir}/sdkconfig" "${operator_dir}"/*.elf \
                "${operator_dir}/merged-binary.bin"; do
                if [[ -f "${file}" ]]; then
                    cp "${file}" "${artifact_dir}/"
                fi
            done
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
if [[ "${espnn_kernels}" == optimized ]]; then
    espnn_ops="${espnn_ops},aten::sin.out"
fi

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
espnn_firmware_dir="${build_dir}/espnn-${espnn_kernels}-firmware"
espnn_sdkconfig="${build_dir}/espnn-${espnn_kernels}-sdkconfig"
espnn_defaults="${build_dir}/espnn-${espnn_kernels}.defaults"
printf '%s=y\n' "${espnn_config}" > "${espnn_defaults}"
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
    espnn_log_prefix="espnn-${espnn_kernels}-${name}"
    logged "${espnn_log_prefix}-configure.log" \
        idf.py -C examples/espressif/project -B "${espnn_firmware_dir}" \
        -DIDF_TARGET=esp32s3 \
        -DSDKCONFIG="${espnn_sdkconfig}" \
        "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;${espnn_defaults}" \
        -DET_ESPNN_BACKEND=ON \
        -DET_BUILD_DIR_PATH="${build_dir}/runtime" \
        -DET_PTE_FILE_PATH="${build_dir}/espnn-tests/${name}.bpte" \
        -DET_BUNDLE_IO=ON -DET_NUM_INFERENCES=1 \
        -DET_ATOL="${atol}" -DET_RTOL="${rtol}" reconfigure
    if ! grep -qx "${espnn_config}=y" "${espnn_sdkconfig}"; then
        echo "Expected ${espnn_config}=y in ${espnn_sdkconfig}" >&2
        exit 1
    fi
    logged "${espnn_log_prefix}-build.log" cmake --build "${espnn_firmware_dir}"
    logged "${espnn_log_prefix}-merge.log" \
        idf.py -C examples/espressif/project -B "${espnn_firmware_dir}" merge-bin
    logged "${espnn_log_prefix}-emulator.log" \
        timeout 90s esp-emu --chip esp32s3 \
        --firmware "${espnn_firmware_dir}/merged-binary.bin" \
        --elf "${espnn_firmware_dir}/executorch_esp_runner.elf" \
        --net user --psram-size 8M --timeout 60s --exit-on 'Program complete.'
    echo -n "${espnn_kernels}/${name}: "
    python examples/espressif/test_emulator.py "${build_dir}/${espnn_log_prefix}-emulator.log"
done 3< <(python -c '
import json, sys
for entry in json.load(open(sys.argv[1])):
    print(entry["name"], entry["atol"], entry["rtol"])
' "${build_dir}/espnn-tests/manifest.json")

if [[ "${espnn_kernels}" == optimized ]]; then
    # Stop on failure so later cases cannot overwrite its firmware and logs.
    ESPRESSIF_BUILD_DIR="${build_dir}" python -m pytest \
        -x -v --tb=short backends/espressif/test/ops -k implementation \
        2>&1 | tee "${build_dir}/operator-library-tests.log"
    for sdkconfig in "${build_dir}"/operator-library-*/sdkconfig; do
        if ! grep -qx 'CONFIG_NN_OPTIMIZED=y' "${sdkconfig}"; then
            echo "Expected CONFIG_NN_OPTIMIZED=y in ${sdkconfig}" >&2
            exit 1
        fi
    done
fi
