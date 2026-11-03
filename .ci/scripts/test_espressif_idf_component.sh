#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Generates the ExecuTorch ESP-IDF component with every shipped profile,
# builds the idf_component_smoke consumer for ESP32-S3 against each, and runs
# the smoke profile build on the emulator.

set -euo pipefail

et_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
build_dir="${ESPRESSIF_BUILD_DIR:-${et_root}/cmake-out-esp-idf-component}"
export ESPRESSIF_TOOLS_DIR="${ESPRESSIF_TOOLS_DIR:-${RUNNER_TEMP:-/tmp}/executorch-espressif}"
mkdir -p "${build_dir}"
build_dir=$(cd "${build_dir}" && pwd)
cd "${et_root}"

collect_artifacts() {
    if [[ -n "${RUNNER_ARTIFACT_DIR:-}" ]]; then
        mkdir -p "${RUNNER_ARTIFACT_DIR}/esp-idf-component"
        for file in "${build_dir}"/*.log "${build_dir}/components/executorch/VERSION"; do
            if [[ -f "${file}" ]]; then
                cp "${file}" "${RUNNER_ARTIFACT_DIR}/esp-idf-component/"
            fi
        done
    fi
}
trap collect_artifacts EXIT

bash examples/espressif/setup.sh --minimal 2>&1 | tee "${build_dir}/setup.log"
# shellcheck source=/dev/null
source "${ESPRESSIF_TOOLS_DIR}/setup_path.sh"

python examples/espressif/export_smoke_model.py \
    --output "${build_dir}/smoke.bpte" --pte-output "${build_dir}/smoke.pte" \
    2>&1 | tee "${build_dir}/export.log"

component_dir="${build_dir}/components/executorch"
profiles=$(find examples/espressif/idf_profiles -name '*.ops' -exec basename {} .ops \; \
    | sort | paste -sd, -)
bash examples/espressif/build_idf_component.sh --profiles "${profiles}" \
    --out "${component_dir}" 2>&1 | tee "${build_dir}/generate.log"

project=examples/espressif/idf_component_smoke
for profile in ${profiles//,/ }; do
    profile_upper=$(echo "${profile}" | tr '[:lower:]-' '[:upper:]_')
    firmware_dir="${build_dir}/firmware-${profile}"
    printf 'CONFIG_ET_OP_PROFILE_%s=y\n' "${profile_upper}" \
        > "${build_dir}/${profile}.defaults"
    idf.py -C "${project}" -B "${firmware_dir}" \
        -DIDF_TARGET=esp32s3 \
        -DSDKCONFIG="${build_dir}/sdkconfig-${profile}" \
        "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;${build_dir}/${profile}.defaults" \
        -DET_COMPONENT_DIR="${component_dir}" \
        -DET_SMOKE_PTE="${build_dir}/smoke.pte" build \
        2>&1 | tee "${build_dir}/firmware-${profile}.log"
done

firmware_dir="${build_dir}/firmware-smoke"
idf.py -C "${project}" -B "${firmware_dir}" merge-bin \
    2>&1 | tee "${build_dir}/firmware-merge.log"
timeout 130s esp-emu --chip esp32s3 \
    --firmware "${firmware_dir}/merged-binary.bin" \
    --elf "${firmware_dir}/et_component_smoke.elf" \
    --net user --timeout 120s --exit-on 'Program complete.' \
    2>&1 | tee "${build_dir}/emulator.log"
for marker in "ET_COMPONENT_SMOKE: PASS" "Program complete."; do
    if ! grep -qF "${marker}" "${build_dir}/emulator.log"; then
        echo "FAIL: missing '${marker}'; see ${build_dir}/emulator.log" >&2
        exit 1
    fi
done
echo "PASS: generated ESP-IDF component ran the smoke model on the ESP32-S3 emulator."
