#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Generates the ExecuTorch ESP-IDF component and a kernel bundle for the smoke
# model, builds the idf_component_smoke consumer for ESP32-S3 and, with the
# Cadence kernels too, for ESP32-C6, and runs the ESP32-S3 build on the
# emulator. Also checks that stale or mismatched bundles fail at configure time,
# and that an app declaring nothing links to the same size as one without the
# component.

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
        for file in "${build_dir}"/*.log "${build_dir}/components/executorch/VERSION" \
            "${build_dir}/smoke_kernels/kernels.json"; do
            if [[ -f "${file}" ]]; then
                cp "${file}" "${RUNNER_ARTIFACT_DIR}/esp-idf-component/"
            fi
        done
    fi
}
trap collect_artifacts EXIT

bash examples/espressif/setup.sh --minimal --targets=esp32s3,esp32c6 2>&1 \
    | tee "${build_dir}/setup.log"
# shellcheck source=/dev/null
source "${ESPRESSIF_TOOLS_DIR}/setup_path.sh"

python examples/espressif/export_smoke_model.py \
    --output "${build_dir}/smoke.bpte" --pte-output "${build_dir}/smoke.pte" \
    2>&1 | tee "${build_dir}/export.log"

component_dir="${build_dir}/components/executorch"
kernels_dir="${build_dir}/smoke_kernels"
bash examples/espressif/build_idf_component.sh --out "${component_dir}" \
    2>&1 | tee "${build_dir}/generate.log"
python "${component_dir}/tools/gen_kernels.py" --pte "${build_dir}/smoke.pte" \
    --out "${kernels_dir}" 2>&1 | tee "${build_dir}/gen_kernels.log"
# The ESP32-C6 build also registers the ops of a Cadence-quantized streaming
# wake word model, so CI compiles and links the Cadence generic kernels.
wakeword_ops=aten::_softmax.out,aten::_to_copy.out,aten::cat.out,aten::copy_
wakeword_ops+=,aten::permute_copy.out,aten::slice_copy.Tensor_out
wakeword_ops+=,aten::squeeze_copy.dim_out,aten::transpose_copy.int_out
wakeword_ops+=,cadence::dequantize_per_tensor.out,cadence::quantize_per_tensor.out
wakeword_ops+=,cadence::quantized_conv2d_nhwc.per_tensor_out
wakeword_ops+=,cadence::quantized_fully_connected.per_tensor_out
wakeword_ops+=,cadence::quantized_relu.per_tensor_out
python "${component_dir}/tools/gen_kernels.py" --pte "${build_dir}/smoke.pte" \
    --ops "${wakeword_ops}" --out "${build_dir}/wakeword_kernels" \
    2>&1 | tee -a "${build_dir}/gen_kernels.log"

project=examples/espressif/idf_component_smoke

# Runs idf.py on the smoke consumer: build name, target, then idf.py arguments.
consumer() {
    local name=$1 target=$2
    shift 2
    IDF_TARGET="${target}" idf.py -C "${project}" -B "${build_dir}/firmware-${name}" \
        -DSDKCONFIG="${build_dir}/sdkconfig-${name}" "$@" \
        > "${build_dir}/firmware-${name}.log" 2>&1
}

# Configures a case that must fail: build name, expected message, -D arguments.
expect_configure_error() {
    local name=$1 expected=$2
    shift 2
    if consumer "${name}" esp32s3 -DET_COMPONENT_DIR="${component_dir}" "$@" reconfigure; then
        echo "FAIL: ${name} configured; expected an error with '${expected}'" >&2
        exit 1
    fi
    # CMake wraps long messages, so match with whitespace runs collapsed.
    if ! tr -s ' \n' '  ' < "${build_dir}/firmware-${name}.log" | grep -qF "${expected}"; then
        echo "FAIL: ${name} did not report '${expected}'; see firmware-${name}.log" >&2
        exit 1
    fi
    echo "OK: ${name} failed at configure with '${expected}'"
}

# A re-exported smoke.pte.
mkdir -p "${build_dir}/stale"
cp "${build_dir}/smoke.pte" "${build_dir}/stale/smoke.pte"
printf '\0' >> "${build_dir}/stale/smoke.pte"
expect_configure_error stale-model "smoke.pte: changed since the bundle was made" \
    -DET_SMOKE_PTE="${build_dir}/stale/smoke.pte" -DET_SMOKE_KERNELS="${kernels_dir}"

rm -rf "${build_dir}/mismatch_kernels"
cp -r "${kernels_dir}" "${build_dir}/mismatch_kernels"
python - "${build_dir}/mismatch_kernels/kernels.json" <<'PY'
import json
import sys

bundle = json.load(open(sys.argv[1]))
bundle["component_version"] = "0.0.0"
json.dump(bundle, open(sys.argv[1], "w"))
PY
expect_configure_error version-mismatch component_version \
    -DET_SMOKE_PTE="${build_dir}/smoke.pte" \
    -DET_SMOKE_KERNELS="${build_dir}/mismatch_kernels"

# A bundle made by a component of the same version from another commit, as any
# two commits between version.txt bumps are.
rm -rf "${build_dir}/mismatch_kernels"
cp -r "${kernels_dir}" "${build_dir}/mismatch_kernels"
python - "${build_dir}/mismatch_kernels/kernels.json" <<'PY'
import json
import sys

bundle = json.load(open(sys.argv[1]))
bundle["component_commit"] = "0" * 40
json.dump(bundle, open(sys.argv[1], "w"))
PY
expect_configure_error commit-mismatch component_commit \
    -DET_SMOKE_PTE="${build_dir}/smoke.pte" \
    -DET_SMOKE_KERNELS="${build_dir}/mismatch_kernels"

expect_configure_error no-bundle "no kernel bundle is" -DET_SMOKE_PTE="${build_dir}/smoke.pte"

# A model the bundle was not made from, declared instead of the one it was.
mkdir -p "${build_dir}/other"
cp "${build_dir}/stale/smoke.pte" "${build_dir}/other/other.pte"
expect_configure_error other-model \
    "other.pte: not in the bundle smoke.pte: in the bundle, not declared" \
    -DET_SMOKE_PTE="${build_dir}/other/other.pte" -DET_SMOKE_KERNELS="${kernels_dir}"

rm -rf "${build_dir}/no_count_kernels"
cp -r "${kernels_dir}" "${build_dir}/no_count_kernels"
rm -r "${build_dir}/no_count_kernels/include"
expect_configure_error no-kernel-count "has no kernel count" \
    -DET_SMOKE_PTE="${build_dir}/smoke.pte" -DET_SMOKE_KERNELS="${build_dir}/no_count_kernels"

# A bundle used without declaring its models is used as is, with a warning that
# names them.
consumer as-is esp32s3 -DET_COMPONENT_DIR="${component_dir}" \
    -DET_SMOKE_PTE="${build_dir}/smoke.pte" -DET_SMOKE_UNDECLARED=1 \
    -DET_SMOKE_KERNELS="${kernels_dir}" reconfigure
if ! tr -s ' \n' '  ' < "${build_dir}/firmware-as-is.log" \
    | grep -qF "is not checked against smoke.pte"; then
    echo "FAIL: no warning naming smoke.pte; see ${build_dir}/firmware-as-is.log" >&2
    exit 1
fi
echo "OK: as-is warned that smoke.pte is not checked"

# An app that runs a model while declaring nothing fails to link.
if consumer undeclared esp32s3 -DET_COMPONENT_DIR="${component_dir}" \
    -DET_SMOKE_PTE="${build_dir}/smoke.pte" -DET_SMOKE_UNDECLARED=1 build; then
    echo "FAIL: undeclared linked; expected et_requires_executorch_add_kernels to be undefined" >&2
    exit 1
fi
if ! grep -qF "undefined reference to \`et_requires_executorch_add_kernels'" \
    "${build_dir}/firmware-undeclared.log"; then
    echo "FAIL: undeclared did not fail on et_requires_executorch_add_kernels; see firmware-undeclared.log" >&2
    exit 1
fi
echo "OK: undeclared failed to link on et_requires_executorch_add_kernels"

# A kernel registry smaller than the bundle; the emulator run below uses one
# larger than it.
printf 'CONFIG_ET_MAX_KERNEL_NUM=1\n' > "${build_dir}/small_registry.defaults"
printf 'CONFIG_ET_MAX_KERNEL_NUM=64\n' > "${build_dir}/large_registry.defaults"
expect_configure_error small-registry "CONFIG_ET_MAX_KERNEL_NUM=1 is below" \
    "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;${build_dir}/small_registry.defaults" \
    -DET_SMOKE_PTE="${build_dir}/smoke.pte" -DET_SMOKE_KERNELS="${kernels_dir}"

# An app that declares nothing must not grow because the component is in the
# build. The baseline swaps the component for an empty one of the same name.
mkdir -p "${build_dir}/empty_component/executorch"
echo "idf_component_register()" > "${build_dir}/empty_component/executorch/CMakeLists.txt"
for name in empty-baseline empty; do
    if [[ "${name}" == empty-baseline ]]; then
        dir="${build_dir}/empty_component/executorch"
    else
        dir="${component_dir}"
    fi
    consumer "${name}" esp32s3 -DET_COMPONENT_DIR="${dir}" build
    consumer "${name}" esp32s3 -DET_COMPONENT_DIR="${dir}" size --format json2 \
        --output-file "${build_dir}/size-${name}.json"
done
python - "${build_dir}/size-empty-baseline.json" "${build_dir}/size-empty.json" <<'PY'
import json
import sys

baseline, empty = (json.load(open(p))["total_size"] for p in sys.argv[1:])
if baseline != empty:
    sys.exit(f"FAIL: empty app is {empty} B with the component, {baseline} B without")
print(f"OK: empty app is {empty} B with and without the component")
PY

# With newlib's nano printf, the component compiles ExecuTorch logging out and
# says so.
printf 'CONFIG_LIBC_NEWLIB=y\nCONFIG_LIBC_NEWLIB_NANO_FORMAT=y\n' > "${build_dir}/nano.defaults"
consumer nano esp32s3 -DET_COMPONENT_DIR="${component_dir}" \
    "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;${build_dir}/nano.defaults" \
    -DET_SMOKE_PTE="${build_dir}/smoke.pte" -DET_SMOKE_KERNELS="${kernels_dir}" reconfigure
if ! tr -s ' \n' '  ' < "${build_dir}/firmware-nano.log" \
    | grep -qF "ExecuTorch logging is DISABLED"; then
    echo "FAIL: no nano printf warning; see ${build_dir}/firmware-nano.log" >&2
    exit 1
fi

consumer smoke-esp32s3 esp32s3 -DET_COMPONENT_DIR="${component_dir}" \
    "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;${build_dir}/large_registry.defaults" \
    -DET_SMOKE_PTE="${build_dir}/smoke.pte" -DET_SMOKE_KERNELS="${kernels_dir}" build
consumer smoke-esp32c6 esp32c6 -DET_COMPONENT_DIR="${component_dir}" \
    -DET_SMOKE_PTE="${build_dir}/smoke.pte" -DET_SMOKE_KERNELS="${build_dir}/wakeword_kernels" build

firmware_dir="${build_dir}/firmware-smoke-esp32s3"
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
