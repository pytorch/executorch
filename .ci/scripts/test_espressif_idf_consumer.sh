#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Builds the ExecuTorch ESP-IDF component the way a project using the published
# component gets it: generates and packs the component, adds the archive to a
# file:// component mirror, and builds the idf_component_consumer project, which
# resolves executorch from that mirror, with ESP-IDF's own Python environment
# (no ExecuTorch) and with source, model and build paths that have a space.
# Each build runs the smoke model on the emulator; the ESP32-S3 and ESP32-C6
# builds also run a Cadence-quantized streaming wake word model frame by frame
# against PyTorch and a quantized mul on the esp_nn:: op library, and the
# ESP32-S3 build an ESP-NN delegate program.
# ESPRESSIF_IDF_VERSION selects the ESP-IDF release: v6.1 builds every target
# an emulator runs (QEMU for ESP32, esp-emu for the rest), other releases
# ESP32-S3 and ESP32-C6. Also builds the component's example the way a registry
# user gets it, and lints the component's manifest.

set -euo pipefail

et_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
idf_version="${ESPRESSIF_IDF_VERSION:-v6.1}"
build_dir="${ESPRESSIF_BUILD_DIR:-${et_root}/cmake-out-esp-idf-consumer-${idf_version}}"
export ESPRESSIF_TOOLS_DIR="${ESPRESSIF_TOOLS_DIR:-${RUNNER_TEMP:-/tmp}/executorch-espressif-${idf_version}}"
mkdir -p "${build_dir}"
build_dir=$(cd "${build_dir}" && pwd)
cd "${et_root}"

if [[ "${idf_version}" == v6.1 ]]; then
    targets=(esp32s3 esp32 esp32c3 esp32c5 esp32c6 esp32p4)
else
    targets=(esp32s3 esp32c6)
fi

collect_artifacts() {
    if [[ -n "${RUNNER_ARTIFACT_DIR:-}" ]]; then
        mkdir -p "${RUNNER_ARTIFACT_DIR}/esp-idf-consumer"
        cp "${build_dir}"/*.log "${RUNNER_ARTIFACT_DIR}/esp-idf-consumer/" || true
    fi
}
trap collect_artifacts EXIT

# ESP-IDF's QEMU needs these libraries, and setup.sh fails its install check
# without them.
if [[ " ${targets[*]} " == *" esp32 "* ]] && ! python -c '
import ctypes
ctypes.CDLL("libSDL2-2.0.so.0")
ctypes.CDLL("libslirp.so.0")' 2> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends libsdl2-2.0-0 libslirp0
fi
bash examples/espressif/setup.sh --minimal --idf-version="${idf_version}" \
    --targets="$(IFS=,; echo "${targets[*]}")" 2>&1 | tee "${build_dir}/setup.log"
# shellcheck source=/dev/null
source "${ESPRESSIF_TOOLS_DIR}/setup_path.sh"

# Projects keep their models and kernel bundles in their tree, so these get a
# path with a space too.
assets="${build_dir}/consumer assets"
rm -rf "${assets}"
mkdir -p "${assets}"
python examples/espressif/export_smoke_model.py --output "${assets}/smoke.bpte" \
    2>&1 | tee "${build_dir}/export.log"
python examples/espressif/idf_component_smoke/export_wakeword_model.py \
    --output "${assets}/wakeword.bpte" --frames 40 2>&1 | tee -a "${build_dir}/export.log"
python examples/espressif/idf_component_smoke/export_op_library_model.py \
    --output "${assets}/mul.bpte" 2>&1 | tee -a "${build_dir}/export.log"
python examples/espressif/export_espnn_test_programs.py --output-dir "${build_dir}/espnn" \
    --only conv-act-pool-linear 2>&1 | tee -a "${build_dir}/export.log"
cp "${build_dir}/espnn/conv-act-pool-linear.bpte" "${assets}/delegate.bpte"
split=examples/espressif/idf_component_smoke/split_bundled_program.py
python "${split}" "${assets}/smoke.bpte" --atol 0 --rtol 0 --repeat 2
python "${split}" "${assets}/wakeword.bpte" --atol 1e-6 --rtol 0
# test_mul.py's tolerance, torch.finfo(torch.float32).eps, rounded up.
python "${split}" "${assets}/mul.bpte" --atol 0 --rtol 1.2e-7
read -r atol rtol < <(python -c '
import json, sys
(entry,) = json.load(open(sys.argv[1]))
print(entry["atol"], entry["rtol"])
' "${build_dir}/espnn/manifest.json")
python "${split}" "${assets}/delegate.bpte" --atol "${atol}" --rtol "${rtol}"

component_dir="${build_dir}/component/executorch"
bash examples/espressif/build_idf_component.sh --allow-dev-stamp --out "${component_dir}" \
    2>&1 | tee "${build_dir}/generate.log"
compote manifest lint "${component_dir}"
version=$(sed -n 's/^version: "\(.*\)"$/\1/p' "${component_dir}/idf_component.yml")
(cd "${component_dir}" && compote component pack --name executorch --dest-dir "${build_dir}/dist")
archive="${build_dir}/dist/executorch_${version}.tgz"
mirror="${build_dir}/mirror"
rm -rf "${mirror}"
python .ci/scripts/esp_idf_mirror.py "${archive}" "${mirror}" \
    --namespace pytorch --name executorch
# Bundles come from the archive's gen_kernels.py, the one consumers run, so a
# file the packing drops fails here.
packed="${build_dir}/packed"
rm -rf "${packed}"
mkdir -p "${packed}"
tar -xzf "${archive}" -C "${packed}"

gen_kernels() {
    local out=$1
    shift
    python "${packed}/tools/gen_kernels.py" "$@" --out "${out}" \
        2>&1 | tee -a "${build_dir}/gen_kernels.log"
}
cpu_models="${assets}/smoke.pte;${assets}/wakeword.pte;${assets}/mul.pte"
all_models="${cpu_models};${assets}/delegate.pte"
gen_kernels "${assets}/smoke_kernels" --pte "${assets}/smoke.pte"
gen_kernels "${assets}/cpu_kernels" --pte "${assets}/smoke.pte" \
    --pte "${assets}/wakeword.pte" --pte "${assets}/mul.pte"
gen_kernels "${assets}/all_kernels" --pte "${assets}/smoke.pte" \
    --pte "${assets}/wakeword.pte" --pte "${assets}/mul.pte" --pte "${assets}/delegate.pte"

# A fresh ESP-IDF Python environment, which cannot import executorch, and no
# EXECUTORCH_PYTHON to bring one back.
idf_python_env="${build_dir}/idf_python_env"
mkdir -p "${idf_python_env}"
IDF_PYTHON_ENV_PATH="${idf_python_env}" python "${IDF_PATH}/tools/idf_tools.py" \
    install-python-env 2>&1 | tee "${build_dir}/idf_python_env.log"
if "${idf_python_env}/bin/python" -c "import executorch" 2> /dev/null; then
    echo "FAIL: ESP-IDF's Python environment can import executorch" >&2
    exit 1
fi
unset EXECUTORCH_PYTHON
# Built from a copy, so the component manager's state and the consumer's
# generated manifest stay out of the source tree, under a path with a space.
sources="${build_dir}/consumer sources"
rm -rf "${sources}"
mkdir -p "${sources}/idf_component_smoke"
cp -r examples/espressif/idf_component_consumer "${sources}/"
cp -r examples/espressif/idf_component_smoke/main "${sources}/idf_component_smoke/"
project="${sources}/idf_component_consumer"
firmware_root="${build_dir}/consumer builds"

# Runs idf.py on the consumer with ESP-IDF's Python and the mirror: build name,
# target, then idf.py arguments.
consumer() {
    local name=$1 target=$2
    shift 2
    IDF_TARGET="${target}" IDF_PYTHON_ENV_PATH="${idf_python_env}" \
        IDF_COMPONENT_STORAGE_URL="file://${mirror};default" \
        "${idf_python_env}/bin/python" "${IDF_PATH}/tools/idf.py" -C "${project}" \
        -B "${firmware_root}/${name}" -DSDKCONFIG="${firmware_root}/${name}/sdkconfig" \
        -DET_COMPONENT_VERSION="${version}" "$@"
}

# Runs a build on its emulator and checks its output: build name, target, then
# the per-model markers.
run_firmware() {
    local name=$1 target=$2
    shift 2
    local firmware_dir="${firmware_root}/${name}" log="${build_dir}/emulator-${name}.log"
    if [[ "${target}" == esp32 ]]; then
        # QEMU has no exit-on string, so stop it once the app is done.
        setsid bash -c 'consumer "$@" qemu' _ "${name}" "${target}" < /dev/null > "${log}" 2>&1 &
        local pid=$!
        for _ in $(seq 240); do
            if grep -qF "Program complete." "${log}" || ! kill -0 "${pid}" 2> /dev/null; then
                break
            fi
            sleep 0.5
        done
        kill -- "-${pid}" 2> /dev/null || true
        wait "${pid}" || true
    else
        consumer "${name}" "${target}" merge-bin > "${build_dir}/merge-${name}.log" 2>&1
        local elf=("${firmware_dir}"/*.elf)
        timeout 130s esp-emu --chip "${target}" \
            --firmware "${firmware_dir}/merged-binary.bin" --elf "${elf[0]}" \
            --net user --timeout 120s --exit-on 'Program complete.' \
            > "${log}" 2>&1 || true
    fi
    for marker in "$@" "ET_COMPONENT_SMOKE: PASS" "Program complete."; do
        if ! grep -qF "${marker}" "${log}"; then
            echo "FAIL: ${name} printed no '${marker}'; see emulator-${name}.log" >&2
            exit 1
        fi
    done
    echo "OK: ${name} passed on the ${target} emulator"
}
export -f consumer
export idf_python_env mirror project firmware_root version

for target in "${targets[@]}"; do
    if [[ "${target}" == esp32s3 ]]; then
        consumer "${target}" "${target}" -DET_SMOKE_MODELS="${all_models}" \
            -DET_SMOKE_KERNELS="${assets}/all_kernels" build \
            > "${build_dir}/firmware-${target}.log" 2>&1
        run_firmware "${target}" "${target}" "smoke: PASS" "wakeword: PASS, 40/40 cases" \
            "mul: PASS" "delegate: PASS"
    elif [[ "${target}" == esp32c6 ]]; then
        consumer "${target}" "${target}" -DET_SMOKE_MODELS="${cpu_models}" \
            -DET_SMOKE_KERNELS="${assets}/cpu_kernels" build \
            > "${build_dir}/firmware-${target}.log" 2>&1
        run_firmware "${target}" "${target}" "smoke: PASS" "wakeword: PASS, 40/40 cases" \
            "mul: PASS"
    else
        consumer "${target}" "${target}" -DET_SMOKE_MODELS="${assets}/smoke.pte" \
            -DET_SMOKE_KERNELS="${assets}/smoke_kernels" build \
            > "${build_dir}/firmware-${target}.log" 2>&1
        run_firmware "${target}" "${target}" "smoke: PASS"
    fi
    python - "${project}/dependencies.lock" "${mirror}/components/pytorch/executorch.json" <<'PY'
import json
import sys

import yaml

entry = yaml.safe_load(open(sys.argv[1]))["dependencies"]["pytorch/executorch"]
(mirrored,) = json.load(open(sys.argv[2]))["versions"]
if (str(entry["version"]), entry.get("component_hash")) != (
    mirrored["version"],
    mirrored["component_hash"],
):
    sys.exit(f"FAIL: pytorch/executorch did not come from the mirror: {entry}")
PY
done

# The component's example as idf.py create-project-from-example gives it to a
# registry user: copied out of the archive, without the override_path that the
# registry removes, so its exact version pin resolves from the mirror. The
# component manager cannot download examples from a file:// mirror itself.
example="${sources}/hello_executorch"
rm -rf "${example}"
mkdir -p "${example}"
tar -xzf "${archive}" -C "${example}" --strip-components=3 ./examples/hello_executorch
python - "${example}/main/idf_component.yml" <<'PY'
import sys

import yaml

manifest = yaml.safe_load(open(sys.argv[1]))
del manifest["dependencies"]["pytorch/executorch"]["override_path"]
yaml.safe_dump(manifest, open(sys.argv[1], "w"))
PY
project="${example}" consumer hello_executorch esp32s3 build \
    > "${build_dir}/firmware-hello_executorch.log" 2>&1
project="${example}" run_firmware hello_executorch esp32s3 "smoke: PASS"
echo "PASS: the packed ESP-IDF component built from a mirror and ran on ${targets[*]}."
