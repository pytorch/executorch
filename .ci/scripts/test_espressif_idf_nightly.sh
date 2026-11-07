#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# The ExecuTorch ESP-IDF component's nightly matrix.
#
#   test_espressif_idf_nightly.sh prepare <dir>
#     In the ExecuTorch environment: exports the test models (smoke, the
#     esp_nn:: mul, an ESP-NN delegate program and the Cadence-quantized wake
#     word), generates and packs the component into a file:// component mirror,
#     and makes each model's kernel bundle, all under <dir>. all_ops is the
#     smoke model with a bundle that registers every op of every kernel
#     library.
#   test_espressif_idf_nightly.sh build <dir> <shard> <shards>
#     Needs only <dir> from prepare: installs ESPRESSIF_IDF_VERSION's ESP-IDF
#     (default v6.1), lists the idf_component_consumer configurations with
#     idf-build-apps (sdkconfig.ci.* times every target, filtered by
#     .build-test-rules.yml), builds shard <shard> of <shards> with ESP-IDF's
#     own Python environment, checks that each configuration's settings took
#     effect, and runs each build on its emulator unless the rules disable its
#     test. Reports every failure, then fails.

set -euo pipefail

et_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
mode=${1:?usage: $0 prepare <dir> | build <dir> <shard> <shards>}
dir=${2:?usage: $0 prepare <dir> | build <dir> <shard> <shards>}
mkdir -p "${dir}"
dir=$(cd "${dir}" && pwd)
cd "${et_root}"
targets=(esp32 esp32s3 esp32c3 esp32c5 esp32c6 esp32p4)

# Runs a command with its output in a log, and shows the log's tail if it fails.
logged() {
    local log=$1
    shift
    if ! "$@" > "${log}" 2>&1; then
        tail -n 30 "${log}" >&2
        echo "FAIL: $*; see ${log}" >&2
        exit 1
    fi
}

prepare() {
    export ESPRESSIF_TOOLS_DIR="${ESPRESSIF_TOOLS_DIR:-${RUNNER_TEMP:-/tmp}/executorch-espressif}"
    local logs="${dir}/logs" models="${dir}/models"
    mkdir -p "${logs}" "${models}"
    # The component manager comes with ESP-IDF's Python packages.
    logged "${logs}/setup.log" bash examples/espressif/setup.sh --minimal --targets=esp32s3 \
        --skip-emulator
    # shellcheck source=/dev/null
    source "${ESPRESSIF_TOOLS_DIR}/setup_path.sh"

    python examples/espressif/export_smoke_model.py --output "${models}/smoke.bpte"
    python examples/espressif/idf_component_smoke/export_op_library_model.py \
        --output "${models}/mul.bpte"
    python examples/espressif/export_espnn_test_programs.py --output-dir "${dir}/espnn" \
        --only conv-act-pool-linear
    cp "${dir}/espnn/conv-act-pool-linear.bpte" "${models}/delegate.bpte"
    python examples/espressif/export_wakeword_model.py --output "${models}/wakeword.bpte"
    local split=examples/espressif/idf_component_smoke/split_bundled_program.py atol rtol
    python "${split}" "${models}/smoke.bpte" --atol 0 --rtol 0 --repeat 2
    python "${split}" "${models}/mul.bpte" --atol 0 --rtol 1.2e-7 --repeat 2
    read -r atol rtol < <(python -c '
import json, sys
(entry,) = json.load(open(sys.argv[1]))
print(entry["atol"], entry["rtol"])
' "${dir}/espnn/manifest.json")
    python "${split}" "${models}/delegate.bpte" --atol "${atol}" --rtol "${rtol}" --repeat 2
    python "${split}" "${models}/wakeword.bpte" --atol 1e-6 --rtol 0
    rm -r "${models}"/*.bpte "${dir}/espnn"

    local component_dir="${dir}/component/executorch"
    logged "${logs}/generate.log" bash examples/espressif/build_idf_component.sh \
        --allow-dev-stamp --out "${component_dir}"
    local version
    version=$(sed -n 's/^version: "\(.*\)"$/\1/p' "${component_dir}/idf_component.yml")
    (cd "${component_dir}" && compote component pack --name executorch --dest-dir "${dir}/dist")
    python .ci/scripts/esp_idf_mirror.py "${dir}/dist/executorch_${version}.tgz" "${dir}/mirror" \
        --namespace pytorch --name executorch
    local model
    for model in smoke mul delegate wakeword; do
        python "${component_dir}/tools/gen_kernels.py" --pte "${models}/${model}.pte" \
            --out "${models}/${model}_kernels"
    done
    python "${component_dir}/tools/gen_kernels.py" --pte "${models}/smoke.pte" \
        --pte "${models}/mul.pte" --out "${models}/smoke_mul_kernels"
    cp "${models}/smoke.pte" "${models}/all_ops.pte"
    cp "${models}/smoke.io" "${models}/all_ops.io"
    local oplists=() yaml
    mkdir -p "${dir}/all_ops"
    for yaml in "${component_dir}"/tools/libraries/*.yaml; do
        python -m executorch.codegen.tools.gen_oplist \
            --output_path="${dir}/all_ops/$(basename "${yaml}")" --ops_schema_yaml_path="${yaml}"
        oplists+=(--oplist "${dir}/all_ops/$(basename "${yaml}")")
    done
    python "${component_dir}/tools/gen_kernels.py" --pte "${models}/all_ops.pte" "${oplists[@]}" \
        --out "${models}/all_ops_kernels"
    rm -r "${dir}/all_ops"
    echo "${version}" > "${dir}/version"
}

build() {
    local shard=${3:?shard} shards=${4:?shards}
    local idf_version="${ESPRESSIF_IDF_VERSION:-v6.1}"
    local work="${ESPRESSIF_BUILD_DIR:-${et_root}/cmake-out-esp-idf-nightly-${idf_version}-${shard}}"
    export ESPRESSIF_TOOLS_DIR="${ESPRESSIF_TOOLS_DIR:-${RUNNER_TEMP:-/tmp}/executorch-espressif-${idf_version}}"
    mkdir -p "${work}"
    work=$(cd "${work}" && pwd)

    # ESP-IDF's QEMU needs these libraries, and setup.sh fails its install check
    # without them.
    if ! python -c '
import ctypes
ctypes.CDLL("libSDL2-2.0.so.0")
ctypes.CDLL("libslirp.so.0")' 2> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y --no-install-recommends libsdl2-2.0-0 libslirp0
    fi
    logged "${work}/setup.log" bash examples/espressif/setup.sh --minimal \
        --idf-version="${idf_version}" --targets="$(IFS=,; echo "${targets[*]}")"
    # shellcheck source=/dev/null
    source "${ESPRESSIF_TOOLS_DIR}/setup_path.sh"

    # A fresh ESP-IDF Python environment, which cannot import executorch, and
    # idf-build-apps in a venv of its own, so it changes nothing ESP-IDF runs.
    export IDF_PYTHON_ENV_PATH="${work}/idf_python_env"
    mkdir -p "${IDF_PYTHON_ENV_PATH}"
    logged "${work}/idf_python_env.log" python "${IDF_PATH}/tools/idf_tools.py" install-python-env
    "${IDF_PYTHON_ENV_PATH}/bin/python" -m venv "${work}/idf_build_apps"
    "${work}/idf_build_apps/bin/python" -m pip install --quiet "idf-build-apps>=3.1,<4"
    export IDF_COMPONENT_STORAGE_URL="file://${dir}/mirror;default"
    local version
    version=$(cat "${dir}/version")

    # Built from a copy, so the component manager's state and the consumer's
    # generated manifest stay out of the source tree.
    local sources="${work}/sources"
    rm -rf "${sources}"
    mkdir -p "${sources}/examples/espressif/idf_component_smoke"
    cp -r examples/espressif/idf_component_consumer "${sources}/examples/espressif/"
    cp -r examples/espressif/idf_component_smoke/main "${sources}/examples/espressif/idf_component_smoke/"
    local app=examples/espressif/idf_component_consumer
    (cd "${sources}" && "${work}/idf_build_apps/bin/idf-build-apps" find -p "${app}" --target all \
        --default-build-targets "${targets[@]}" --config-rules 'sdkconfig.ci.*=' \
        --manifest-files "${app}/.build-test-rules.yml" --output-format raw \
        --output "${work}/apps.txt")
    # One "target config test-disabled" line per app of this shard.
    python - "${work}/apps.txt" "${shard}" "${shards}" > "${work}/shard.txt" <<'PY'
import json
import sys

shard, shards = int(sys.argv[2]), int(sys.argv[3])
for index, line in enumerate(open(sys.argv[1])):
    if index % shards == shard - 1:
        app = json.loads(line)
        print(app["target"], app["config_name"], str(bool(app["test_comment"])).lower())
PY
    if [[ ! -s "${work}/shard.txt" ]]; then
        echo "FAIL: shard ${shard}/${shards} has no apps; see ${work}/apps.txt" >&2
        exit 1
    fi

    local project="${sources}/${app}" builds="${work}/builds" failures=()
    local target config test_disabled name log model idf_args
    # On fd 3, so the emulators cannot read the list from stdin.
    while read -r target config test_disabled <&3; do
        name="${target}-${config}"
        log="${work}/build-${name}.log"
        idf_args=(-C "${project}" -B "${builds}/${name}" -DSDKCONFIG="${builds}/${name}/sdkconfig"
            "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.ci.${config}"
            -DET_COMPONENT_VERSION="${version}" -DET_CONSUMER_MODELS_DIR="${dir}/models")
        if ! IDF_TARGET="${target}" "${IDF_PYTHON_ENV_PATH}/bin/python" "${IDF_PATH}/tools/idf.py" \
            "${idf_args[@]}" build > "${log}" 2>&1; then
            echo "FAIL: ${name} did not build; see ${log}" >&2
            tail -n 30 "${log}" >&2
            failures+=("${name} (build)")
            continue
        fi
        if ! settings_applied "${project}/sdkconfig.ci.${config}" "${builds}/${name}/sdkconfig"; then
            failures+=("${name} (settings)")
            continue
        fi
        if [[ "${test_disabled}" == true ]]; then
            echo "OK: ${name} built (its test is disabled)"
            continue
        fi
        model=$(sed -n 's/^CONFIG_ET_CONSUMER_MODEL="\(.*\)"$/\1/p' "${builds}/${name}/sdkconfig")
        log="${work}/emulator-${name}.log"
        run_firmware "${builds}/${name}" "${target}" "${log}" "${idf_args[@]}"
        if grep -qF "${model}: PASS" "${log}" && grep -qF "ET_COMPONENT_SMOKE: PASS" "${log}"; then
            echo "OK: ${name} passed on the ${target} emulator"
        else
            echo "FAIL: ${name} did not print '${model}: PASS'; see ${log}" >&2
            tail -n 30 "${log}" >&2
            failures+=("${name} (run)")
        fi
    done 3< "${work}/shard.txt"

    if (( ${#failures[@]} )); then
        printf 'FAIL: %s\n' "${failures[@]}" >&2
        exit 1
    fi
    echo "PASS: shard ${shard}/${shards} of the ESP-IDF ${idf_version} matrix built and ran."
}

# Checks that every setting of a sdkconfig.ci.* file reached the build's
# sdkconfig, since ESP-IDF only warns about a symbol it does not know:
# defaults file, sdkconfig.
settings_applied() {
    local defaults=$1 sdkconfig=$2 line applied=true
    while IFS= read -r line; do
        [[ "${line}" == CONFIG_* ]] || continue
        if [[ "${line}" == *=n ]]; then
            line="# ${line%=n} is not set"
        fi
        if ! grep -qxF "${line}" "${sdkconfig}"; then
            echo "FAIL: ${sdkconfig} does not have '${line}' from ${defaults}" >&2
            applied=false
        fi
    done < "${defaults}"
    [[ "${applied}" == true ]]
}

# Runs a build on its emulator: build dir, target, log, then idf.py arguments.
run_firmware() {
    local build=$1 target=$2 log=$3
    shift 3
    local args=("$@")
    if [[ "${target}" == esp32 ]]; then
        # QEMU has no exit-on string, so stop it once the app is done.
        IDF_TARGET="${target}" setsid "${IDF_PYTHON_ENV_PATH}/bin/python" \
            "${IDF_PATH}/tools/idf.py" "${args[@]}" qemu < /dev/null > "${log}" 2>&1 &
        local pid=$!
        for _ in $(seq 240); do
            if grep -qF "Program complete." "${log}"; then
                break
            fi
            sleep 0.5
        done
        kill -- "-${pid}" 2> /dev/null || true
        wait "${pid}" || true
    else
        # Without a merged binary there is nothing to run; the caller reports
        # the missing PASS marker with the log.
        IDF_TARGET="${target}" "${IDF_PYTHON_ENV_PATH}/bin/python" "${IDF_PATH}/tools/idf.py" \
            "${args[@]}" merge-bin > "${log}" 2>&1 || return 0
        timeout 130s esp-emu --chip "${target}" --firmware "${build}/merged-binary.bin" \
            --elf "${build}/et_component_smoke.elf" --net user --timeout 120s \
            --exit-on 'Program complete.' >> "${log}" 2>&1 || true
    fi
}

case "${mode}" in
    prepare) prepare ;;
    build) build "$@" ;;
    *) echo "unknown mode ${mode}; use prepare or build" >&2; exit 1 ;;
esac
