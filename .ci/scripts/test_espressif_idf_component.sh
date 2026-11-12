#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Generates the ExecuTorch ESP-IDF component and runs the idf_component_smoke
# consumer on the emulator with three models: the portable-kernel smoke model, a
# mul on the esp_nn:: op library and an ESP-NN delegate program. An ESP32-S3
# build runs all three with a kernel bundle the build makes itself; an ESP32-C6
# build runs the first two, with the Cadence kernels too, and must not link the
# delegate; two more ESP32-S3 builds run the delegate alone, from a bundle with
# no kernel libraries and from executorch_add_backend with no bundle, and the
# component's own example runs too. Also checks the version stamping, the
# manifest and SBOM, that stale or mismatched bundles fail at configure time,
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

# Needs only git and a Python with PyYAML, so a stamp regression fails before
# the toolchain installs.
bash examples/espressif/test_version_stamp.sh

bash examples/espressif/setup.sh --minimal --targets=esp32s3,esp32c6 2>&1 \
    | tee "${build_dir}/setup.log"
# shellcheck source=/dev/null
source "${ESPRESSIF_TOOLS_DIR}/setup_path.sh"

python examples/espressif/export_smoke_model.py --output "${build_dir}/smoke.bpte" \
    2>&1 | tee "${build_dir}/export.log"
python examples/espressif/idf_component_smoke/export_op_library_model.py \
    --output "${build_dir}/mul.bpte" 2>&1 | tee -a "${build_dir}/export.log"
python examples/espressif/export_espnn_test_programs.py --output-dir "${build_dir}/espnn" \
    --only conv-act-pool-linear 2>&1 | tee -a "${build_dir}/export.log"
cp "${build_dir}/espnn/conv-act-pool-linear.bpte" "${build_dir}/delegate.bpte"
split=examples/espressif/idf_component_smoke/split_bundled_program.py
# Each case twice, so a second execute of the same method is checked too.
python "${split}" "${build_dir}/smoke.bpte" --atol 0 --rtol 0 --repeat 2
# test_mul.py's tolerance, torch.finfo(torch.float32).eps (1.1920929e-07),
# rounded up.
python "${split}" "${build_dir}/mul.bpte" --atol 0 --rtol 1.2e-7 --repeat 2
read -r atol rtol < <(python -c '
import json, sys
(entry,) = json.load(open(sys.argv[1]))
print(entry["atol"], entry["rtol"])
' "${build_dir}/espnn/manifest.json")
python "${split}" "${build_dir}/delegate.bpte" --atol "${atol}" --rtol "${rtol}" --repeat 2
cpu_models="${build_dir}/smoke.pte;${build_dir}/mul.pte"
models="${cpu_models};${build_dir}/delegate.pte"

component_dir="${build_dir}/components/executorch"
kernels_dir="${build_dir}/smoke_kernels"
bash examples/espressif/build_idf_component.sh --allow-dev-stamp --out "${component_dir}" \
    2>&1 | tee "${build_dir}/generate.log"
compote manifest lint "${component_dir}"
(cd "${component_dir}" && esp-idf-sbom manifest validate idf_component.yml sbom.yml sbom_*.yml)

# The release path that publishing takes: a clean clone of HEAD, tagged as the
# release in the clone only, generated without --allow-dev-stamp.
release_version=$(cat version.txt)
release_clone="${build_dir}/release-clone"
rm -rf "${release_clone}" "${build_dir}/release_component" "${build_dir}/rc_component" \
    "${build_dir}/dev_component"
git clone -q --shared --no-checkout "${et_root}" "${release_clone}"
git -C "${release_clone}" checkout -q --detach "$(git rev-parse HEAD)"
git -C "${release_clone}" submodule init -q third-party/flatbuffers
git -C "${release_clone}" config submodule.third-party/flatbuffers.url \
    "${et_root}/third-party/flatbuffers"
git -C "${release_clone}" -c protocol.file.allow=always submodule update -q third-party/flatbuffers
git -C "${release_clone}" tag -f "v${release_version}" > /dev/null
bash "${release_clone}/examples/espressif/build_idf_component.sh" --version-suffix "" \
    --out "${build_dir}/release_component" 2>&1 | tee "${build_dir}/generate-release.log"
compote manifest lint "${build_dir}/release_component"
python - "${build_dir}/release_component" "${release_version}" "$(git rev-parse HEAD)" \
    "${release_clone}" <<'PY'
import re
import sys
from pathlib import Path

component, version, commit, clone = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
lines = (component / "VERSION").read_text().splitlines()
for line in (f"component_version: {version}", f"executorch_commit: {commit}"):
    if line not in lines:
        sys.exit(f"FAIL: the release component's VERSION has no '{line}'")
if not any(l.startswith("generator: ") and l.endswith(" --version-suffix ''") for l in lines):
    sys.exit("FAIL: the release component's VERSION does not record --version-suffix ''")
sys.path.insert(0, clone)
from install_requirements import TORCH_URL_BASE

if not any(re.fullmatch(r"torch_requirement: torch==\d+\.\d+\.\d+\S*", l) for l in lines):
    sys.exit("FAIL: the release component's VERSION has no torch_requirement: torch==X.Y.Z")
if f"torch_index: {TORCH_URL_BASE}/cpu" not in lines:
    sys.exit(f"FAIL: the release component's VERSION has no torch_index: {TORCH_URL_BASE}/cpu")
if f'version: "{version}"' not in (component / "idf_component.yml").read_text().splitlines():
    sys.exit(f"FAIL: the release component's idf_component.yml is not version {version}")
readme = (component / "README.md").read_text().splitlines()
torch = next(l.split(": ", 1)[1] for l in lines if l.startswith("torch_requirement: "))
for line in (
    f'pip install "{torch}" --index-url {TORCH_URL_BASE}/cpu',
    f'pip install "executorch=={version}"',
):
    if line not in readme:
        sys.exit(f"FAIL: the release component's README.md has no '{line}'")
print(f"OK: a clean checkout tagged v{version} generates release {version}")
PY
# An RC at the same commit, which a release commit can also be.
git -C "${release_clone}" tag -f "v${release_version}-rc1" > /dev/null
bash "${release_clone}/examples/espressif/build_idf_component.sh" --version-suffix -rc.1 \
    --out "${build_dir}/rc_component" 2>&1 | tee "${build_dir}/generate-rc.log"
compote manifest lint "${build_dir}/rc_component"
if ! grep -qxF "component_version: ${release_version}-rc.1" "${build_dir}/rc_component/VERSION"; then
    echo "FAIL: the RC component's VERSION is not ${release_version}-rc.1" >&2
    exit 1
fi
rc_install="pip install \"executorch==${release_version}\" --extra-index-url https://download.pytorch.org/whl/test/cpu"
if ! grep -qxF "${rc_install}" "${build_dir}/rc_component/README.md"; then
    echo "FAIL: the RC component's README.md has no '${rc_install}'" >&2
    exit 1
fi
# A dev version from a local nightly branch at the same commit.
git -C "${release_clone}" branch -f nightly HEAD
bash "${release_clone}/examples/espressif/build_idf_component.sh" --version-suffix -dev \
    --out "${build_dir}/dev_component" 2>&1 | tee "${build_dir}/generate-dev.log"
dev_date=$(TZ=UTC git log -1 --format=%cd --date=format-local:%Y%m%d HEAD)
if ! grep -qxF "component_version: ${release_version}-dev.${dev_date}" \
    "${build_dir}/dev_component/VERSION"; then
    echo "FAIL: the dev component's VERSION is not ${release_version}-dev.${dev_date}" >&2
    exit 1
fi
dev_install="pip install \"executorch==${release_version}.dev${dev_date}\" --extra-index-url https://download.pytorch.org/whl/nightly/cpu"
if ! grep -qxF "${dev_install}" "${build_dir}/dev_component/README.md"; then
    echo "FAIL: the dev component's README.md has no '${dev_install}'" >&2
    exit 1
fi
echo "OK: the clone's local nightly branch generates dev ${release_version}-dev.${dev_date}"
echo "OK: the clean checkout tagged v${release_version}-rc1 generates RC ${release_version}-rc.1"
gen_kernels() {
    local out=$1
    shift
    python "${component_dir}/tools/gen_kernels.py" "$@" --out "${out}" \
        2>&1 | tee -a "${build_dir}/gen_kernels.log"
}
gen_kernels "${kernels_dir}" --pte "${build_dir}/smoke.pte" --pte "${build_dir}/mul.pte" \
    --pte "${build_dir}/delegate.pte"
# The ESP32-C6 build also registers the ops of a Cadence-quantized streaming
# wake word model, so CI compiles and links the Cadence generic kernels.
wakeword_ops=aten::_softmax.out,aten::_to_copy.out,aten::cat.out,aten::copy_
wakeword_ops+=,aten::permute_copy.out,aten::slice_copy.Tensor_out
wakeword_ops+=,aten::squeeze_copy.dim_out,aten::transpose_copy.int_out
wakeword_ops+=,cadence::dequantize_per_tensor.out,cadence::quantize_per_tensor.out
wakeword_ops+=,cadence::quantized_conv2d_nhwc.per_tensor_out
wakeword_ops+=,cadence::quantized_fully_connected.per_tensor_out
wakeword_ops+=,cadence::quantized_relu.per_tensor_out
gen_kernels "${build_dir}/cpu_kernels" --pte "${build_dir}/smoke.pte" \
    --pte "${build_dir}/mul.pte" --ops "${wakeword_ops}"
gen_kernels "${build_dir}/delegate_kernels" --pte "${build_dir}/delegate.pte"
python - "${build_dir}/delegate_kernels/kernels.json" <<'PY'
import json
import sys

libraries = json.load(open(sys.argv[1]))["libraries"]
if libraries:
    sys.exit(f"FAIL: the delegate-only bundle has kernel libraries {sorted(libraries)}")
PY

project=examples/espressif/idf_component_smoke

# Runs idf.py on the smoke consumer: build name, target, then idf.py arguments.
consumer() {
    local name=$1 target=$2
    shift 2
    IDF_TARGET="${target}" idf.py -C "${project}" -B "${build_dir}/firmware-${name}" \
        -DSDKCONFIG="${build_dir}/sdkconfig-${name}" "$@" \
        > "${log:-${build_dir}/firmware-${name}.log}" 2>&1
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
    -DET_SMOKE_MODELS="${build_dir}/stale/smoke.pte;${build_dir}/mul.pte;${build_dir}/delegate.pte" \
    -DET_SMOKE_KERNELS="${kernels_dir}"

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
    -DET_SMOKE_MODELS="${models}" \
    -DET_SMOKE_KERNELS="${build_dir}/mismatch_kernels"

# A bundle made by a component of the same version from another commit, as two
# development builds from the same day are.
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
    -DET_SMOKE_MODELS="${models}" \
    -DET_SMOKE_KERNELS="${build_dir}/mismatch_kernels"

# Without an ExecuTorch Python to generate one, a missing bundle is an error.
python -m venv --without-pip --clear "${build_dir}/bare_python"
EXECUTORCH_PYTHON="${build_dir}/bare_python/bin/python" expect_configure_error \
    no-bundle "cannot import executorch" -DET_SMOKE_MODELS="${models}"
expect_configure_error relative-python "must be the absolute path" \
    -DET_SMOKE_MODELS="${models}" -DEXECUTORCH_PYTHON=python

# A model the bundle was not made from, declared instead of the one it was.
mkdir -p "${build_dir}/other"
cp "${build_dir}/stale/smoke.pte" "${build_dir}/other/other.pte"
cp "${build_dir}/smoke.io" "${build_dir}/other/other.io"
expect_configure_error other-model \
    "other.pte: not in the bundle smoke.pte: in the bundle, not declared" \
    -DET_SMOKE_MODELS="${build_dir}/other/other.pte;${build_dir}/mul.pte;${build_dir}/delegate.pte" \
    -DET_SMOKE_KERNELS="${kernels_dir}"

rm -rf "${build_dir}/no_count_kernels"
cp -r "${kernels_dir}" "${build_dir}/no_count_kernels"
rm -r "${build_dir}/no_count_kernels/include"
expect_configure_error no-kernel-count "has no kernel count" \
    -DET_SMOKE_MODELS="${models}" -DET_SMOKE_KERNELS="${build_dir}/no_count_kernels"

# A bundle used without declaring its models is used as is, with a warning that
# names them.
consumer as-is esp32s3 -DET_COMPONENT_DIR="${component_dir}" \
    -DET_SMOKE_MODELS="${models}" -DET_SMOKE_UNDECLARED=1 \
    -DET_SMOKE_KERNELS="${kernels_dir}" reconfigure
if ! tr -s ' \n' '  ' < "${build_dir}/firmware-as-is.log" \
    | grep -qF "is not checked against delegate.pte, mul.pte, smoke.pte"; then
    echo "FAIL: no warning naming the bundle's models; see ${build_dir}/firmware-as-is.log" >&2
    exit 1
fi
echo "OK: as-is warned that the bundle's models are not checked"

# An app that runs a model while declaring nothing fails to link.
if consumer undeclared esp32s3 -DET_COMPONENT_DIR="${component_dir}" \
    -DET_SMOKE_MODELS="${models}" -DET_SMOKE_UNDECLARED=1 build; then
    echo "FAIL: undeclared linked; expected et_requires_executorch_add_kernels to be undefined" >&2
    exit 1
fi
if ! grep -qF "undefined reference to \`et_requires_executorch_add_kernels'" \
    "${build_dir}/firmware-undeclared.log"; then
    echo "FAIL: undeclared did not fail on et_requires_executorch_add_kernels; see firmware-undeclared.log" >&2
    exit 1
fi
echo "OK: undeclared failed to link on et_requires_executorch_add_kernels"

# A kernel registry smaller than the bundle; all-esp32s3 below uses one larger
# than it, and the other builds the one sized from the bundle.
printf 'CONFIG_ET_MAX_KERNEL_NUM=1\n' > "${build_dir}/small_registry.defaults"
printf 'CONFIG_ET_MAX_KERNEL_NUM=64\n' > "${build_dir}/large_registry.defaults"
expect_configure_error small-registry "CONFIG_ET_MAX_KERNEL_NUM=1 is below" \
    "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;${build_dir}/small_registry.defaults" \
    -DET_SMOKE_MODELS="${models}" -DET_SMOKE_KERNELS="${kernels_dir}"

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
    log="${build_dir}/size-${name}.log" consumer "${name}" esp32s3 \
        -DET_COMPONENT_DIR="${dir}" size --format json2 \
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
    -DET_SMOKE_MODELS="${models}" -DET_SMOKE_KERNELS="${kernels_dir}" reconfigure
if ! tr -s ' \n' '  ' < "${build_dir}/firmware-nano.log" \
    | grep -qF "ExecuTorch logging is DISABLED"; then
    echo "FAIL: no nano printf warning; see ${build_dir}/firmware-nano.log" >&2
    exit 1
fi

# Runs a build on the emulator and checks its output: build name, chip, then
# the per-model markers.
run_firmware() {
    local name=$1 chip=$2
    shift 2
    local firmware_dir="${build_dir}/firmware-${name}"
    IDF_TARGET="${chip}" idf.py -C "${project}" -B "${firmware_dir}" merge-bin \
        > "${build_dir}/merge-${name}.log" 2>&1
    local elf=("${firmware_dir}"/*.elf)
    timeout 130s esp-emu --chip "${chip}" \
        --firmware "${firmware_dir}/merged-binary.bin" --elf "${elf[0]}" \
        --net user --timeout 120s --exit-on 'Program complete.' \
        2>&1 | tee "${build_dir}/emulator-${name}.log"
    for marker in "$@" "ET_COMPONENT_SMOKE: PASS" "Program complete."; do
        if ! grep -qF "${marker}" "${build_dir}/emulator-${name}.log"; then
            echo "FAIL: ${name} printed no '${marker}'; see emulator-${name}.log" >&2
            exit 1
        fi
    done
    echo "OK: ${name} passed on the ${chip} emulator"
}

# No bundle declared: the build makes one with its own Python, which has
# executorch here, and it must match the one gen_kernels.py made above.
consumer all-esp32s3 esp32s3 -DET_COMPONENT_DIR="${component_dir}" \
    "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;${build_dir}/large_registry.defaults" \
    -DET_SMOKE_MODELS="${models}" build
diff -r "${build_dir}/firmware-all-esp32s3/executorch_kernels" "${kernels_dir}"
run_firmware all-esp32s3 esp32s3 "smoke: PASS" "mul: PASS" "delegate: PASS"
# The SBOM esp-idf-sbom makes for a project using the component: one package
# per vendored library, each with its version.
esp-idf-sbom create "${build_dir}/firmware-all-esp32s3/project_description.json" \
    -o "${build_dir}/sbom-all-esp32s3.spdx"
python - "${build_dir}/sbom-all-esp32s3.spdx" \
    "$(cat .ci/docker/ci_commit_pins/pytorch.txt)" <<'PY'
import re
import sys

spdx, pytorch = open(sys.argv[1]).read(), sys.argv[2]
packages = {}
for block in spdx.split("\n\n"):
    name = re.search(r"^PackageName: (.*)$", block, re.M)
    if name:
        version = re.search(r"^PackageVersion: (.*)$", block, re.M)
        packages.setdefault(name.group(1), []).append(version and version.group(1))
for name, version in (("c10", pytorch), ("torch-headeronly", pytorch)):
    if packages.get(name) != [version]:
        sys.exit(f"FAIL: the SBOM has {name} {packages.get(name)}, expected [{version!r}]")
flatbuffers = packages.get("flatbuffers", [])
if len(flatbuffers) != 1 or not flatbuffers[0]:
    sys.exit(f"FAIL: the SBOM has flatbuffers {flatbuffers}, expected one with a version")
if "{}" in spdx:
    sys.exit("FAIL: the SBOM has an unsubstituted {} in a cpe or purl")
print("OK: the SBOM lists c10, torch-headeronly and flatbuffers once each, with versions")
PY

# A project may provide esp-nn as its own component instead of the managed one.
local_project="${build_dir}/local-esp-nn"
rm -rf "${local_project}"
mkdir -p "${local_project}/components"
cp -r "${project}/CMakeLists.txt" "${project}/main" "${project}/sdkconfig.defaults" \
    "${local_project}/"
cp -r "${project}/managed_components/espressif__esp-nn" "${local_project}/components/esp-nn"
project="${local_project}" consumer local-esp-nn esp32s3 -DET_COMPONENT_DIR="${component_dir}" \
    -DET_SMOKE_MODELS="${build_dir}/delegate.pte" \
    -DET_SMOKE_KERNELS="${build_dir}/delegate_kernels" reconfigure
python - "${build_dir}/firmware-local-esp-nn/project_description.json" <<'PY'
import json
import sys

components = json.load(open(sys.argv[1]))["build_components"]
espnn = sorted(c for c in components if c.endswith("esp-nn"))
if espnn != ["esp-nn"]:
    sys.exit(f"FAIL: local-esp-nn builds {espnn}, not only its own esp-nn")
PY
echo "OK: a project with its own components/esp-nn configures with it"

consumer cpu-esp32c6 esp32c6 -DET_COMPONENT_DIR="${component_dir}" \
    -DET_SMOKE_MODELS="${cpu_models}" -DET_SMOKE_KERNELS="${build_dir}/cpu_kernels" build
# The esp_nn:: op library must not pull in the delegate.
riscv32-esp-elf-nm -C "${build_dir}/firmware-cpu-esp32c6/et_component_smoke.elf" \
    > "${build_dir}/nm-cpu-esp32c6.txt"
if grep -q EspNNBackend "${build_dir}/nm-cpu-esp32c6.txt"; then
    echo "FAIL: cpu-esp32c6 links EspNNBackend, which none of its models uses" >&2
    exit 1
fi
run_firmware cpu-esp32c6 esp32c6 "smoke: PASS" "mul: PASS"

consumer delegate-esp32s3 esp32s3 -DET_COMPONENT_DIR="${component_dir}" \
    -DET_SMOKE_MODELS="${build_dir}/delegate.pte" \
    -DET_SMOKE_KERNELS="${build_dir}/delegate_kernels" build
xtensa-esp32s3-elf-nm -C "${build_dir}/firmware-delegate-esp32s3/et_component_smoke.elf" \
    > "${build_dir}/nm-delegate-esp32s3.txt"
if ! grep -q EspNNBackend "${build_dir}/nm-delegate-esp32s3.txt"; then
    echo "FAIL: delegate-esp32s3 does not link EspNNBackend" >&2
    exit 1
fi
# A bundle without kernel libraries or prim ops never builds the kernels.
kernel_lib=$(find "${build_dir}/firmware-delegate-esp32s3" -name '*executorch_kernels.a')
if [[ -n "${kernel_lib}" ]]; then
    echo "FAIL: delegate-esp32s3 built ${kernel_lib} for a bundle without kernels" >&2
    exit 1
fi
run_firmware delegate-esp32s3 esp32s3 "delegate: PASS"

# Models not known at build time: only the delegate is declared.
consumer backend-esp32s3 esp32s3 -DET_COMPONENT_DIR="${component_dir}" \
    -DET_SMOKE_MODELS="${build_dir}/delegate.pte" -DET_SMOKE_BACKENDS=EspNNBackend build
run_firmware backend-esp32s3 esp32s3 "delegate: PASS"
# Without a bundle nothing registers kernels, so the registry reserves no room.
xtensa-esp32s3-elf-nm -S -C "${build_dir}/firmware-backend-esp32s3/et_component_smoke.elf" \
    > "${build_dir}/nm-backend-esp32s3.txt"
registry=$(awk '/registered_kernels_data/ {print $2}' "${build_dir}/nm-backend-esp32s3.txt")
if [[ -z "${registry}" ]]; then
    echo "FAIL: backend-esp32s3 has no registered_kernels_data; see nm-backend-esp32s3.txt" >&2
    exit 1
fi
if (( 16#${registry} > 64 )); then
    echo "FAIL: backend-esp32s3 reserves $((16#${registry})) B for kernels it never registers" >&2
    exit 1
fi

# The example the component ships, built through its override_path.
example="${component_dir}/examples/hello_executorch"
project="${example}" consumer example esp32s3 build
project="${example}" run_firmware example esp32s3 "smoke: PASS"

echo "PASS: generated ESP-IDF component ran every build on the emulator."
