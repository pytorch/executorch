#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
EXECUTORCH_PROJ_ROOT="$(realpath "${SCRIPT_DIR}/../..")"
ZEPHYR_README_PATH="zephyr/README.md"

ZEPHYR_SAMPLES_README_PATH="zephyr/samples/hello-executorch/README.md"
TARGETS_ARG="${TARGET_LIST:-}"

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --zephyr-samples-readme-path <path>  README containing test_<TARGET>* command blocks
  --targets <list>                    Comma-separated target list, e.g. ethos-u55,cortex-m55,ethos-u85
  -h, --help                           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --targets)
      TARGETS_ARG="$2"
      shift 2
      ;;
    --zephyr-samples-readme-path)
      ZEPHYR_SAMPLES_README_PATH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${TARGETS_ARG}" ]]; then
  echo "ERROR: --targets or TARGET_LIST must be set" >&2
  usage >&2
  exit 1
fi

IFS=',' read -r -a TARGETS <<< "${TARGETS_ARG}"

export EXECUTORCH_PROJ_ROOT

cd "${EXECUTORCH_PROJ_ROOT}"

# Source utility scripts.
. .ci/scripts/utils.sh
. .ci/scripts/zephyr-utils.sh

run_target_test_blocks_from_readme() {
  local readme_path="$1"
  local target="$2"
  local resolved_readme_path marker markers

  resolved_readme_path="$(_utils_path_from_root "${readme_path}")"
  markers="$(awk -v target="${target}" '
    {
      line = $0
      while (match(line, /<!--[[:space:]]*RUN[[:space:]]+[^>]*-->/)) {
        marker = substr(line, RSTART, RLENGTH)
        if (index(marker, "<!-- RUN test_" target) == 1) {
          print marker
        }
        line = substr(line, RSTART + RLENGTH)
      }
    }
  ' "${resolved_readme_path}")"

  if [[ -z "${markers}" ]]; then
    echo "ERROR: No test blocks matching <!-- RUN test_${target}* --> in ${readme_path}" >&2
    return 1
  fi

  while IFS= read -r marker; do
    echo "---- ${target} ${marker} ----"
    run_command_block_from_readme "${readme_path}" "${marker}"
  done <<< "${markers}"
}

# Check that zephyr/README.md and zephyr/executorch.yaml are in sync.
verify_zephyr_readme

# Based on instructions in zephyr/README.md and the selected sample README.
run_command_block_from_readme "${ZEPHYR_README_PATH}" "<!-- RUN install_reqs -->"

# Make sure to backup the zephyr_scratch folder if it exists to allow for local
# testing that does not lose code/data.
if [[ -d "zephyr_scratch" ]]; then
  mv "zephyr_scratch" "zephyr_scratch.backup.$(date +%Y%m%d%H%M%S)"
fi
mkdir -p zephyr_scratch/

cd zephyr_scratch
export ZEPHYR_PROJ_ROOT="$(realpath "$(pwd)")"

echo "---- Zephyr SDK ----"
# Use ZephyrSDK if on the disk (e.g. setup in the docker)
# Check for a zephyr-sdk-0.17.4 directory and make a symlink if found in parent directories
if sdk_dir=$(find ../../.. -maxdepth 4 -type d -name 'zephyr-sdk-0.17.4' -print -quit) && [ -n "${sdk_dir}" ]; then
  echo "---- Found pre downloaded Zephyr SDK in ${sdk_dir} ----"
  ln -s "${sdk_dir}" .
fi

# Download and setup Zephyr SDK 0.17.4 if not already present
if [ ! -d "zephyr-sdk-0.17.4" ]; then
  echo "---- Downloading Zephyr SDK ----"
  wget https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v0.17.4/zephyr-sdk-0.17.4_linux-x86_64.tar.xz
  tar -xf zephyr-sdk-0.17.4_linux-x86_64.tar.xz
  rm -f zephyr-sdk-0.17.4_linux-x86_64.tar.xz*
fi

./zephyr-sdk-0.17.4/setup.sh -c -t arm-zephyr-eabi
export ZEPHYR_SDK_INSTALL_DIR=$(realpath ./zephyr-sdk-0.17.4)

cd ${ZEPHYR_PROJ_ROOT}

run_command_block_from_readme "${ZEPHYR_README_PATH}" "<!-- RUN west_init -->"

cp "${EXECUTORCH_PROJ_ROOT}/zephyr/executorch.yaml" zephyr/submanifests/

run_command_block_from_readme "${ZEPHYR_README_PATH}" "<!-- RUN west_config -->"

# Switch to executorch in this PR e.g. replace modules/lib/executorch with the
# root folder of this repo instead of doing a re-checkout and figuring out the
# correct commit hash.
rm -Rf modules/lib/executorch
ln -s "${EXECUTORCH_PROJ_ROOT}" modules/lib/executorch

# Setup git local user for Executorch git to allow
# modules/lib/executorch/examples/arm/setup.sh to run inside CI later.
if ! git config --get user.name >/dev/null 2>&1; then
  git config --global user.name "Github Executorch"
fi
if ! git config --get user.email >/dev/null 2>&1; then
  git config --global user.email "github_executorch@arm.com"
fi

run_command_block_from_readme "${ZEPHYR_README_PATH}" "<!-- RUN install_executorch -->"
run_command_block_from_readme "${ZEPHYR_README_PATH}" "<!-- RUN install_arm_tools -->"

for TARGET in "${TARGETS[@]}"; do
  TARGET="$(echo "${TARGET}" | xargs)"

  echo "---- ${TARGET} ----"
  rm -Rf build

  if [[ ${TARGET} == "ethos-u55" || ${TARGET} == "cortex-m55" ]]; then
    BOARD="corstone300"
  elif [[ ${TARGET} == "ethos-u85" ]]; then
    BOARD="corstone320"
  else
    echo "Fail unsupported target selection ${TARGET}"
    exit 1
  fi

  echo "---- ${TARGET} Board ${BOARD} FVP setup ----"
  run_command_block_from_readme "${ZEPHYR_SAMPLES_README_PATH}" "<!-- RUN setup_${BOARD} -->"

  # Run all blocks that match <!-- RUN test_${target}* -->
  run_target_test_blocks_from_readme "${ZEPHYR_SAMPLES_README_PATH}" "${TARGET}"
done
