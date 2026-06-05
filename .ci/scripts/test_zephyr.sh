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
SKIP_ZEPHYR_SETUP=0
SETUP_ONLY=0
ZEPHYR_DEV_ROOT="${ZEPHYR_DEV_ROOT:-zephyr_dev_root}"

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --zephyr-samples-readme-path <path>  README containing test_<TARGET>* command blocks
  --targets <list>                    Comma-separated target list, e.g. ethos-u55,cortex-m55,ethos-u85
  --zephyr-dev-root <path>             Zephyr workspace directory (default: ${ZEPHYR_DEV_ROOT})
  --skip-zephyr-setup                  Reuse an existing Zephyr workspace
  --setup-only                         Set up Zephyr workspace and exit without running sample tests
  -h, --help                           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zephyr-samples-readme-path)
      ZEPHYR_SAMPLES_README_PATH="$2"
      shift 2
      ;;
    --zephyr-dev-root)
      ZEPHYR_DEV_ROOT="$2"
      shift 2
      ;;
    --targets)
      TARGETS_ARG="$2"
      shift 2
      ;;
    --skip-zephyr-setup)
      SKIP_ZEPHYR_SETUP=1
      shift
      ;;
    --setup-only)
      SETUP_ONLY=1
      shift
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

if [[ ${SKIP_ZEPHYR_SETUP} -eq 1 && ${SETUP_ONLY} -eq 1 ]]; then
  echo "ERROR: --skip-zephyr-setup and --setup-only cannot be used together" >&2
  usage >&2
  exit 1
fi

if [[ ${SETUP_ONLY} -eq 0 && -z "${TARGETS_ARG}" ]]; then
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

run_west_sdk_install_with_proxy_fallback() {
  local sdk_version proxy_port proxy_url proxy_cache_dir proxy_pid

  # Try expected setup first.
  if run_command_block_from_readme "${ZEPHYR_README_PATH}" "<!-- RUN west_sdk_install -->"; then
    return 0
  fi

  echo "west sdk install failed; retrying through local release proxy"

  sdk_version="1.0.1"
  if [[ -f "zephyr/SDK_VERSION" ]]; then
    sdk_version="$(<zephyr/SDK_VERSION)"
  fi

  proxy_port="${ZEPHYR_SDK_RELEASE_PROXY_PORT:-8765}"
  proxy_url="http://127.0.0.1:${proxy_port}/releases"
  proxy_cache_dir="${ZEPHYR_SDK_RELEASE_PROXY_CACHE_DIR:-${HOME}/.cache/zephyr-sdk/v${sdk_version}}"

  python3 "${EXECUTORCH_PROJ_ROOT}/.ci/scripts/zephyr_sdk_release_proxy.py" \
    --version "${sdk_version}" \
    --cache-dir "${proxy_cache_dir}" \
    --port "${proxy_port}" &
  proxy_pid="$!"
  trap 'kill "${proxy_pid}" >/dev/null 2>&1 || true' RETURN

  for _ in {1..30}; do
    if wget -qO- "${proxy_url}?page=1" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done

  west sdk install \
    --version "${sdk_version}" \
    --api-url "${proxy_url}" \
    --gnu-toolchains arm-zephyr-eabi

  kill "${proxy_pid}" >/dev/null 2>&1 || true
  trap - RETURN
}

setup_zephyr_workspace() {
  # Check that zephyr/README.md and zephyr/executorch.yaml are in sync.
  verify_zephyr_readme

  # Based on instructions in zephyr/README.md and the selected sample README.
  run_command_block_from_readme "${ZEPHYR_README_PATH}" "<!-- RUN install_reqs -->"

  # Make sure to backup the Zephyr workspace folder if it exists to allow for local
  # testing that does not lose code/data.
  if [[ -d "${ZEPHYR_DEV_ROOT}" ]]; then
    mv "${ZEPHYR_DEV_ROOT}" "${ZEPHYR_DEV_ROOT}.backup.$(date +%Y%m%d%H%M%S)"
  fi
  mkdir -p "${ZEPHYR_DEV_ROOT}"

  cd "${ZEPHYR_DEV_ROOT}"
  export ZEPHYR_PROJ_ROOT="$(realpath "$(pwd)")"

  cd "${ZEPHYR_PROJ_ROOT}"

  run_command_block_from_readme "${ZEPHYR_README_PATH}" "<!-- RUN west_init -->"

  cp "${EXECUTORCH_PROJ_ROOT}/zephyr/executorch.yaml" zephyr/submanifests/

  run_command_block_from_readme "${ZEPHYR_README_PATH}" "<!-- RUN west_config -->"

  # Switch to executorch in this PR e.g. replace modules/lib/executorch with the
  # root folder of this repo instead of doing a re-checkout and figuring out the
  # correct commit hash.
  rm -Rf modules/lib/executorch
  ln -s "${EXECUTORCH_PROJ_ROOT}" modules/lib/executorch

  run_command_block_from_readme "${ZEPHYR_README_PATH}" "<!-- RUN west_packages_install -->"

  # Sometimes west sdk install can fail due to network issues or problems with
  # rate limiting on the Zephyr SDK download server. If it fails, retry through
  # a temporary local proxy that serves the release files from a local cache.
  run_west_sdk_install_with_proxy_fallback

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
}

use_existing_zephyr_workspace() {
  if [[ ! -d "${ZEPHYR_DEV_ROOT}" ]]; then
    echo "ERROR: --skip-zephyr-setup requires an existing ${ZEPHYR_DEV_ROOT} directory" >&2
    exit 1
  fi
  cd "${ZEPHYR_DEV_ROOT}"
  export ZEPHYR_PROJ_ROOT="$(realpath "$(pwd)")"
}

if [[ ${SKIP_ZEPHYR_SETUP} -eq 1 ]]; then
  use_existing_zephyr_workspace
else
  setup_zephyr_workspace
fi

if [[ ${SETUP_ONLY} -eq 1 ]]; then
  exit 0
fi

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
