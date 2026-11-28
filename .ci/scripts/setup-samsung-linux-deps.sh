#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) Samsung Electronics Co. LTD
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

API_KEY=$SAMSUNG_AI_LITECORE_KEY
if [[ -z "${API_KEY}" ]]; then
  echo "ERROR: It didn't set up SAMSUNG_AI_LITECORE_KEY." >&2
  exit 1
fi

export DEVICE_CONNECT_ENABLED=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-device-connect)
      export DEVICE_CONNECT_ENABLED=0
      shift
      ;;
    *)
      # Unknown option
      shift
      ;;
  esac
done

# Waiting for a device and each devicefarm-cli call are bounded so a busy or
# unresponsive farm cannot hold the job until its own timeout.
DEVICEFARM_WAIT_TIMEOUT_SEC="${DEVICEFARM_WAIT_TIMEOUT_SEC:-1800}"
DEVICEFARM_CALL_TIMEOUT_SEC="${DEVICEFARM_CALL_TIMEOUT_SEC:-120}"
DEVICEFARM_POLL_INTERVAL_SEC="${DEVICEFARM_POLL_INTERVAL_SEC:-60}"

LITECORE_VERSION="v1.2.1"
LITECORE_FILE_NAME="ai-litecore-ubuntu2204-${LITECORE_VERSION}.tar.gz"
DEVICEFARM_CLI_VERSION="beta-v1.1.0"
DEVICEFARM_FILE_NAME="devicefarmcli-${DEVICEFARM_CLI_VERSION}.zip"

LITECORE_URL="https://soc-developer.semiconductor.samsung.com/api/v1/resource/download-file/${LITECORE_FILE_NAME}"
DEVICEFARM_URL="https://soc-developer.semiconductor.samsung.com/api/v1/resource/download-file/${DEVICEFARM_FILE_NAME}"

download_and_extract() {
  local download_url="$1"
  local out_dir="$2"
  local out_file="$3"

  echo "Downloading from ${download_url}..."
  curl -fsSL --retry 3 --retry-all-errors \
    -H "apikey: ${API_KEY}" \
    -o "${out_file}" \
    "${download_url}"

  echo "Download completed: ${out_file}"

  mkdir -p "${out_dir}"
  case "${out_file##*.}" in
  tar|tgz|gz)
    echo "Extracting TAR.GZ..."
    tar -C "${out_dir}" --strip-components=1 -xzvf "${out_file}"
    ;;

  zip)
    echo "Extracting ZIP..."
    unzip -qo -d "${out_dir}" "${out_file}"
    ;;

  *)
    exit 1
    ;;
  esac
  echo "Extracted to: ${out_dir}"
}

download_ai_lite_core() {
  local litecore_version="${1:-${LITECORE_VERSION}}"
  local litecore_out="/tmp/${LITECORE_FILE_NAME}"
  local litecore_dir="/tmp/exynos_ai_lite_core"

  download_and_extract \
    "${LITECORE_URL}" \
    "${litecore_dir}" \
    "${litecore_out}"

  export EXYNOS_AI_LITECORE_ROOT="${litecore_dir}"
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${EXYNOS_AI_LITECORE_ROOT}/lib/x86_64-linux"
}

install_devicefarm_cli() {
  local cli_version="${1:-${DEVICEFARM_CLI_VERSION}}"
  local cli_out="/tmp/${DEVICEFARM_FILE_NAME}"
  local cli_dir="/tmp/devicefarm_cli"

  download_and_extract \
    "${DEVICEFARM_URL}" \
    "${cli_dir}" \
    "${cli_out}"

  export PATH="${PATH%:}:${cli_dir}"
  chmod +x "${cli_dir}/devicefarm-cli"
}

# Run devicefarm-cli with a deadline; KILL follows TERM for a CLI that ignores it.
devicefarm_cli() {
  timeout --kill-after=10 "${DEVICEFARM_CALL_TIMEOUT_SEC}" devicefarm-cli "$@"
}

# Drop the queued request (or the assigned device) so it does not outlive the job.
release_device_queue() {
  echo "[INFO] Clearing the device request (-d)..."
  devicefarm_cli -d || echo "::warning::Failed to clear the device request (-d)." >&2
  export DEVICE_QUEUED=0
}

acquire_device() {
  export DEVICE_ACQUIRED=0
  export DEVICE_QUEUED=0
  if ! command -v devicefarm-cli >/dev/null 2>&1; then
    echo "[WARN] devicefarm-cli is not installed." >&2
    return 1
  fi

  echo "[INFO] Enqueue request (-Q)..."
  # Enqueue device request
  if ! devicefarm_cli -Q; then
    echo "::warning::Failed to enqueue device request (-Q)." >&2
    echo "[WARN] Device queue registration failed - continuing without device." >&2
    # A -Q that timed out may still have registered the request.
    release_device_queue
    return 0
  fi
  export DEVICE_QUEUED=1

  local interval_sec="${DEVICEFARM_POLL_INTERVAL_SEC}"
  local deadline=$(( $(date +%s) + DEVICEFARM_WAIT_TIMEOUT_SEC ))
  local out rc remaining status

  echo "[INFO] Polling assignment status (-C) every ${interval_sec}s for up to ${DEVICEFARM_WAIT_TIMEOUT_SEC}s..."

  while true; do
    rc=0
    out="$(devicefarm_cli -C 2>&1)" || rc=$?

    # Determine status: assigned / waiting / unavailable. A -C call that hit
    # its deadline counts as waiting until the overall deadline.
    if [[ "${rc}" -eq 124 || "${rc}" -eq 137 ]]; then
      echo "[WARN] -C did not answer within ${DEVICEFARM_CALL_TIMEOUT_SEC}s." >&2
      status="waiting"
    elif printf '%s' "$out" | grep -qiE 'waiting|not[[:space:]-]*assigned'; then
      status="waiting"
    elif printf '%s' "$out" | grep -qi 'assigned'; then
      status="assigned"
    else
      status="unknown"
    fi

    case "$status" in
      assigned)
        echo "[INFO] Device assigned."
        echo "$out"
        # Execute test command
        devicefarm_cli -E "ls /" || true
        export DEVICE_ACQUIRED=1
        echo "[INFO] Device successfully assigned and connected."
        return 0
        ;;
      waiting)
        remaining=$(( deadline - $(date +%s) ))
        if (( remaining <= 0 )); then
          echo "::warning::No Samsung device was assigned within ${DEVICEFARM_WAIT_TIMEOUT_SEC}s - continuing without device." >&2
          release_device_queue
          return 0
        fi
        echo "[INFO] Status: $status"
        sleep "$(( remaining < interval_sec ? remaining : interval_sec ))"
        ;;
      *)
        echo "[WARN] Unknown status from -C. Output:"
        echo "$out"
        release_device_queue
        return 0
        ;;
    esac
  done
}

install_enn_backend() {
  local ndk_dir="/opt/ndk"
  local ndk_version="r28c"

  if [[ ! -d "${ndk_dir}" ]]; then
    sudo mkdir -p "${ndk_dir}"
    sudo chown "$(whoami)":"$(whoami)" "${ndk_dir}"
  fi

  export ANDROID_NDK_ROOT="${ndk_dir}"
  echo "NDK will be installed/used at: ${ANDROID_NDK_ROOT}"

  bash backends/samsung/build.sh --build all
  bash examples/samsung/build.sh

  export EXECUTORCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  export PYTHONPATH="${PYTHONPATH:-}:${EXECUTORCH_ROOT}/.."
}

download_ai_lite_core ${LITECORE_VERSION}
install_enn_backend

if [[ "${DEVICE_CONNECT_ENABLED}" == "1" ]]; then
  install_devicefarm_cli "${DEVICEFARM_CLI_VERSION}"
  acquire_device
else
  export DEVICE_ACQUIRED=0
  export DEVICE_QUEUED=0
fi
