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

OS_NAME="Ubuntu 22.04"
LITECORE_BASE="https://soc-developer.semiconductor.samsung.com/api/v1/resource/ai-litecore/download"
DEVICEFARM_BASE="https://soc-developer.semiconductor.samsung.com/api/v1/resource/remotelab/download"

parse_url() {
  local json="$1"
  if command -v jq >/dev/null 2>&1; then
    jq -r '.data // empty' <<<"$json"
  else
    sed -n 's/.*"data":[[:space:]]*"\([^"]*\)".*/\1/p' <<<"$json"
  fi
}

download_and_extract() {
  local base_url="$1"
  local version="$2"
  local out_dir="$3"
  local out_file="$4"

  local resp
  resp=$(curl -fsSL -G \
    -H "apikey: ${API_KEY}" \
    --data-urlencode "version=${version}" \
    --data-urlencode "os=${OS_NAME}" \
    "${base_url}")

  local download_url
  download_url=$(parse_url "$resp")
  if [[ -z "${download_url}" ]]; then
    echo "ERROR: It failed to download from ${base_url} ."
    echo "Response: $resp" >&2
    exit 1
  fi

  curl -fsSL -L --retry 3 -o "${out_file}" "${download_url}"
  echo "Download completed: ${out_file}"

  mkdir -p "${out_dir}"
  case "${out_file##*.}" in
  tar|tgz|gz)
    echo "Extracting TAR.GZ..."
    tar -C "${out_dir}" --strip-components=1 -xzvf "${out_file}"
    ;;

  zip)
    echo "Extracting ZIP..."
    unzip -q -d "${out_dir}" "${out_file}"
    ;;

  *)
    exit 1
    ;;
  esac
  echo "Extracted to: ${out_dir}"
}

download_ai_lite_core() {
  local litecore_version="${1:-1.0}"
  local litecore_out="/tmp/exynos-ai-litecore-v${litecore_version}.tar.gz"
  local litecore_dir="/tmp/exynos_ai_lite_core"

  download_and_extract \
    "${LITECORE_BASE}" \
    "${litecore_version}" \
    "${litecore_dir}" \
    "${litecore_out}"

  export EXYNOS_AI_LITECORE_ROOT="${litecore_dir}"
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${EXYNOS_AI_LITECORE_ROOT}/lib/x86_64-linux"
}

install_devicefarm_cli() {
  local cli_version="${1:-beta-1.0.7}"
  local cli_out="/tmp/devicefarm-cli-v${cli_version}.zip"
  local cli_dir="/tmp/devicefarm_cli"

  download_and_extract \
    "${DEVICEFARM_BASE}" \
    "${cli_version}" \
    "${cli_dir}" \
    "${cli_out}"

  export PATH="${PATH%:}:${cli_dir}"
  chmod +x "${cli_dir}/devicefarm-cli"
}

reserve_if_needed() {
  if ! command -v devicefarm-cli >/dev/null 2>&1; then
    echo "[WARN] devicefarm-cli is not installed." >&2
    return 1
  fi

  local raw_info info_lines
  raw_info="$(devicefarm-cli -I)"

  info_lines="$(printf '%s\n' "$raw_info" | grep -v '^\\[INFO\\]')"

  local found_count

  found_count=$(printf '%s\n' "$info_lines" \
    | grep -Eo 'Found available reservations *: *[0-9]+' \
    | grep -Eo '[0-9]+')
  [[ -z "$found_count" ]] && found_count=0

  echo "[INFO] Current Reserved Count: $found_count"

  local THRESHOLD_SECONDS=12600
  local any_below_threshold=0

  if (( found_count > 0 )); then
    local table_body
    table_body=$(printf '%s\n' "$info_lines" | sed -n '2,$p')

    while IFS= read -r line; do
      if [[ "$line" =~ ^[0-9]+[[:space:]]+([0-9]{1,2}:[0-9]{2}:[0-9]{2}) ]]; then
        local time_str="${BASH_REMATCH[1]}"
        IFS=: read -r hh mm ss <<<"$time_str"
        (( seconds = 10#$hh * 3600 + 10#$mm * 60 + 10#$ss ))
        if (( seconds <= THRESHOLD_SECONDS )); then
          any_below_threshold=1
          break
        fi
      fi
    done <<<"$table_body"
  else
    any_below_threshold=1
  fi

  if (( any_below_threshold )); then
    echo "[INFO] Reserving now."
    devicefarm-cli -R
  else
    echo "[INFO] Don't need to be reserved."
  fi

  local info_after reservation_id max_seconds=0 max_id

  info_after="$(devicefarm-cli -I)"

  local body_after
  body_after=$(printf '%s\n' "$info_after" | grep -v '^\\[INFO\\]' | sed -n '2,$p')

  while IFS= read -r line; do
    if [[ "$line" =~ ^[0-9]+[[:space:]]+([0-9]{1,2}:[0-9]{2}:[0-9]{2})[[:space:]].*([0-9a-f-]{36})$ ]]; then
      local time_str="${BASH_REMATCH[1]}"
      local id="${BASH_REMATCH[2]}"
      IFS=: read -r hh mm ss <<<"$time_str"
      (( seconds = 10#$hh * 3600 + 10#$mm * 60 + 10#$ss ))
      if (( seconds > max_seconds )); then
        max_seconds=$seconds
        max_id=$id
      fi
    fi
  done <<<"$body_after"

  reservation_id=$max_id

  if [[ -n "$reservation_id" ]]; then
    devicefarm-cli -C "$reservation_id"
    devicefarm-cli -E "ls /"
  else
    echo "[WARN] There is no available devices."
  fi
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

  export EXECUTORCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  export PYTHONPATH="${PYTHONPATH:-}:${EXECUTORCH_ROOT}/.."
}

litecore_ver="1.0"
devicefarm_ver="beta-1.0.7"

download_ai_lite_core ${litecore_ver}
install_devicefarm_cli "${devicefarm_ver}"
install_enn_backend
reserve_if_needed
