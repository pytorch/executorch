#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) Samsung Electronics Co. LTD
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex


download_ai_lite_core() {
  API_BASE="https://soc-developer.semiconductor.samsung.com/api/v1/resource/ai-litecore/download"
  API_KEY=${SAMSUNG_AI_LITECORE_KEY}

  VERSION="0.5"
  OS_NAME="Ubuntu 22.04"
  OUT_FILE="/tmp/exynos-ai-litecore-v${VERSION}.tar.gz"
  TARGET_PATH="/tmp/exynos_ai_lite_core"

  mkdir -p ${TARGET_PATH}
  # Presigned issue URL
  JSON_RESP=$(curl -sS -G \
    --location --fail --retry 3 \
    -H "apikey: ${API_KEY}" \
    --data-urlencode "version=${VERSION}" \
    --data-urlencode "os=${OS_NAME}" \
    "${API_BASE}")

  DOWNLOAD_URL=$(echo "$JSON_RESP" | sed -n 's/.*"data":[[:space:]]*"\([^"]*\)".*/\1/p')

  if [[ -z "$DOWNLOAD_URL" ]]; then
    echo "Failed to extract download URL"
    echo "$JSON_RESP"
    exit 1
  fi

  # Download LiteCore
  curl -sS -L --fail --retry 3 \
    --output "$OUT_FILE" \
    "$DOWNLOAD_URL"

  echo "Download done: $OUT_FILE"


  tar -C "${TARGET_PATH}" --strip-components=1 -xzvf "${OUT_FILE}"

  export EXYNOS_AI_LITECORE_ROOT=${TARGET_PATH}
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:${EXYNOS_AI_LITECORE_ROOT}/lib/x86_64-linux
}

install_enn_backend() {
  NDK_INSTALLATION_DIR=/opt/ndk
  rm -rf "${NDK_INSTALLATION_DIR}" && sudo mkdir -p "${NDK_INSTALLATION_DIR}"
  ANDROID_NDK_VERSION=r27b

  # build Exynos backend
  export ANDROID_NDK_ROOT=${ANDROID_NDK_ROOT:-/opt/ndk}
  bash backends/samsung/build.sh --build all
  # set env variable
  export EXECUTORCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
  export PYTHONPATH=${PYTHONPATH:-}:${EXECUTORCH_ROOT}/..
}

AI_LITE_CORE_VERSION=0.5.0

download_ai_lite_core ${AI_LITE_CORE_VERSION}
install_enn_backend
