#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) Samsung Electronics Co. LTD
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex


download_ai_lite_core() {
  AI_LITE_CORE_VERSION=$1
  _exynos_ai_lite_core_url="https://ko.ai-studio-farm.com:26310/api/v1/buckets/lite-core/objects/download?prefix=05/ubuntu2204/exynos-ai-litecore-v0.5.0,tar.gz"

  _exynos_lite_core_dir=/tmp/exynos_ai_lite_core
  mkdir -p ${_exynos_lite_core_dir}

  _tmp_archive="/tmp/exynos-ai-litecore-v0.5.0.tar.gz"

  sudo update-ca-certificates

  curl --silent --show-error --location --fail --retry 3 \
    --output "${_tmp_archive}" "${_exynos_ai_lite_core_url}"

  tar -C "${_exynos_lite_core_dir}" --strip-components=1 -xzvf "${_tmp_archive}"

  export EXYNOS_AI_LITECORE_ROOT=${_exynos_lite_core_dir}
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${EXYNOS_AI_LITECORE_ROOT}/lib/x86_64-linux
}

install_enn_backend() {
  # Please export ANDROID_NDK_ROOT if enable on-device test
  export EXECUTORCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

  # Remove --build and add --ndk parameter if enable on-device test
  bash backends/samsung/build.sh --build x86_64
  export PYTHONPATH=${PYTHONPATH}:${EXECUTORCH_ROOT}/..
}

AI_LITE_CORE_VERSION=0.5.0

download_ai_lite_core ${AI_LITE_CORE_VERSION}
install_enn_backend
