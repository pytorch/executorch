#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

install_vulkan_sdk() {
  VULKAN_SDK_VERSION=$1
  _vulkan_sdk_url="https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/mac/vulkansdk-macos-${VULKAN_SDK_VERSION}.zip"
  _vulkan_sdk_dir=/tmp/vulkansdk
  _vulkan_sdk_extract_dir=/tmp/vulkansdk-installer

  mkdir -p "${_vulkan_sdk_dir}"
  rm -rf "${_vulkan_sdk_extract_dir}"
  mkdir -p "${_vulkan_sdk_extract_dir}"

  _tmp_archive="/tmp/vulkansdk-macos.zip"

  curl --silent --show-error --location --fail --retry 3 \
    --output "${_tmp_archive}" "${_vulkan_sdk_url}"

  unzip -q -o "${_tmp_archive}" -d "${_vulkan_sdk_extract_dir}"

  _vulkan_sdk_app_path="$(
    find "${_vulkan_sdk_extract_dir}" -maxdepth 3 -type d \
      \( -name "vulkansdk-macOS-${VULKAN_SDK_VERSION}.app" \
      -o -name "vulkansdk-macos-${VULKAN_SDK_VERSION}.app" \) \
      -print -quit
  )"
  if [[ -z "${_vulkan_sdk_app_path}" ]]; then
    echo "Failed to find Vulkan SDK installer app"
    exit 1
  fi
  _vulkan_sdk_installer="${_vulkan_sdk_app_path}/Contents/MacOS/$(basename "${_vulkan_sdk_app_path}" .app)"
  if [[ ! -x "${_vulkan_sdk_installer}" ]]; then
    echo "Failed to find Vulkan SDK installer binary at ${_vulkan_sdk_installer}"
    exit 1
  fi

  "${_vulkan_sdk_installer}" \
    --root "${_vulkan_sdk_dir}/${VULKAN_SDK_VERSION}" \
    --accept-licenses \
    --default-answer \
    --confirm-command install

  export PATH="${PATH}:${_vulkan_sdk_dir}/${VULKAN_SDK_VERSION}/macOS/bin/"
  export VULKAN_SDK="${_vulkan_sdk_dir}/${VULKAN_SDK_VERSION}/macOS"
  export DYLD_LIBRARY_PATH="${_vulkan_sdk_dir}/${VULKAN_SDK_VERSION}/macOS/lib${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"

  _moltenvk_icd="${VULKAN_SDK}/share/vulkan/icd.d/MoltenVK_icd.json"
  if [[ -f "${_moltenvk_icd}" ]]; then
    export VK_DRIVER_FILES="${_moltenvk_icd}"
  fi
}

VULKAN_SDK_VERSION="1.4.321.0"

install_vulkan_sdk "${VULKAN_SDK_VERSION}"
