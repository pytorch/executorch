
#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

install_vulkan_sdk() {
  VULKAN_SDK_VERSION=$1
  _vulkan_sdk_url="https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/mac/vulkansdk-macos-${VULKAN_SDK_VERSION}.zip"

  _vulkan_sdk_dir=/tmp/vulkansdk
  mkdir -p $_vulkan_sdk_dir

  _tmp_archive="/tmp/vulkansdk.zip"

  curl --silent --show-error --location --fail --retry 3 \
    --output "${_tmp_archive}" "${_vulkan_sdk_url}"

  unzip -q -o "${_tmp_archive}" -d "${_vulkan_sdk_dir}"

  export VULKAN_SDK="${_vulkan_sdk_dir}/VulkanSDK/${VULKAN_SDK_VERSION}/macOS"
  export PATH="${VULKAN_SDK}/bin:${PATH}"
  export VK_ICD_FILENAMES="${VULKAN_SDK}/share/vulkan/icd.d/MoltenVK_icd.json"
  export VK_LAYER_PATH="${VULKAN_SDK}/share/vulkan/explicit_layer.d"
  export DYLD_LIBRARY_PATH="${VULKAN_SDK}/lib:${DYLD_LIBRARY_PATH:-}"
}

VULKAN_SDK_VERSION="1.4.321.1"

install_vulkan_sdk "${VULKAN_SDK_VERSION}"
