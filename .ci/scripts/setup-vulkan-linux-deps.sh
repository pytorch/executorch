
#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

install_swiftshader() {
  _https_amazon_aws=https://ossci-android.s3.amazonaws.com
  _swiftshader_archive=swiftshader-abe07b943-prebuilt.tar.gz
  _swiftshader_dir=/tmp/swiftshader
  mkdir -p $_swiftshader_dir

  _tmp_archive="/tmp/${_swiftshader_archive}"

  curl --silent --show-error --location --fail --retry 3 \
    --output "${_tmp_archive}" "$_https_amazon_aws/${_swiftshader_archive}"

  tar -C "${_swiftshader_dir}" -xzf "${_tmp_archive}"

  export VK_ICD_FILENAMES="${_swiftshader_dir}/swiftshader/build/Linux/vk_swiftshader_icd.json"
  export LD_LIBRARY_PATH="${_swiftshader_dir}/swiftshader/build/Linux/"
}

install_vulkan_sdk() {
  VULKAN_SDK_VERSION=$1
  _vulkan_sdk_url="https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz"

  _vulkan_sdk_dir=/tmp/vulkansdk
  mkdir -p $_vulkan_sdk_dir

  _tmp_archive="/tmp/vulkansdk.tar.gz"

  curl --silent --show-error --location --fail --retry 3 \
    --output "${_tmp_archive}" "${_vulkan_sdk_url}"

  tar -C "${_vulkan_sdk_dir}" -xzf "${_tmp_archive}"

  export PATH="${PATH}:${_vulkan_sdk_dir}/${VULKAN_SDK_VERSION}/x86_64/bin/"
}

VULKAN_SDK_VERSION="1.2.198.1"

install_swiftshader
install_vulkan_sdk "${VULKAN_SDK_VERSION}"
