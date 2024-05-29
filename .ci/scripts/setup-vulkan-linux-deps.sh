
#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

install_swiftshader() {
  SWIFTSHADER_COMMIT=$1
  pushd /tmp || return

  git clone https://github.com/google/swiftshader.git

  cd swiftshader
  git checkout $SWIFTSHADER_COMMIT

  cd build
  cmake ..
  cmake --build . --parallel

  export LD_LIBRARY_PATH="$(pwd)/Linux/libvulkan.so.1"

  popd || return
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

SWIFTSHADER_COMMIT=ec5dbd2dfb4623f5b2721a77bb5388d79fafc506
VULKAN_SDK_VERSION="1.2.198.1"

install_swiftshader "${SWIFTSHADER_COMMIT}"
install_vulkan_sdk "${VULKAN_SDK_VERSION}"
