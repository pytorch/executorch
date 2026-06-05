#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# SwiftShader: software Vulkan adapter for GPU-less CI (LunarG SDK not needed).
install_swiftshader() {
  _https_amazon_aws=https://ossci-android.s3.amazonaws.com
  _swiftshader_archive=swiftshader-abe07b943-prebuilt.tar.gz
  _swiftshader_dir=/tmp/swiftshader
  mkdir -p $_swiftshader_dir

  _tmp_archive="/tmp/${_swiftshader_archive}"

  curl --silent --show-error --location --fail --retry 3 --retry-all-errors \
    --output "${_tmp_archive}" "$_https_amazon_aws/${_swiftshader_archive}"

  tar -C "${_swiftshader_dir}" -xzf "${_tmp_archive}"

  export VK_ICD_FILENAMES="${_swiftshader_dir}/swiftshader/build/Linux/vk_swiftshader_icd.json"
  export LD_LIBRARY_PATH="${_swiftshader_dir}/swiftshader/build/Linux/:${LD_LIBRARY_PATH}"
  export ETVK_USING_SWIFTSHADER=1
}

install_swiftshader
bash backends/webgpu/scripts/setup-wgpu-native.sh
