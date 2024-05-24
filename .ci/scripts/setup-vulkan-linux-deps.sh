
#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

install_swiftshader() {
  _https_amazon_aws=https://ossci-android.s3.amazonaws.com
  _swiftshader_archive=swiftshader-abe07b943-prebuilt.tar.gz
  _swiftshader_dir=/var/lib/swiftshader
  mkdir -p $_swiftshader_dir

  _tmp_archive="/tmp/${_swiftshader_archive}"

  curl --silent --show-error --location --fail --retry 3 \
    --output "${_tmp_archive}" "$_https_amazon_aws/${_swiftshader_archive}"

  tar -C "${_swiftshader_dir}" -xzf "${_tmp_archive}"

  export VK_ICD_FILENAMES="${_swiftshader_dir}/build/Linux/vk_swiftshader_icd.json"
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${_swiftshader_dir}/build/Linux/libvulkan.so.1"
}

install_glslc() {
  _shaderc_url_base="https://storage.googleapis.com/shaderc/artifacts/prod/graphics_shader_compiler/shaderc/linux/continuous_clang_release"
  _shaderc_version="448/20240305-065535"
  _shaderc_url="${_shaderc_url_base}/${_shaderc_version}/install.tgz"

  _glslc_dir=/var/lib/shaderc
  mkdir -p $_glslc_dir

  _tmp_archive="/tmp/install.tgz"

  curl --silent --show-error --location --fail --retry 3 \
    --output "${_tmp_archive}" "${_shaderc_url}"

  tar -C "${_glslc_dir}" -xzf "${_tmp_archive}"

  export PATH="${PATH}:${_glslc_dir}/install/bin/"
}

install_swiftshader
install_glslc
