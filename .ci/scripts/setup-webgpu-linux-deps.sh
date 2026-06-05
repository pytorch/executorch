#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Vendor Dawn (Tint) + SwiftShader for the WebGPU backend CI WITHOUT hosting a
# private prebuilt:
#   * Dawn  : Google's official nightly prebuilt, downloaded directly from
#             github.com/google/dawn/releases (pinned tag+rev+sha256) -- the same
#             "fetch upstream's published binary" pattern as setup-wgpu-native.sh.
#   * SwiftShader : reuse the prebuilt ALREADY on the ossci-android bucket (the one
#             setup-vulkan-linux-deps.sh uses). No new S3 uploads anywhere.
# Dawn (Chrome's WebGPU impl; its WGSL compiler Tint is the spec reference) on
# SwiftShader gives a headless, deterministic, spec-faithful CLI backend.
#
# Exports Dawn_DIR / VK_ICD_FILENAMES / LD_LIBRARY_PATH for the cmake build+run.
# Local/rig override: set DAWN_PREBUILT_DIR=<dir containing lib64/cmake/Dawn> to
# skip the Dawn download.
set -ex

# --- pinned versions (bump rev+sha together when upgrading Dawn) --------------
DAWN_TAG="${DAWN_TAG:-v20260423.175430}"
DAWN_REV="${DAWN_REV:-31e25af254ab572c77054edec4946d2244e184dd}"
DAWN_SHA256="${DAWN_SHA256:-ac76fac090162dc1ecea5ed0f28a557bb8f49efc47faab01886105ace82b7b64}"
SWIFTSHADER_ARCHIVE="${SWIFTSHADER_ARCHIVE:-swiftshader-abe07b943-prebuilt.tar.gz}"

_dawn_dir="${DAWN_PREBUILT_DIR:-/tmp/dawn-ci}"
_ss_dir=/tmp/swiftshader

# --- toolchain prereqs --------------------------------------------------------
# Dawn dlopens the system Vulkan loader at runtime (libvulkan1). And the
# ubuntu-latest prebuilt is built with a bleeding-edge GCC: it references
# libstdc++ symbols newer than ubuntu-22.04's default (e.g. _M_replace_cold,
# GCC 13+), so the static .a won't link against the stock runtime. Pull a current
# libstdc++ from the ubuntu-toolchain-r PPA when the symbol floor isn't met. All
# of this is scoped to the WebGPU CI job; newer libstdc++ is backward-compatible.
if command -v apt-get >/dev/null 2>&1; then
  _SUDO=""; command -v sudo >/dev/null 2>&1 && _SUDO="sudo"
  ${_SUDO} apt-get update -y || true
  ${_SUDO} apt-get install -y libvulkan1 software-properties-common || true
  if ! strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 2>/dev/null \
      | grep -q "GLIBCXX_3.4.32"; then
    ${_SUDO} add-apt-repository -y ppa:ubuntu-toolchain-r/test || true
    ${_SUDO} apt-get update -y || true
    ${_SUDO} apt-get install -y libstdc++6 || true  # newest GCC runtime
  fi
fi

# --- Dawn: official prebuilt from GitHub (no S3) ------------------------------
mkdir -p "${_dawn_dir}"
if [[ ! -d "${_dawn_dir}/lib64/cmake/Dawn" ]]; then
  _dawn_tar="/tmp/Dawn-${DAWN_REV}-ubuntu-latest-Release.tar.gz"
  curl --silent --show-error --location --fail --retry 3 --retry-all-errors \
    --output "${_dawn_tar}" \
    "https://github.com/google/dawn/releases/download/${DAWN_TAG}/Dawn-${DAWN_REV}-ubuntu-latest-Release.tar.gz"
  echo "${DAWN_SHA256}  ${_dawn_tar}" | sha256sum -c -
  # archive top dir is Dawn-<rev>-ubuntu-latest-Release/{lib64,include,bin}
  tar -C "${_dawn_dir}" --strip-components=1 -xzf "${_dawn_tar}"
fi

# --- SwiftShader: reuse the existing ossci prebuilt (no new upload) -----------
if [[ ! -f "${_ss_dir}/swiftshader/build/Linux/vk_swiftshader_icd.json" ]]; then
  _ss_aws=https://ossci-android.s3.amazonaws.com
  mkdir -p "${_ss_dir}"
  curl --silent --show-error --location --fail --retry 3 --retry-all-errors \
    --output "/tmp/${SWIFTSHADER_ARCHIVE}" "${_ss_aws}/${SWIFTSHADER_ARCHIVE}"
  tar -C "${_ss_dir}" -xzf "/tmp/${SWIFTSHADER_ARCHIVE}"
fi

export Dawn_DIR="${_dawn_dir}/lib64/cmake/Dawn"
export VK_ICD_FILENAMES="${_ss_dir}/swiftshader/build/Linux/vk_swiftshader_icd.json"
export LD_LIBRARY_PATH="${_ss_dir}/swiftshader/build/Linux/:${LD_LIBRARY_PATH:-}"
export WEBGPU_USING_SWIFTSHADER=1
