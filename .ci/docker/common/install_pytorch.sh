#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

install_domains() {
  echo "Install torchvision and torchaudio"
  pip_install --no-build-isolation --user "git+https://github.com/pytorch/audio.git@${TORCHAUDIO_VERSION}"
  pip_install --no-build-isolation --user "git+https://github.com/pytorch/vision.git@${TORCHVISION_VERSION}"
}

select_pytorch_source_build_compiler() {
  local cxx_version
  local clang_major

  if ! command -v c++ >/dev/null 2>&1; then
    return
  fi

  cxx_version="$(c++ --version 2>/dev/null | head -n 1 || true)"
  if [[ "${cxx_version}" != *clang* && "${cxx_version}" != *Clang* ]]; then
    return
  fi

  clang_major="$(sed -nE 's/.*clang version ([0-9]+).*/\1/p' <<<"${cxx_version}" | head -n 1)"
  if [[ -z "${clang_major}" ]]; then
    clang_major="$(sed -nE 's/.*Apple clang version ([0-9]+).*/\1/p' <<<"${cxx_version}" | head -n 1)"
  fi

  if [[ -n "${clang_major}" && "${clang_major}" -lt 16 ]]; then
    if command -v gcc >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1; then
      echo "Using gcc/g++ for PyTorch source build because ${cxx_version} is too old"
      export CC=gcc
      export CXX=g++
      export CMAKE_C_COMPILER=gcc
      export CMAKE_CXX_COMPILER=g++
    else
      echo "gcc/g++ not found; continuing PyTorch source build with ${cxx_version}"
    fi
  fi
}

install_pytorch_and_domains() {
  git clone https://github.com/pytorch/pytorch.git

  # Fetch the target commit
  pushd pytorch || true
  git checkout "${TORCH_VERSION}"
  git submodule update --init --recursive

  chown -R ci-user .

  export _GLIBCXX_USE_CXX11_ABI=1
  if [[ "$(uname -m)" == "aarch64" ]]; then
    export BUILD_IGNORE_SVE_UNAVAILABLE=1
  fi
  if [[ -n "${PYTORCH_BUILD_MAX_JOBS:-}" ]]; then
    export MAX_JOBS="${PYTORCH_BUILD_MAX_JOBS}"
  fi
  select_pytorch_source_build_compiler
  # Then build and install PyTorch
  conda_run python setup.py bdist_wheel
  pip_install "$(echo dist/*.whl)"

  # Grab the pinned audio and vision commits from PyTorch
  TORCHAUDIO_VERSION=release/2.11
  export TORCHAUDIO_VERSION
  TORCHVISION_VERSION=release/0.28
  export TORCHVISION_VERSION

  install_domains

  popd || true
  # Clean up the cloned PyTorch repo to reduce the Docker image size
  rm -rf pytorch

  # Print sccache stats for debugging
  as_ci_user sccache --show-stats
}

install_pytorch_and_domains
