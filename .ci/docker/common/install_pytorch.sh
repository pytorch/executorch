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

configure_pytorch_compiler() {
  local cxx_version
  cxx_version=$(/usr/bin/c++ --version | head -n1)
  if [[ "${cxx_version}" == *clang* ]]; then
    local clang_major
    clang_major=$(/usr/bin/c++ -dumpversion | cut -d. -f1)
    if [[ "${clang_major}" =~ ^[0-9]+$ && "${clang_major}" -lt 16 ]] &&
      command -v gcc >/dev/null &&
      command -v g++ >/dev/null; then
      export CC=gcc
      export CXX=g++
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
  configure_pytorch_compiler
  # PyTorch no longer supports "python setup.py bdist_wheel"; it now builds
  # through scikit-build-core (PEP 517). Build the wheel with the standard
  # frontend and keep build isolation off, so PyTorch builds against this
  # environment's numpy and toolchain (avoids an ABI mismatch) and reuses
  # sccache.
  #
  # With isolation off the frontend does not fetch the PEP 517 build
  # requirements, so install them here. They go into a throwaway venv rather
  # than into the image, because scikit-build-core registers a setuptools
  # build_ext plugin: left in the image it hijacks every later
  # "pip install --no-build-isolation" and turns on C++20 module scanning that
  # the image compiler cannot satisfy. The venv inherits the image's
  # site-packages, so PyTorch still builds against the same numpy.
  #
  # Keep the list in sync with pytorch/pyproject.toml [build-system].requires.
  local build_venv=/tmp/pytorch-build-venv
  rm -rf "${build_venv}"
  conda_run python -m venv --system-site-packages "${build_venv}"
  conda_run "${build_venv}/bin/pip" install build "scikit-build-core>=1.0" \
    "setuptools>=77.0.0,<82" "cmake>=3.27,<4" ninja "packaging>=24.2" \
    "typing-extensions>=4.10.0" pyyaml six
  conda_run "${build_venv}/bin/python" -m build --wheel --no-isolation
  rm -rf "${build_venv}"
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
