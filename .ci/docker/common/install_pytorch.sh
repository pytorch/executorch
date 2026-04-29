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

install_pytorch_and_domains() {
  if [ "${TORCH_CHANNEL}" != "nightly" ]; then
    # Test/release: install the published wheels directly. The specs and URL
    # are passed in as docker build args (computed from torch_pin.py by
    # .ci/docker/build.sh). RC wheels at /whl/test/ get re-uploaded under the
    # same version, so use --no-cache-dir there to avoid stale cache hits.
    local cache_flag=""
    if [ "${TORCH_CHANNEL}" = "test" ]; then
      cache_flag="--no-cache-dir"
    fi
    pip_install --force-reinstall ${cache_flag} \
      "${TORCH_SPEC}" "${TORCHVISION_SPEC}" "${TORCHAUDIO_SPEC}" \
      --index-url "${TORCH_INDEX_URL}/cpu"
    return
  fi

  # Nightly: build pytorch from source against the pinned SHA in pytorch.txt
  # so we catch upstream regressions, then install audio/vision from the
  # commits that pytorch itself pins.
  git clone https://github.com/pytorch/pytorch.git

  # Fetch the target commit
  pushd pytorch || true
  git checkout "${TORCH_VERSION}"
  git submodule update --init --recursive

  chown -R ci-user .

  export _GLIBCXX_USE_CXX11_ABI=1
  # PyTorch's FindARM.cmake hard-fails when the SVE+BF16 compile probe
  # doesn't pass — gcc-11 in this image is too old to accept the combined
  # NEON/SVE/bfloat16 intrinsics the probe exercises. Executorch's aarch64
  # runtime targets (phones, embedded) don't use SVE, so bypass the check.
  export BUILD_IGNORE_SVE_UNAVAILABLE=1
  # Then build and install PyTorch
  conda_run python setup.py bdist_wheel
  pip_install "$(echo dist/*.whl)"

  # Defer to PyTorch's own pinned audio/vision commits.
  TORCHAUDIO_VERSION=$(cat .github/ci_commit_pins/audio.txt)
  export TORCHAUDIO_VERSION
  TORCHVISION_VERSION=$(cat .github/ci_commit_pins/vision.txt)
  export TORCHVISION_VERSION

  install_domains

  popd || true
  # Clean up the cloned PyTorch repo to reduce the Docker image size
  rm -rf pytorch

  # Print sccache stats for debugging
  as_ci_user sccache --show-stats
}

install_pytorch_and_domains
