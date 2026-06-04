#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

install_pytorch_and_domains() {
  local cache_args=()
  if [ "${TORCH_CHANNEL}" = "test" ]; then
    cache_args=("--no-cache-dir")
  fi

  local wheelhouse
  wheelhouse=$(mktemp -d)
  chown ci-user:ci-user "${wheelhouse}"

  local system_name
  local system_arch
  local python_version
  system_name=$(uname)
  system_arch=$(uname -m)
  python_version=$(conda_run python -c 'import platform; v=platform.python_version_tuple(); print(f"{v[0]}{v[1]}")')
  local torch_wheel_cache_path="cached_artifacts/pytorch/executorch/pytorch_wheels/${system_name}/${system_arch}/${python_version}/cpu/${TORCH_CHANNEL}"
  local torch_wheel_cache_uri="s3://gha-artifacts/${torch_wheel_cache_path}"

  # Do not cache test-channel wheels in S3: RC artifacts may be re-uploaded
  # under the same package version.
  if [[ "${TORCH_CHANNEL}" != "test" ]] && command -v aws >/dev/null 2>&1; then
    aws s3 sync "${torch_wheel_cache_uri}" "${wheelhouse}" || true
  fi

  conda_run python -m pip download --progress-bar off --no-deps "${cache_args[@]}" \
    --dest "${wheelhouse}" \
    --find-links "${wheelhouse}" \
    "${TORCH_SPEC}" "${TORCHVISION_SPEC}" "${TORCHAUDIO_SPEC}" \
    --index-url "${TORCH_INDEX_URL}/cpu"

  if [[ "${TORCH_CHANNEL}" != "test" && -z "${GITHUB_RUNNER:-}" ]] && command -v aws >/dev/null 2>&1; then
    aws s3 sync "${wheelhouse}" "${torch_wheel_cache_uri}" \
      --exclude "*" --include "*.whl" || true
  fi

  pip_install --force-reinstall "${cache_args[@]}" \
    --find-links "${wheelhouse}" \
    "${TORCH_SPEC}" "${TORCHVISION_SPEC}" "${TORCHAUDIO_SPEC}" \
    --index-url "${TORCH_INDEX_URL}/cpu"
  rm -rf "${wheelhouse}"
}

install_pytorch_and_domains
