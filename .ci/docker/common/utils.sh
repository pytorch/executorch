#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

as_ci_user() {
  # NB: unsetting the environment variables works around a conda bug
  #     https://github.com/conda/conda/issues/6576
  # NB: Pass on PATH and LD_LIBRARY_PATH to sudo invocation
  # NB: This must be run from a directory that the user has access to
  sudo -E -H -u ci-user env -u SUDO_UID -u SUDO_GID -u SUDO_COMMAND -u SUDO_USER env "PATH=${PATH}" "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" "$@"
}

conda_install() {
  # Ensure that the install command don't upgrade/downgrade Python
  # This should be called as
  #   conda_install pkg1 pkg2 ... [-c channel]
  as_ci_user conda install -q -n "py_${PYTHON_VERSION}" -y python="${PYTHON_VERSION}" "$@"
}

conda_run() {
  as_ci_user conda run -n "py_${PYTHON_VERSION}" --no-capture-output "$@"
}

pip_install() {
  as_ci_user conda run -n "py_${PYTHON_VERSION}" pip install --progress-bar off "$@"
}

init_sccache() {
  # This is the remote cache bucket
  export SCCACHE_BUCKET=ossci-compiler-cache-circleci-v2
  export SCCACHE_S3_KEY_PREFIX=executorch
  export SCCACHE_IDLE_TIMEOUT=0
  export SCCACHE_ERROR_LOG=/tmp/sccache_error.log
  export RUST_LOG=sccache::server=error

  # NB: This function is adopted from PyTorch core at
  # https://github.com/pytorch/pytorch/blob/main/.ci/pytorch/common-build.sh
  as_ci_user sccache --stop-server > /dev/null 2>&1 || true
  rm -f "${SCCACHE_ERROR_LOG}" || true

  # Clear sccache stats before using it
  as_ci_user sccache --zero-stats || true
}
