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

  # NB: This function is adopted from PyTorch core at
  # https://github.com/pytorch/pytorch/blob/main/.ci/pytorch/common-build.sh
  if which sccache > /dev/null; then
    # Save sccache logs to file
    as_ci_user sccache --stop-server > /dev/null 2>&1 || true
    rm -f ~/sccache_error.log || true

    function sccache_epilogue() {
      echo "::group::Sccache Compilation Log"
      echo '=================== sccache compilation log ==================='
      cat ~/sccache_error.log || true
      echo '=========== If your build fails, please take a look at the log above for possible reasons ==========='
      as_ci_user sccache --show-stats
      as_ci_user sccache --stop-server || true
      echo "::endgroup::"
    }

    # Register the function here so that the error log can be printed even when
    # sccache fails to start, i.e. timeout error
    trap_add sccache_epilogue EXIT

    export SCCACHE_IDLE_TIMEOUT=0
    export SCCACHE_ERROR_LOG=~/sccache_error.log
    export RUST_LOG=sccache::server=error

    # Clear sccache stats before using it
    as_ci_user sccache --zero-stats || true
  fi
}
