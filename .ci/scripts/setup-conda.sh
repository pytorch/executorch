#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

install_conda() {
  pushd .ci/docker || return
  ${CONDA_INSTALL} -c conda-forge -y --file conda-env-ci.txt
  popd || return
}

# Without a conda env, e.g. under _macos_job_venv.yml, install the build tools
# from conda-env-ci.txt with pip. libomp comes from Homebrew in setup-macos.sh,
# the rest is only needed to build PyTorch from source.
install_pip() {
  local packages
  packages=$(sed -n -E 's/^(cmake|ninja)=(.+)$/\1==\2/p' .ci/docker/conda-env-ci.txt)
  # shellcheck disable=SC2086
  python -m pip install --progress-bar off ${packages}
}

if [[ -n "${CONDA_INSTALL:-}" ]]; then
  install_conda
else
  install_pip
fi
