#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

install_conda() {
  pushd .ci/docker || return

  # The env created by pytorch/test-infra's setup-miniconda is pre-populated
  # with cmake=3.22 ninja=1.10 pkg-config=0.29 wheel=0.37 from the anaconda
  # defaults channel. Mixing those with conda-forge's newer transitive deps
  # required by our cmake=3.31.2 pin (libzlib>=1.3.1, rhash>=1.4.5) has been
  # intermittently failing the libmamba solver, especially on ephemeral
  # GitHub-hosted macOS runners where the env is fresh every job. Tear down
  # the pre-populated env and recreate it from conda-forge only so the solve
  # never has to reconcile two channels.
  PY_VERSION=$(conda run -p "${CONDA_PREFIX}" python -c \
    'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

  conda env remove --prefix "${CONDA_PREFIX}" -y
  conda create --prefix "${CONDA_PREFIX}" -c conda-forge --override-channels -y \
    "python=${PY_VERSION}" --file conda-env-ci.txt

  popd || return
}

install_conda
