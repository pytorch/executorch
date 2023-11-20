#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

install_miniconda() {
  BASE_URL="https://repo.anaconda.com/miniconda"
  CONDA_FILE="Miniconda3-py${PYTHON_VERSION//./}_${MINICONDA_VERSION}-Linux-x86_64.sh"

  mkdir -p /opt/conda
  chown ci-user:ci-user /opt/conda

  pushd /tmp
  wget -q "${BASE_URL}/${CONDA_FILE}"
  # Install miniconda
  as_ci_user bash "${CONDA_FILE}" -b -f -p "/opt/conda"
  # Clean up the download file
  rm "${CONDA_FILE}"
  popd

  sed -e 's|PATH="\(.*\)"|PATH="/opt/conda/bin:\1"|g' -i /etc/environment
  export PATH="/opt/conda/bin:$PATH"
}

install_python() {
  pushd /opt/conda
  # Install the correct Python version
  as_ci_user conda create -n "py_${PYTHON_VERSION}" -y --file /opt/conda/conda-env-ci.txt python="${PYTHON_VERSION}"
  popd
}

install_pip_dependencies() {
  pushd /opt/conda
  # Install all Python dependencies, including PyTorch
  pip_install -r /opt/conda/requirements-ci.txt
  popd
}

install_miniconda
install_python
install_pip_dependencies
