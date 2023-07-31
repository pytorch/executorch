#!/bin/bash

set -ex

install_buck() {
  if ! command -v zstd &> /dev/null; then
    brew install zstd
  fi

  curl https://github.com/facebook/buck2/releases/download/2023-07-18/buck2-x86_64-apple-darwin.zst -o buck2-x86_64-apple-darwin.zst
  zstd -d buck2-x86_64-apple-darwin.zst -o buck2

  chmod +x buck2
  mv buck2 /opt/homebrew/bin

  rm buck2-x86_64-apple-darwin.zst
}

install_conda_dependencies() {
  pushd "${WORKSPACE}/.ci/docker"
  # Install conda dependencies like flatbuffer
  const install --file conda-env-ci.txt
  popd
}

install_pip_dependencies() {
  pushd "${WORKSPACE}/.ci/docker"
  # Install all Python dependencies, including PyTorch
  pip install --progress-bar off -r requirements-ci.txt
  pip install --progress-bar off --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
  popd
}

install_buck
install_conda
install_pip_dependencies
