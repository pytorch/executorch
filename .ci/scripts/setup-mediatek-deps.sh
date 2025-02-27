#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux

MEDIATEK_INSTALLATION_DIR=/tmp/neuropilot
EXECUTORCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

install_neuropilot() {
  echo "Start installing neuropilot."
  mkdir -p "${MEDIATEK_INSTALLATION_DIR}"

  curl -Lo /tmp/neuropilot-express-sdk-8.0.4-build20241016.tar.gz "https://s3.ap-southeast-1.amazonaws.com/mediatek.neuropilot.com/8c1ff4e4-4256-47ed-9e24-67818b4cc4b9.gz"
  echo "Finishing downloading neuropilot sdk."
  tar zxvf /tmp/neuropilot-express-sdk-8.0.4-build20241016.tar.gz --strip-components=1 --directory "${MEDIATEK_INSTALLATION_DIR}"
  echo "Finishing unzip neuropilot sdk."

  # Print the content for manual verification
  ls -lah "${MEDIATEK_INSTALLATION_DIR}"
}

install_android() {
  copy ../docker/common/install_android.sh install_android.sh
  ./install_android.sh
  rm install_android.sh
}

setup_neuropilot() {
  pip3 install -r ${EXECUTORCH_ROOT}/backends/mediatek/requirements.txt
  pip3 install ${MEDIATEK_INSTALLATION_DIR}/mtk_neuron-8.2.13-py3-none-linux_x86_64.whl
  pip3 install ${MEDIATEK_INSTALLATION_DIR}/mtk_converter-8.9.1_public_packages/mtk_converter-8.9.1+public-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
}

setup_calibration_data() {
  curl -Lo /tmp/imagenette2-160.tgz https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
  tar zxvf /tmp/imagenette2-160.tgz --strip-components=1 --directory "${MEDIATEK_INSTALLATION_DIR}"
}

install_android
install_neuropilot
setup_neuropilot
setup_calibration_data
