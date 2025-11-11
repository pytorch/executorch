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

  curl -Lo /tmp/neuropilot-express.tar.gz "https://s3.ap-southeast-1.amazonaws.com/mediatek.neuropilot.com/06302508-4c94-4bf2-9789-b0ee44e83e27.gz"
  echo "Finishing downloading neuropilot sdk."
  tar zxvf /tmp/neuropilot-express.tar.gz --strip-components=1 --directory "${MEDIATEK_INSTALLATION_DIR}"
  echo "Finishing unzip neuropilot sdk."

  # Copy NP header
  cp ${MEDIATEK_INSTALLATION_DIR}/api/NeuronAdapter.h ${EXECUTORCH_ROOT}/backends/mediatek/runtime/include/api/

  # Print the content for manual verification
  ls -lah "${MEDIATEK_INSTALLATION_DIR}"
}

setup_neuropilot() {
  pip3 install -r ${EXECUTORCH_ROOT}/backends/mediatek/requirements.txt
  pip3 install ${MEDIATEK_INSTALLATION_DIR}/mtk_neuron-8.2.19-py3-none-linux_x86_64.whl
  pip3 install ${MEDIATEK_INSTALLATION_DIR}/mtk_converter-8.13.0_public_packages/mtk_converter-8.13.0+public-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
}

setup_calibration_data() {
  curl -Lo /tmp/imagenette2-160.tgz https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
  tar zxvf /tmp/imagenette2-160.tgz --strip-components=1 --directory "${MEDIATEK_INSTALLATION_DIR}"
}

install_neuropilot
setup_neuropilot
setup_calibration_data
