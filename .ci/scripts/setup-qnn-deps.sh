#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

install_qnn() {
  echo "Start installing qnn."
  QNN_SDK_PATH = "/tmp/qairt/v2.23.0.24.06.24"

  pushd /tmp
  curl -Lo v2.23.0.24.06.24.zip "https://softwarecenter.qualcomm.com/api/download/software/qualcomm_neural_processing_sdk/v2.23.0.24.06.24.zip"
  echo "Finishing downloading qnn sdk."
  unzip -qo "v2.23.0.24.06.24.zip"
  echo "Finishing unzip qnn sdk."


  # Print the content for manual verification
  ls -lah "qairt"
  # mv "qairt"/* "${QNN_INSTALLATION_DIR}"
  echo "Finishing installing qnn /tmp/${QNN_INSTALLATION_DIR} ."

  ls -lah $QNN_SDK_PATH

  export PATH="${PATH}:${QNN_SDK_PATH}"
  ls -lah "${QNN_SDK_PATH}"
  popd
}

install_qnn
