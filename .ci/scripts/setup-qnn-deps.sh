#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

install_qnn() {
  echo "Start installing qnn."
  QNN_INSTALLATION_DIR=/tmp/qnn
  mkdir -p "${QNN_INSTALLATION_DIR}"

  curl -Lo /tmp/v2.23.0.24.06.24.zip "https://softwarecenter.qualcomm.com/api/download/software/qualcomm_neural_processing_sdk/v2.23.0.24.06.24.zip"
  echo "Finishing downloading qnn sdk."
  unzip -qo /tmp/v2.23.0.24.06.24.zip -d /tmp
  echo "Finishing unzip qnn sdk."


  # Print the content for manual verification
  ls -lah "/tmp/qairt"
  mv "/tmp/qairt"/* "${QNN_INSTALLATION_DIR}"
  echo "Finishing installing qnn '${QNN_INSTALLATION_DIR}' ."

  export ANDROID_NDK=/opt/ndk
  export QNN_SDK_ROOT=/tmp/qnn/2.23.0.240531
  export EXECUTORCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

  ls -lah "${QNN_INSTALLATION_DIR}"
}

install_qnn
