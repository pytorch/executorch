#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

install_qnn() {
  echo "Start installing qnn."
  QNN_INSTALLATION_DIR=/pytorch/qnn
  mkdir -p "${QNN_INSTALLATION_DIR}"

  pushd /tmp
  curl -Os --retry 3 "https://softwarecenter.qualcomm.com/api/download/software/qualcomm_neural_processing_sdk/v2.23.0.24.06.24.zip"
  echo "Finishing downloading qnn sdk."
  unzip -qo "v2.23.0.24.06.24.zip"
  echo "Finishing unzip qnn sdk."


  pushd /tmp
  # The NDK installation is cached on ossci-android S3 bucket
  curl -Os --retry 3 "https://softwarecenter.qualcomm.com/api/download/software/qualcomm_neural_processing_sdk/v2.23.0.24.06.24.zip"
  unzip -qo "v2.23.0.24.06.24.zip"

  # Print the content for manual verification
  ls -lah "qairt"
  mv "qairt"/* "${NDK_INSTALLATION_DIR}"
  echo "Finishing installing qnn."

  popd
}

install_qnn
