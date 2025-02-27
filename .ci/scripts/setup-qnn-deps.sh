#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

verify_pkg_installed() {
  echo $(dpkg-query -W --showformat='${Status}\n' $1|grep "install ok installed")
}

install_qnn() {
  echo "Start installing qnn."
  QNN_INSTALLATION_DIR=/tmp/qnn
  mkdir -p "${QNN_INSTALLATION_DIR}"

  curl -Lo /tmp/v2.28.0.24.10.29.zip "https://softwarecenter.qualcomm.com/api/download/software/qualcomm_neural_processing_sdk/v2.28.0.241029.zip"
  echo "Finishing downloading qnn sdk."
  unzip -qo /tmp/v2.28.0.24.10.29.zip -d /tmp
  echo "Finishing unzip qnn sdk."


  # Print the content for manual verification
  ls -lah "/tmp/qairt"
  mv "/tmp/qairt"/* "${QNN_INSTALLATION_DIR}"
  echo "Finishing installing qnn '${QNN_INSTALLATION_DIR}' ."

  ls -lah "${QNN_INSTALLATION_DIR}"
}

setup_libc++() {
  clang_version=$1
  sudo apt-get update
  pkgs_to_check=("libc++-${clang_version}-dev")
  j=0
  while [ $j -lt ${#pkgs_to_check[*]} ]; do
    install_status=$(verify_pkg_installed ${pkgs_to_check[$j]})
    if [ "$install_status" == "" ]; then
      sudo apt-get install -y ${pkgs_to_check[$j]}
      if [[ $? -ne 0 ]]; then
        echo "ERROR: Failed to install required packages for libc++"
        exit 1
      fi
    fi
    j=$(( $j +1));
  done
}

# This needs to match with the clang version from the Docker image
setup_libc++ 12
install_qnn
