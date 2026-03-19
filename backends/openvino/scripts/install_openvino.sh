#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Downloads the OpenVINO toolkit archive and sources setupvars.sh to set
# OpenVINO_DIR and related environment variables.

set -ex

OPENVINO_VERSION="2025.3"
OPENVINO_BUILD="2025.3.0.19807.44526285f24"
OPENVINO_ARCHIVE="openvino_toolkit_ubuntu22_${OPENVINO_BUILD}_x86_64"
OPENVINO_URL="https://storage.openvinotoolkit.org/repositories/openvino/packages/${OPENVINO_VERSION}/linux/${OPENVINO_ARCHIVE}.tgz"

install_openvino() {
  # Skip if OpenVINO_DIR is already set and valid
  if [[ -n "${OpenVINO_DIR:-}" && -d "${OpenVINO_DIR:-}" ]]; then
    echo "OpenVINO already set to ${OpenVINO_DIR} - skipping installation"
    return
  fi

  # Skip if already extracted
  if [[ -f "openvino/setupvars.sh" ]]; then
    echo "OpenVINO already extracted at $(pwd)/openvino"
    source openvino/setupvars.sh
    return
  fi

  echo "Downloading OpenVINO ${OPENVINO_VERSION}..."
  curl -Lo /tmp/openvino_toolkit.tgz --retry 3 --fail "${OPENVINO_URL}"

  echo "Extracting OpenVINO archive..."
  tar -xzf /tmp/openvino_toolkit.tgz
  mv "${OPENVINO_ARCHIVE}" openvino
  rm -f /tmp/openvino_toolkit.tgz

  source openvino/setupvars.sh
  echo "OpenVINO_DIR=${OpenVINO_DIR}"
}
