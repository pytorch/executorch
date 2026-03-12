#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

# Parse arguments
USE_NIGHTLY=false
for arg in "$@"; do
  case $arg in
    --nightly) USE_NIGHTLY=true ;;
  esac
done

# Download and install OpenVINO from release packages
OPENVINO_VERSION="2026.0"
OPENVINO_BUILD="2026.0.0.20965.c6d6a13a886"
OPENVINO_STABLE_URL="https://storage.openvinotoolkit.org/repositories/openvino/packages/${OPENVINO_VERSION}/linux/openvino_toolkit_ubuntu22_${OPENVINO_BUILD}_x86_64.tgz"

OPENVINO_NIGHTLY_BUILD_ID="2026.1.0-21296-4589d335731"
OPENVINO_NIGHTLY_BUILD="2026.1.0.dev20260311"
OPENVINO_NIGHTLY_URL="https://storage.openvinotoolkit.org/repositories/openvino/packages/nightly/${OPENVINO_NIGHTLY_BUILD_ID}/openvino_toolkit_ubuntu22_${OPENVINO_NIGHTLY_BUILD}_x86_64.tgz"

if [ "${USE_NIGHTLY}" = true ]; then
  OPENVINO_URL="${OPENVINO_NIGHTLY_URL}"
  OPENVINO_EXTRACTED_DIR="openvino_toolkit_ubuntu22_${OPENVINO_NIGHTLY_BUILD}_x86_64"
  echo "Using OpenVINO nightly build: ${OPENVINO_NIGHTLY_BUILD_ID}"
else
  OPENVINO_URL="${OPENVINO_STABLE_URL}"
  OPENVINO_EXTRACTED_DIR="openvino_toolkit_ubuntu22_${OPENVINO_BUILD}_x86_64"
  echo "Using OpenVINO stable release: ${OPENVINO_BUILD}"
fi

curl -Lo /tmp/openvino_toolkit.tgz --retry 3 --fail ${OPENVINO_URL}
tar -xzf /tmp/openvino_toolkit.tgz
mv "${OPENVINO_EXTRACTED_DIR}" openvino

set +u
source openvino/setupvars.sh
set -u
pip install -r backends/openvino/requirements.txt
pushd backends/openvino/scripts
./openvino_build.sh --enable_python
popd