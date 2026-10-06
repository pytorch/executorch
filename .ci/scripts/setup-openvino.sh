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

# OpenVINO deletes old nightly builds, so pin a release that contains what the
# test harness needs instead of a nightly.
OPENVINO_NIGHTLY_VERSION="2026.1"
OPENVINO_NIGHTLY_BUILD="2026.1.0.21367.63e31528c62"
OPENVINO_NIGHTLY_URL="https://storage.openvinotoolkit.org/repositories/openvino/packages/${OPENVINO_NIGHTLY_VERSION}/linux/openvino_toolkit_ubuntu22_${OPENVINO_NIGHTLY_BUILD}_x86_64.tgz"

if [ "${USE_NIGHTLY}" = true ]; then
  OPENVINO_URL="${OPENVINO_NIGHTLY_URL}"
  OPENVINO_EXTRACTED_DIR="openvino_toolkit_ubuntu22_${OPENVINO_NIGHTLY_BUILD}_x86_64"
  echo "Using OpenVINO release: ${OPENVINO_NIGHTLY_BUILD}"
else
  OPENVINO_URL="${OPENVINO_STABLE_URL}"
  OPENVINO_EXTRACTED_DIR="openvino_toolkit_ubuntu22_${OPENVINO_BUILD}_x86_64"
  echo "Using OpenVINO stable release: ${OPENVINO_BUILD}"
fi

curl -Lo /tmp/openvino_toolkit.tgz --retry 3 --retry-all-errors --fail ${OPENVINO_URL}
# The storage server answers a missing file with an HTML page and HTTP 200, so
# check the archive against its published checksum before extracting it.
curl -Lo /tmp/openvino_toolkit.tgz.sha256 --retry 3 --retry-all-errors --fail ${OPENVINO_URL}.sha256
echo "$(cut -d ' ' -f 1 /tmp/openvino_toolkit.tgz.sha256)  /tmp/openvino_toolkit.tgz" | sha256sum --check
tar -xzf /tmp/openvino_toolkit.tgz
mv "${OPENVINO_EXTRACTED_DIR}" openvino

set +u
source openvino/setupvars.sh
set -u
pip install -r backends/openvino/requirements.txt
pushd backends/openvino/scripts
./openvino_build.sh --enable_python
popd