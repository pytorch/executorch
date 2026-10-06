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

# Download and install OpenVINO from release packages. --nightly selects a newer
# release rather than a nightly build, because OpenVINO deletes old nightlies.
if [ "${USE_NIGHTLY}" = true ]; then
  OPENVINO_VERSION="2026.1"
  OPENVINO_BUILD="2026.1.0.21367.63e31528c62"
  OPENVINO_SHA256="3b4d92fec96860dfea844cd7c23e190d76c243e75815491d53405b4ced892103"
else
  OPENVINO_VERSION="2026.0"
  OPENVINO_BUILD="2026.0.0.20965.c6d6a13a886"
  OPENVINO_SHA256="3c99a294d1a12a96a945c56a3b27a099df885980f7667b996cff67b4ff3cf46a"
fi
OPENVINO_EXTRACTED_DIR="openvino_toolkit_ubuntu22_${OPENVINO_BUILD}_x86_64"
OPENVINO_URL="https://storage.openvinotoolkit.org/repositories/openvino/packages/${OPENVINO_VERSION}/linux/${OPENVINO_EXTRACTED_DIR}.tgz"
echo "Using OpenVINO release: ${OPENVINO_BUILD}"

curl -Lo /tmp/openvino_toolkit.tgz --retry 3 --retry-all-errors --fail ${OPENVINO_URL}
# The storage server answers a missing file with an HTML page and HTTP 200, so
# compare the download with the pinned hash before extracting it.
echo "${OPENVINO_SHA256}  /tmp/openvino_toolkit.tgz" | sha256sum -c -
tar -xzf /tmp/openvino_toolkit.tgz
mv "${OPENVINO_EXTRACTED_DIR}" openvino

set +u
source openvino/setupvars.sh
set -u
pip install -r backends/openvino/requirements.txt
pushd backends/openvino/scripts
./openvino_build.sh --enable_python
popd