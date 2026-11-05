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

OPENVINO_PACKAGES_URL="https://storage.openvinotoolkit.org/repositories/openvino/packages"
if [ "${USE_NIGHTLY}" = true ]; then
  # Only for quick testing. OpenVINO deletes nightlies after a few months, so
  # take the newest one instead of pinning a build that will disappear.
  OPENVINO_URL=$(curl -fsSL --retry 3 --retry-all-errors "${OPENVINO_PACKAGES_URL}/nightly/latest.txt" | sed -n 's/^ubuntu22_x86_64://p')
  : "${OPENVINO_URL:?could not get the ubuntu22_x86_64 nightly from latest.txt}"
  OPENVINO_SHA256=$(curl -fsSL --retry 3 --retry-all-errors "${OPENVINO_URL}.sha256" | sed -nE 's/^([0-9a-f]{64}).*/\1/p')
  : "${OPENVINO_SHA256:?could not get the SHA-256 of the nightly from its .sha256 file}"
else
  OPENVINO_VERSION="2026.4.1"
  OPENVINO_BUILD="2026.4.1.22982.07f9c262b05"
  OPENVINO_SHA256="e2541afe65a81d5aabe981f9a86ad2c7efe557922b71e5649b6aafffec7c982b"
  OPENVINO_URL="${OPENVINO_PACKAGES_URL}/${OPENVINO_VERSION}/linux/openvino_toolkit_ubuntu22_${OPENVINO_BUILD}_x86_64.tgz"
fi
OPENVINO_EXTRACTED_DIR=$(basename "${OPENVINO_URL}" .tgz)
echo "Using OpenVINO: ${OPENVINO_EXTRACTED_DIR}"

curl -Lo /tmp/openvino_toolkit.tgz --retry 3 --retry-all-errors --fail "${OPENVINO_URL}"
# The storage server answers a missing file with an HTML page and HTTP 200, so
# compare the download with the expected hash before extracting it.
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