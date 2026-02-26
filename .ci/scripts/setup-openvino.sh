#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

OPENVINO_SETUP_ROOT=$(pwd)

# Download and install OpenVINO from release packages
OPENVINO_VERSION="2026.0"
OPENVINO_BUILD="2026.0.0.20965.c6d6a13a886"
OPENVINO_URL="https://storage.openvinotoolkit.org/repositories/openvino/packages/${OPENVINO_VERSION}/linux/openvino_toolkit_ubuntu22_${OPENVINO_BUILD}_x86_64.tgz"

curl -Lo /tmp/openvino_toolkit.tgz --retry 3 --fail ${OPENVINO_URL}
tar -xzf /tmp/openvino_toolkit.tgz
mv openvino_toolkit_ubuntu22_${OPENVINO_BUILD}_x86_64 openvino

source openvino/setupvars.sh
cd backends/openvino
pip install -r requirements.txt
cd scripts
./openvino_build.sh --enable_python
cd "${OPENVINO_SETUP_ROOT}"
