#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

source "$(dirname "${BASH_SOURCE[0]}")/../../backends/openvino/scripts/install_openvino.sh"
install_openvino

cd backends/openvino
pip install -r requirements.txt
cd scripts
./openvino_build.sh --enable_python
