#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"
install_executorch
source openvino/dist/setupvars.sh
cd backends/openvino/tests
python test_runner.py --test_type ops
python test_runner.py --test_type models
