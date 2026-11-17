#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

source openvino/setupvars.sh
cd backends/openvino/tests
# Run the model tests even when an op test fails, and still fail the job.
status=0
python test_runner.py --test_type ops || status=1
python test_runner.py --test_type models || status=1
exit "${status}"
