#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Test the end-to-end flow of mps runner.

set -e

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/../../../.ci/scripts/setup-qnn-deps.sh"

setup_libcpp 12
setup_android_ndk
install_qnn
