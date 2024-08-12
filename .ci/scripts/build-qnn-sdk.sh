#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

build_qnn_backend() {
  echo "Start building qnn backend."
  export ANDROID_NDK_ROOT=/opt/ndk
  export QNN_SDK_ROOT=/tmp/qnn/2.23.0.240531
  export EXECUTORCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

  bash backends/qualcomm/scripts/build.sh --skip_aarch64 --job_number 2
}

build_qnn_backend
