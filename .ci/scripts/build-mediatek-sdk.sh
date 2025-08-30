#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux

build_neuron_backend() {
  echo "Start building neuron backend."
  export ANDROID_NDK=/opt/ndk
  export MEDIATEK_SDK_ROOT=/tmp/neuropilot
  export NEURON_BUFFER_ALLOCATOR_LIB=${MEDIATEK_SDK_ROOT}/libneuron_buffer_allocator.so
  export EXECUTORCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

  cd ${EXECUTORCH_ROOT}
  ./backends/mediatek/scripts/mtk_build.sh
  ./examples/mediatek/mtk_build_examples.sh
}

build_neuron_backend
