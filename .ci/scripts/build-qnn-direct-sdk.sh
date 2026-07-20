#!/bin/bash
# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux

source "$(dirname "${BASH_SOURCE[0]}")/../../backends/qualcomm/scripts/install_qnn_sdk.sh"

setup_android_ndk
install_qnn
install_hexagon_sdk

bash backends/qualcomm/scripts/build.sh \
    --build_direct_mode 3 --soc_model SM8750 \
    --skip_x86_64 --skip_linux_android \
    --release

ARTIFACT="build-direct/backends/qualcomm/libqnn_executorch_backend.so"
if [ ! -f "${ARTIFACT}" ]; then
    echo "ERROR: direct-mode build did not produce ${ARTIFACT}" >&2
    exit 1
fi

MAX_SIZE_BYTES=$((200 * 1024))
ARTIFACT_SIZE=$(stat -c%s "${ARTIFACT}")
if [ "${ARTIFACT_SIZE}" -gt "${MAX_SIZE_BYTES}" ]; then
    echo "ERROR: ${ARTIFACT} is ${ARTIFACT_SIZE} bytes, exceeds ${MAX_SIZE_BYTES}-byte (200 KiB) limit" >&2
    exit 1
fi
echo "PASSED: direct-mode build produced ${ARTIFACT} (${ARTIFACT_SIZE} bytes, under ${MAX_SIZE_BYTES}-byte limit)"
