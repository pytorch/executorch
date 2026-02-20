#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) Samsung Electronics Co. LTD
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set +ex

if [[ "${DEVICE_ACQUIRED:-0}" != "1" ]]; then
  exit 0
fi

if ! command -v devicefarm-cli >/dev/null 2>&1; then
  echo "[WARN] Skip device disconnect (devicefarm-cli not installed)." >&2
  exit 0
fi

echo "[INFO] Disconnecting device (-d)..."
devicefarm-cli -d || echo "::warning::Device disconnect failed (ignored)"

set -ex
