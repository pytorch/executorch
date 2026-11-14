#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) Samsung Electronics Co. LTD
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set +ex

if [[ "${DEVICE_ACQUIRED:-0}" != "1" && "${DEVICE_QUEUED:-0}" != "1" ]]; then
  if [[ "${DEVICE_REQUESTED:-0}" == "1" ]]; then
    echo "::warning::Enqueue outcome is unknown; no confirmed request to cancel. Check the device queue." >&2
  fi
  exit 0
fi

if ! command -v devicefarm-cli >/dev/null 2>&1; then
  echo "[WARN] Skip device disconnect (devicefarm-cli not installed)." >&2
  exit 0
fi

echo "[INFO] Cancelling device request / disconnecting device (-d)..."
python "$(dirname "${BASH_SOURCE[0]}")/samsung-device-command.py" 30 devicefarm-cli -d || echo "::warning::Device cleanup failed; remote queue/device state is unknown (ignored)"

set -ex
