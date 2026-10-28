#!/usr/bin/env bash
# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

# Build and flash the camera example from an activated ESP-IDF shell. A running
# viewer.py releases the serial port while this script flashes.
set -euo pipefail
PORT=${1:-${ESPPORT:?usage: flash.sh PORT (or set ESPPORT)}}
cd "$(dirname "$0")"
mkdir -p build
idf.py ${ET_BUILD_DIR:+-DET_BUILD_DIR_PATH="$ET_BUILD_DIR"} build
touch build/.viewer_pause
trap 'rm -f build/.viewer_pause' EXIT
sleep 1
idf.py -p "$PORT" -b 921600 flash
