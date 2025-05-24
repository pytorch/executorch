#!/usr/bin/env bash
# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -u

retry_command() {
    local max_attempts=$2
    for ((i=0; i<$max_attempts; i++)); do
        $1 && return 0 || echo "Attempt $((i+1)) failed. Retrying..."
        sleep 2
    done
    echo "All attempts failed."
    return 1
}

# Install neutron-converter
retry_command "pip install --extra-index-url https://eiq.nxp.com/repository neutron-converter_SDK_25_03" 100 || { echo "Failed"; exit 1; }
