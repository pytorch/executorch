#!/usr/bin/env bash
# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -u

# Install neutron-converter
pip install --index-url https://eiq.nxp.com/repository neutron_converter_SDK_25_09

# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Install the required visualization dependencies.
"${SCRIPT_DIR}/../../devtools/install_requirements.sh"
