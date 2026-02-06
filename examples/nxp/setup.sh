#!/usr/bin/env bash
# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -u
EIQ_PYPI_URL=https://eiq.nxp.com/repository


# Install neutron-converter
pip install --index-url ${EIQ_PYPI_URL} neutron_converter_SDK_25_12

# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Install the required visualization dependencies.
"${SCRIPT_DIR}/../../devtools/install_requirements.sh"
