#!/usr/bin/env bash
# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -u
EIQ_PYPI_URL="${EIQ_PYPI_URL:-https://eiq.nxp.com/repository}"

# Install eIQ Neutron dependencies - SDK and simulator
pip install --index-url ${EIQ_PYPI_URL} eiq-neutron-sdk==3.0.1 eiq_nsys

# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Install the required visualization dependencies.
"${SCRIPT_DIR}/../../devtools/install_requirements.sh"

# Install tosa_serializer if needed (required for CortexM dependencies)
# Taken from executorch/.ci/scripts/unittest-linux-cmake.sh
if ! python -c "import tosa_serializer" >/dev/null 2>&1; then
  EXECUTORCH_DIR=$(dirname $(dirname $SCRIPT_DIR))
  cd $EXECUTORCH_DIR
  TOSA_SERIALIZATION_DIR="./examples/arm/arm-scratch/tosa-tools/serialization"
  if [[ ! -d "${TOSA_SERIALIZATION_DIR}" ]]; then
    TOSA_TOOLS_DIR="$(mktemp -d /tmp/tosa-tools.XXXXXX)"
    git clone --depth 1 --branch v2025.11.2 \
      https://git.gitlab.arm.com/tosa/tosa-tools.git "${TOSA_TOOLS_DIR}"
    TOSA_SERIALIZATION_DIR="${TOSA_TOOLS_DIR}/serialization"
  fi

  python -m pip install pybind11==2.10.4
  CMAKE_POLICY_VERSION_MINIMUM=3.5 BUILD_PYBIND=1 \
    python -m pip install --no-dependencies \
    "${TOSA_SERIALIZATION_DIR}"
  python -c "import tosa_serializer"
  cd -
fi
