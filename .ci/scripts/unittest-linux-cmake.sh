#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -eux

# Some ARM/TOSA-adjacent tests import modules that require tosa_serializer.
# Install from a local tosa-tools checkout when available. If absent in this
# checkout layout, clone the pinned upstream tag and install from there.
if ! python -c "import tosa_serializer" >/dev/null 2>&1; then
  TOSA_SERIALIZATION_DIR="./examples/arm/arm-scratch/tosa-tools/serialization"
  if [[ ! -d "${TOSA_SERIALIZATION_DIR}" ]]; then
    TOSA_TOOLS_DIR="$(mktemp -d /tmp/tosa-tools.XXXXXX)"
    git clone --depth 1 --branch v2025.11.0 \
      https://git.gitlab.arm.com/tosa/tosa-tools.git "${TOSA_TOOLS_DIR}"
    TOSA_SERIALIZATION_DIR="${TOSA_TOOLS_DIR}/serialization"
  fi

  CMAKE_POLICY_VERSION_MINIMUM=3.5 BUILD_PYBIND=1 \
    python -m pip install --no-dependencies \
    "${TOSA_SERIALIZATION_DIR}"
  python -c "import tosa_serializer"
fi

# Run pytest with coverage
pytest -n auto --cov=./ --cov-report=xml
# Run gtest
LLVM_PROFDATA=llvm-profdata-12 LLVM_COV=llvm-cov-12 \
test/run_oss_cpp_tests.sh
