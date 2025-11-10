#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This script is used to build unit tests internally. It is not intended to be used in OSS.

set -xueo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${THIS_DIR}/__tests__"
TEST_MODEL_DIR="${THIS_DIR}/test_models/"

if [[ "${THIS_DIR}" != *"xplat"* ]]; then
    echo "This script must be run from xplat, not fbcode or oss"
    exit 1
fi

ENABLE_ETDUMP=0
for arg in "$@"; do
    if [[ "$arg" == "etdump" ]]; then
        ENABLE_ETDUMP=1
    else
        echo "Unknown argument: $arg"
        exit 1
    fi
done

# Build the models
# Using fbcode because the Python scripts are only supported in fbcode
MODEL_TARGET_DIR=$(buck2 build fbcode//executorch/extension/wasm/test:models --show-full-output | awk '{print $2}')

mkdir -p "${TEST_MODEL_DIR}"
cp ${MODEL_TARGET_DIR}/*.pte ${TEST_MODEL_DIR}

if (( ENABLE_ETDUMP != 0 )); then
    ETDUMP_OPTIONS="-DET_EVENT_TRACER_ENABLED=1"
    WASM_TARGET_NAME="wasm_etdump.test"
else
    ETDUMP_OPTIONS=""
    WASM_TARGET_NAME="wasm.test"
fi

# Emscripten build options don't work properly on fbcode; copy test artifacts to xplat and run the test in xplat.
BUILD_TARGET_DIR=$(dirname $(buck2 build :${WASM_TARGET_NAME}.js --show-full-output -c "cxx.extra_cxxflags=-DET_LOG_ENABLED=0 $ETDUMP_OPTIONS" | awk '{print $2}'))

mkdir -p "${OUTPUT_DIR}"
cp ${BUILD_TARGET_DIR}/${WASM_TARGET_NAME}.js ${OUTPUT_DIR}
cp ${BUILD_TARGET_DIR}/${WASM_TARGET_NAME}.wasm ${OUTPUT_DIR}

yarn install
