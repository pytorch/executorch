#!/usr/bin/env bash
#
# Copyright © 2023 Apple Inc. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

SCRIPT_DIR_PATH="$(
    cd -- "$(dirname "$0")" >/dev/null 2>&1
    pwd -P
)"

EXECUTORCH_ROOT_PATH="$SCRIPT_DIR_PATH/../../../"
COREML_DIR_PATH="$EXECUTORCH_ROOT_PATH/backends/coreml"

# Install required python dependencies.
# Dependencies are defined in backends/coreml/pyproject.toml
pip install "$COREML_DIR_PATH"

check_coremltools() {
    python -c "import coremltools"
    if [[ $? -ne 0 ]]; then
        RED='\033[0;31m'
        NC='\033[0m' # NO Color
        echo
        echo "${RED}Please clone coremltools and checkout branch 'executorch_integration'. Then install coremltools using 'pip install -e .' ${NC}"
        echo 
        exit
    fi
}

check_coremltools

pip install "$COREML_DIR_PATH/runtime/inmemoryfs"
