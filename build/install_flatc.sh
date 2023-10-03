#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file is meant to build flatc from the `third-party/flatbuffers` folder and help
# users to ensure it's installed correctly

ROOT=$(cd -- "$(dirname "$(dirname -- "${BASH_SOURCE[0]}")")" &> /dev/null && pwd)
FLATBUFFERS_PATH="$ROOT/third-party/flatbuffers"

# Get the flatbuffers version from the git submodule
get_flatbuffers_version(){
    pushd "$FLATBUFFERS_PATH" || exit
    FLATBUFFERS_VERSION=$(git describe --tags "$(git rev-list --tags --max-count=1)" | sed 's/^v//')
    popd || exit
}

FLATC_PATH="$(which flatc 2>&1)"

GREEN='\033[0;32m' # GREEN Color
RED='\033[0;31m' # Red Color
NC='\033[0m' # No Color

get_flatbuffers_version
echo "flatbuffers version in $FLATBUFFERS_PATH is: $FLATBUFFERS_VERSION"

# Warn the users to uninstall flatc from conda
if [[ "$FLATC_PATH" == *miniconda3* ]]; then
    echo "${RED}From the flatc path: $FLATC_PATH, it seems it's installed with conda. The flatc from conda is problematic and please avoid using it."
    echo "Please run the following lines to remove it: "
    echo "    conda uninstall flatbuffers"
    exit
fi

# Check the following:
# 1. the path to the flatc command if it exists in the system's PATH
# 2. execute the flatc command and check the version
if { command -v flatc; } && { flatc --version | grep "$FLATBUFFERS_VERSION" &> /dev/null; }; then
    echo "${GREEN}flatc is installed successfully and ready to use. ${NC}"
else
    # Check the existence of flatc in flatbuffers path, if it exists, instruct users to export it to path
    if [[ -f "$FLATBUFFERS_PATH/flatc" ]]; then
        echo "${RED}flatc is built in $FLATBUFFERS_PATH/flatc but hasn't added to path, run following lines: "
        echo " export PATH=$FLATBUFFERS_PATH:\$PATH ${NC}"
    else
        # flatc is not in flatbuffers path, build it and instruct users to export it to path
        pushd "$FLATBUFFERS_PATH" || exit
        cmake --build . --target flatc
        echo "flatc path: $FLATBUFFERS_PATH/flatc"
        popd || exit
        echo "${RED}Finished building flatc. Run the following lines to add it to PATH, then re-run this script:"
        echo " export PATH=\"$FLATBUFFERS_PATH:\$PATH\" "
        echo " bash build/install_flatc.sh${NC}"
    fi
fi
