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

EXECUTORCH_ROOT_PATH="$SCRIPT_DIR_PATH/../../../../"
COREML_DIR_PATH="$EXECUTORCH_ROOT_PATH/backends/apple/coreml"

pip install "$COREML_DIR_PATH/runtime/inmemoryfs"

# clone and install coremltools
if [ -d "/tmp/coremltools" ]; then
    rm -rf "/tmp/coremltools"
fi

git clone git@github.com:DawerG/coremltools.git  /tmp/coremltools

mkdir /tmp/coremltools/build
cmake -S /tmp/coremltools/ -B /tmp/coremltools/build
cmake --build /tmp/coremltools/build --parallel

pip install /tmp/coremltools
