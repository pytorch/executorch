#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eu

BUILD_AAR_DIR="$(mktemp -d)"
export BUILD_AAR_DIR

BASEDIR=$(dirname "$0")
mkdir -p "$BASEDIR"/app/libs
bash "$BASEDIR"/../../../../build/build_android_library.sh

cp "$BUILD_AAR_DIR/executorch.aar" "$BASEDIR"/app/libs/executorch.aar
