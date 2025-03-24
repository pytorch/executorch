#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eu

if [ -z "$QNN_SDK_ROOT" ]; then
  echo "You must specify QNN_SDK_ROOT"
  exit 1
fi

BASEDIR=$(dirname "$0")
ANDROID_ABIS="arm64-v8a" bash "$BASEDIR"/setup.sh

BUILD_AAR_DIR="$(mktemp -d)"
export BUILD_AAR_DIR
