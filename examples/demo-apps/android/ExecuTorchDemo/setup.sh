#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eu

BASEDIR=$(dirname "$0")
mkdir -p "$BASEDIR"/app/libs
curl -o "$BASEDIR"/app/libs/executorch.aar https://ossci-android.s3.amazonaws.com/executorch/release/v0.5.0-rc3/executorch.aar
