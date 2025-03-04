#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# Double check if the cadence version is set,
# TODO: set and read DSP names from secrets
[ -n "${CADENCE_SDK}" ]
