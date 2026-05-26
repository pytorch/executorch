#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# QNN SDK Configuration
QNN_VERSION="2.37.0.250724"
QNN_ZIP_URL="https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/${QNN_VERSION}/v${QNN_VERSION}.zip"

# Hexagon SDK Configuration (used only by direct-mode CI build).
# HEXAGON_TOOLS_VERSION must match the toolchain shipped inside HEXAGON_SDK_VERSION.
HEXAGON_SDK_VERSION="6.5.0.0"
HEXAGON_TOOLS_VERSION="19.0.07"
HEXAGON_SDK_ZIP_URL="https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/sdks/Hexagon_SDK/Linux/Debian/${HEXAGON_SDK_VERSION}/Hexagon_SDK_Linux.zip"
