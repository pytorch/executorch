#!/bin/bash
# Copyright 2024,2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# NB: This function could be used to install Arm dependencies
# Setup arm example environment (including TOSA tools)
# Configure git user only if not already set
if ! git config --get user.name >/dev/null 2>&1; then
    git config --global user.name "Github Executorch"
fi
if ! git config --get user.email >/dev/null 2>&1; then
    git config --global user.email "github_executorch@arm.com"
fi
bash examples/arm/setup.sh --i-agree-to-the-contained-eula ${@:-}
