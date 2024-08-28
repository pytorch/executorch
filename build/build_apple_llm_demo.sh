#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

TOKENIZER="${1:-bpe}"
ARTIFACTS_DIR_NAME="$2"

APP_PATH="examples/demo-apps/apple_ios/LLaMA/LLaMA"

xcodebuild build -project "${APP_PATH}.xcodeproj" \
  -scheme LLaMA \
  -destination platform="iOS" \
  -allowProvisioningUpdates \
  DEVELOPMENT_TEAM=78E7V7QP35 \
  CODE_SIGN_STYLE=Manual \
  PROVISIONING_PROFILE_SPECIFIER=iLLaMA \
  CODE_SIGN_IDENTITY="iPhone Distribution" \
  CODE_SIGNING_REQUIRED=No \
  CODE_SIGNING_ALLOWED=No
