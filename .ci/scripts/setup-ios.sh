#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

# This script follows the instructions from GitHub to install an Apple certificate
# https://docs.github.com/en/actions/use-cases-and-examples/deploying/installing-an-apple-certificate-on-macos-runners-for-xcode-development

CERTIFICATE_PATH="${RUNNER_TEMP}"/build_certificate.p12
PP_PATH="${RUNNER_TEMP}"/build_pp.mobileprovision
KEYCHAIN_PATH="${RUNNER_TEMP}"/app-signing.keychain-db

# Import certificate and provisioning profile from secrets
echo -n "$BUILD_CERTIFICATE_BASE64" | base64 --decode -o $CERTIFICATE_PATH
echo -n "$BUILD_PROVISION_PROFILE_BASE64" | base64 --decode -o $PP_PATH

# Create a temporary keychain
security create-keychain -p "$KEYCHAIN_PASSWORD" $KEYCHAIN_PATH
security set-keychain-settings -lut 21600 $KEYCHAIN_PATH
security unlock-keychain -p "$KEYCHAIN_PASSWORD" $KEYCHAIN_PATH

# Import certificate to the keychain
security import $CERTIFICATE_PATH -P "" -A -t cert -f pkcs12 -k $KEYCHAIN_PATH
security set-key-partition-list -S apple-tool:,apple: -k "$KEYCHAIN_PASSWORD" $KEYCHAIN_PATH
security list-keychain -d user -s $KEYCHAIN_PATH

# Apply provisioning profile
mkdir -p ~/Library/MobileDevice/Provisioning\ Profiles
cp $PP_PATH ~/Library/MobileDevice/Provisioning\ Profiles
