#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# This script is originally adopted from https://github.com/pytorch/pytorch/blob/main/android/run_tests.sh
ADB_PATH=$ANDROID_HOME/platform-tools/adb

echo "List all running emulators"
$ADB_PATH devices

# These devices should have already been created by ReactiveCircus/android-emulator-runner GHA
DEVICES_COUNT=$($ADB_PATH devices | awk 'NF' | wc -l)
echo "DEVICES_COUNT:$DEVICES_COUNT"

if [ "$DEVICES_COUNT" -eq 1 ]; then
  echo "Unable to found connected android emulators"
  exit 1
fi

echo "Waiting for emulator boot completed"
$ADB_PATH wait-for-device shell 'while [[ -z $(getprop sys.boot_completed) ]]; do sleep 1; done;'
